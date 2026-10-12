"""Futures jobs on a worker: the training numbers model search scores with, the
backtest of one futures model (Run backtest), and the once-only Final check.

Which prices each one may read (the coordinator serves nothing later than asked for,
and the lockbox only to a running Final check):
- model search: training to choose; held-out only for a model it has already chosen
- Run backtest: training and held-out
- Final check: everything up to the end of the lockbox, for that one model, once

params (filled in by the coordinator): {"model_id", "module", "params", "market":
"futures", "rules": Rules as a dict, "periods": {"train_end", "held_out_end",
"lockbox_end", ...}} and for a Final check also "contracts".
"""
from __future__ import annotations

import math
from datetime import date
from types import ModuleType
from typing import Any, Callable

import numpy as np

from fleet2.common import http
from fleet2.models.futures import module_for
from fleet2.models.futures.base import params_with_defaults
from fleet2.sim import futures_backtest as fb
from fleet2.sim import futures_stats, topstep
from fleet2.sim.control import JobStopped
from fleet2.sim.futures_data import FuturesData, PriceCache

MIN_DAYS_TRADED = 100  # a model must trade on at least this many training days to be kept
TRAINING_PARTS = 4
EXPRESS_ACCOUNTS = 3000  # most Express accounts played per candidate (simulate_sizes)


def day_int(iso: str) -> int:
    d = date.fromisoformat(iso)
    return d.year * 10000 + d.month * 100 + d.day


def first_day_after(data: FuturesData, iso: str) -> int:
    """Index of the first trading day after the date `iso` (the next period's first day)."""
    return data.day_index(day_int(iso), side="right")


def training_numbers(train: FuturesData, module: ModuleType, params: dict[str, Any], rules: topstep.Rules,
                     keep_daily: bool = False, should_stop: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Model search's score and gates, from training prices only, at one contract.

    Score: training is split into 4 back-to-back parts; in each, the Sharpe ratio of
    daily P&L at DOUBLE slippage; the score is the worst of the four, so a setting that
    only worked in one stretch scores low. Gates: trades on at least 100 different days,
    and a profit at normal and at double slippage (an average hold under a day is true
    by construction: the backtester is flat every night, and it is checked anyway).

    Combine (only for a candidate that passed the other gates, beside the normal-slippage
    run, and with keep_daily): "combine" is training_combine(), how the training days
    would have done in Combines at the best contract size, and "combine_score" its one
    number for picking the best of a model file's candidates (see training_combine).
    Both are None for a candidate that failed a gate first."""
    params = params_with_defaults(module, params)
    costs = topstep.costs_of(rules)
    end_of_day = topstep.day_rules(rules)
    targets = fb.model_targets(train, module, params)
    run = fb.run(train, module, params, costs.doubled(), 1, end_of_day, should_stop=should_stop, targets=targets)
    double = run.pnl
    parts = [fb.daily_sharpe(p) for p in np.array_split(double, TRAINING_PARTS)]
    score = min(p if p is not None else -np.inf for p in parts) if parts else -np.inf
    days_traded = int(np.count_nonzero(run.trades))
    holds = run.trade_list["exit_t"] - run.trade_list["entry_t"]
    gates = {
        "days_traded": days_traded >= MIN_DAYS_TRADED,
        "profit_double": float(double.sum()) > 0,
        "within_the_day": bool((holds < 86400).all()),
        "profit_normal": False,
    }
    out: dict[str, Any] = {
        "score": float(score) if np.isfinite(score) else None,
        "part_sharpes": parts,
        "sharpe": fb.daily_sharpe(double),
        **futures_stats.shape(double),
        "days_traded": days_traded,
        "pnl_normal": None,
        "pnl_double": round(float(double.sum()), 2),
        "worst_stretch": fb.worst_stretch(double, run.dip),
        "start": int(run.days[0]) if run.days.size else None,
        "end": int(run.days[-1]) if run.days.size else None,
        "combine": None,
        "combine_score": None,
    }
    # The normal-slippage run only matters for a candidate that passed everything else.
    # Stops sit a set number of ticks from the actual fill, so slippage can move them:
    # it is a backtest of its own, not the double one with costs taken back off.
    if keep_daily or all(v for k, v in gates.items() if k != "profit_normal"):
        normal = fb.run(train, module, params, costs, 1, end_of_day, should_stop=should_stop, targets=targets)
        gates["profit_normal"] = float(normal.pnl.sum()) > 0
        out["pnl_normal"] = round(float(normal.pnl.sum()), 2)
        out["worst_stretch"] = fb.worst_stretch(normal.pnl, normal.dip)
        if keep_daily:
            out["daily"] = {"days": [int(d) for d in normal.days], "pnl": [round(float(x), 2) for x in normal.pnl]}
            out["numbers"] = fb.summarize(normal)
        out["combine"] = training_combine(train, module, params, rules, normal, targets, should_stop)
        out["combine_score"] = out["combine"]["score"]
    out["gates"] = gates
    out["passes"] = all(gates.values())
    return out


def whole_targets(targets: tuple[Any, dict[str, np.ndarray]]) -> bool:
    """True when every target is a whole number (-1, 0 or +1). Then a backtest at k
    contracts is exactly k times the one at one contract, day by day: P&L, dips, slippage
    and fees all grow with the contracts, and stops sit at the same prices."""
    return all(np.array_equal(a, np.rint(a)) for a in targets[1].values())


def combine_floor(rules: topstep.Rules) -> float:
    """The lowest combine_score there is: what an attempt costs at worst when it is never
    paid (every month of Combine fees, and the activation fee), or 0 passes a month
    while a fee or the payout cap is unset."""
    if topstep.money_note(rules) is not None:
        return 0.0
    return -float(rules.combine_monthly * math.ceil(rules.combine_max_days / rules.days_per_month) + rules.activation)


def simulate_sizes(days: list[tuple[np.ndarray, np.ndarray]], rules: topstep.Rules,
                   express_cap: int | None = EXPRESS_ACCOUNTS) -> list[dict[str, Any]]:
    """topstep.simulate()'s numbers for several versions of one period (the same days at
    different contract sizes), with the simulator run once for all of them: the versions
    are laid end to end with express_days empty days between them. An attempt only
    starts where combine_max_days days of its own version are left, and an Express
    account that reaches the gap has nothing left to win or lose there, so nothing
    crosses from one version into the next (the tests check it against simulate()).
    Also the mean days to pass. With payouts on days of 0 dollars or more, each version
    is simulated on its own.

    Attempts that pass on the same day open the same Express account, played once. When
    there are more than express_cap of those, only every so many is played, each standing
    for the next few that would open on the days after it ("express_every" says how
    many): the pass numbers stay exact, the money becomes an estimate."""
    n = days[0][0].shape[0] if days else 0
    if rules.payout_min_day_profit <= 0:
        return [{**_with_mean_days(topstep.simulate(p, d, rules), p, d, rules), "express_every": 1} for p, d in days]
    gap = rules.express_days
    span = n + gap
    pnl = np.zeros(span * len(days))
    dip = np.zeros(span * len(days))
    for i, (p, d) in enumerate(days):
        pnl[i * span:i * span + n], dip[i * span:i * span + n] = p, d
    c = topstep.combine(pnl, dip, rules)
    starts = max(0, n - rules.combine_max_days + 1)
    blocks = [slice(i * span, i * span + starts) for i in range(len(days))]
    begins = [c["end"][b][c["state"][b] == 1] + 1 for b in blocks]
    paid, every = None, 1
    if rules.max_payout is not None and starts:
        unique, which = np.unique(np.concatenate(begins), return_inverse=True)
        paid_unique = np.zeros(unique.size)
        if unique.size:
            every = max(1, math.ceil(unique.size / express_cap)) if express_cap else 1
            at = np.arange(unique.size)
            block = unique // span
            first = np.maximum.accumulate(np.where(np.r_[True, block[1:] != block[:-1]], at, 0))
            played = (at - first) % every == 0  # every so many, always the first of each version
            paid_unique[played] = topstep.express(pnl, dip, unique[played], rules)["paid"]
            paid_unique = paid_unique[np.maximum.accumulate(np.where(played, at, 0))]
        paid = np.split(paid_unique[which], np.cumsum([len(x) for x in begins])[:-1])
    out = []
    for i, b in enumerate(blocks):
        state, used = c["state"][b], c["days"][b]
        passed = state == 1
        out.append({
            "attempts": starts, "passed": int(passed.sum()), "failed": int((state == 2).sum()),
            "gave_up": int((state == 3).sum()),
            "pass_rate": float(passed.sum() / starts) if starts else None,
            "fail_rate": float((state == 2).sum() / starts) if starts else None,
            "median_days_to_pass": float(np.median(used[passed])) if passed.any() else None,
            "mean_days_to_pass": float(used[passed].mean()) if passed.any() else None,
            "mean_months": float(np.mean(np.ceil(used / rules.days_per_month))) if starts else None,
            "mean_paid": float(paid[i].sum() / starts) if paid is not None else None,
            "express_every": every,
        })
    return out


def _with_mean_days(sim: dict[str, Any], pnl: np.ndarray, dip: np.ndarray, rules: topstep.Rules) -> dict[str, Any]:
    c = topstep.combine(pnl, dip, rules)
    passed = c["state"] == 1
    return {**sim, "mean_days_to_pass": float(c["days"][passed].mean()) if passed.any() else None}


def combine_numbers(pnl: np.ndarray, dip: np.ndarray, rules: topstep.Rules,
                    at_size: Callable[[int], tuple[np.ndarray, np.ndarray]] | None = None) -> dict[str, Any]:
    """Topstep Combines started on every training day that has combine_max_days days
    left, at each contract size from 1 up to the size the held-out test would try
    (topstep.size_limit of the one-contract worst stretch), through the same simulator
    as the held-out test. pnl and dip are the days at one contract; at_size(k) gives
    them at k contracts (default: k times the one-contract days).

    Score of a size: with the fees and the payout cap set, the expected dollars per
    attempt (topstep.net_per_attempt, as the held-out ranking uses); without them, the
    Combines passed per month of Combine fees (pass rate / mean months paid), which
    charges an attempt that runs out the clock three months, one that fails fast one,
    and leaves payouts out. A size at which no attempt passes scores the floor
    (combine_floor) however it got there: running out the clock, or failing.
    combine_score: the best size's score, the floor when nothing passes at any size,
    None when training is too short for a single attempt."""
    pnl, dip = np.asarray(pnl, float), np.asarray(dip, float)
    money = topstep.money_note(rules) is None
    floor = combine_floor(rules)
    max_size, too_risky = topstep.size_limit(fb.worst_stretch(pnl, dip), rules)
    days = [(pnl * k, dip * k) if at_size is None or k == 1 else at_size(k) for k in range(1, max_size + 1)]
    sizes: list[dict[str, Any]] = []
    best, top = 0.0, None
    sims = simulate_sizes(days, rules)
    for k, sim in enumerate(sims, start=1):
        n = sim["attempts"]
        net = topstep.net_per_attempt(sim, rules)
        per_month = sim["pass_rate"] / sim["mean_months"] if n else None
        score = None if not n else floor if not sim["passed"] else net if money else per_month
        sizes.append({"contracts": k, "pass_rate": _r(sim["pass_rate"], 4), "fail_rate": _r(sim["fail_rate"], 4),
                      "timeout_rate": _r(sim["gave_up"] / n, 4) if n else None,
                      "mean_days_to_pass": _r(sim["mean_days_to_pass"], 1), "mean_months": _r(sim["mean_months"], 3),
                      "mean_paid": _r(sim["mean_paid"], 2), "net": _r(net, 2), "passes_per_month": _r(per_month, 4),
                      "score": _r(score, 4)})
        if score is not None and (top is None or score > best):
            best, top = score, sizes[-1]
    out: dict[str, Any] = {
        "measure": "net_per_attempt" if money else "passes_per_month", "note": topstep.money_note(rules),
        "slippage": "normal", "sized_by": "scaled" if at_size is None else "rerun",
        "express_every": sims[0]["express_every"], "attempts": max(0, pnl.shape[0] - rules.combine_max_days + 1),
        "max_size": max_size, "too_risky": too_risky, "floor": round(floor, 2),
        "best_size": top["contracts"] if top else None,
        "sizes": sizes,
    }
    for key in ("pass_rate", "fail_rate", "timeout_rate", "mean_days_to_pass", "mean_months", "passes_per_month",
                "score"):
        out[key] = top[key] if top else None
    out["net_per_attempt"] = top["net"] if top else None
    return out


def _r(x: Any, digits: int) -> float | None:
    return None if x is None else round(float(x), digits)


def training_combine(train: FuturesData, module: ModuleType, params: dict[str, Any], rules: topstep.Rules,
                     normal: fb.FuturesRun | None = None, targets: Any = None,
                     should_stop: Callable[[], bool] | None = None) -> dict[str, Any]:
    """combine_numbers() of the training period at NORMAL slippage, from training prices
    only. `normal` is the one-contract normal-slippage training run when there is one
    already (training_numbers has it), `targets` the model's targets.

    Normal, not double, slippage: the held-out test it should predict runs at normal
    slippage, the double-slippage gate and Sharpe score already make a candidate prove it
    survives twice the costs, and a pass needs the profit target reached before the loss
    limit, so double costs on top would mostly turn thin but real edges into timeouts.

    Sizes come from the one-contract days times the size when the targets are whole
    numbers (every model so far: exact, and almost free); otherwise each size is a
    backtest of its own."""
    params = params_with_defaults(module, params)
    targets = targets if targets is not None else fb.model_targets(train, module, params)
    costs, end_of_day = topstep.costs_of(rules), topstep.day_rules(rules)
    if normal is None:
        normal = fb.run(train, module, params, costs, 1, end_of_day, should_stop=should_stop, targets=targets)

    def rerun(k: int) -> tuple[np.ndarray, np.ndarray]:
        run = fb.run(train, module, params, costs, k, end_of_day, should_stop=should_stop, targets=targets)
        return run.pnl, run.dip

    return combine_numbers(normal.pnl, normal.dip, rules, None if whole_targets(targets) else rerun)


def held_out_numbers(held: FuturesData, module: ModuleType, params: dict[str, Any], rules: topstep.Rules,
                     periods: dict[str, Any], worst_stretch: float,
                     should_stop: Callable[[], bool] | None = None) -> dict[str, Any]:
    """The held-out period through the Topstep simulator, at every contract size the
    training worst stretch allows, each beside its coin-flip twins."""
    first = first_day_after(held, periods["train_end"])
    if first >= held.n_days:
        raise ValueError("no held-out prices yet: load more futures prices")
    max_size, too_risky = topstep.size_limit(worst_stretch, rules)
    out = topstep.evaluate(held, module, params, rules, first, None, max_size, should_stop)
    out.update(max_size=max_size, too_risky=too_risky)
    return out


def summary_line(train: dict[str, Any], held: dict[str, Any]) -> str:
    """"Held-out: passes 34% (coin flip 28%) · 412 trades" for the Previous jobs table."""
    first = (held.get("sizes") or [{}])[0]
    sim, twin = first.get("sim") or {}, first.get("twin") or {}

    def pct(v: Any) -> str:
        return "-" if v is None else f"{v * 100:.0f}%"

    return (f"Held-out: passes {pct(sim.get('pass_rate'))} (coin flip {pct(twin.get('pass_rate'))}) · "
            f"{held['numbers']['trades']:,} trades")


def run_futures_backtest(params: dict[str, Any], checkpoint: dict[str, Any] | None, emit: Any,
                         should_stop: Any) -> dict[str, Any]:
    """Run backtest for a futures model: its training numbers and its held-out numbers."""
    module = module_for(str(params["module"]), params.get("params"))
    model_params = params_with_defaults(module, params.get("params"))
    rules = topstep.Rules.from_dict(params["rules"])
    periods = params["periods"]
    cache = PriceCache(params["_context"])
    emit({}, 0.0, "Loading futures prices (training)")
    train = cache.get("train")
    if should_stop():
        raise JobStopped()
    emit({}, 0.1, "Backtesting the training period")
    trained = training_numbers(train, module, model_params, rules, keep_daily=True, should_stop=should_stop)
    emit({}, 0.3, "Loading futures prices (held-out)")
    held = cache.get("held_out")
    emit({}, 0.4, "Backtesting the held-out period through Topstep's rules")
    result = held_out_numbers(held, module, model_params, rules, periods, trained["worst_stretch"], should_stop)
    return {"market": "futures", "feed": held.feed, "periods": periods, "params": model_params, "train": trained,
            "held_out": result, "summary": summary_line(trained, result), "model_id": params.get("model_id")}


def run_final_check(params: dict[str, Any], checkpoint: dict[str, Any] | None, emit: Any, should_stop: Any) -> dict[str, Any]:
    """The lockbox, opened once for one model at the contract size chosen on held-out
    prices. The result goes straight to the coordinator, which keeps it forever."""
    module = module_for(str(params["module"]), params.get("params"))
    model_params = params_with_defaults(module, params.get("params"))
    rules = topstep.Rules.from_dict(params["rules"])
    periods = params["periods"]
    ctx = params["_context"]
    cache = PriceCache(ctx)
    emit({}, 0.0, "Opening the lockbox prices")
    data = cache.get("lockbox", job_id=str(params.get("_job_id") or ""))
    first = first_day_after(data, periods["held_out_end"])
    if first >= data.n_days:
        raise ValueError("no lockbox prices: load more futures prices")
    emit({}, 0.2, "Backtesting the lockbox through Topstep's rules")
    size = max(1, int(params.get("contracts") or 1))
    result = topstep.evaluate(data, module, model_params, rules, first, None, size, should_stop)
    result["contracts"] = size
    model_id = str(params["model_id"])
    body = {"job_id": params.get("_job_id"), "feed": data.feed, "result": result}
    tries = []

    def send() -> Any:
        """The lockbox opens once, so its result is sent until it is stored."""
        tries.append(1)
        try:
            return http.post_json(f"{ctx['host_url']}/api/v1/models/{model_id}/final-check", body,
                                  token=str(ctx["worker_token"]), timeout=30.0)
        except http.HttpError as exc:
            if len(tries) > 1 and exc.status == 409 and "already stored" in exc.detail:
                return None  # an earlier try got through; only its answer was lost
            raise

    http.with_retries(send)
    chosen = result["sizes"][size - 1]
    sim, twin = chosen["sim"], chosen["twin"]

    def pct(v: Any) -> str:
        return "-" if v is None else f"{v * 100:.0f}%"

    return {"summary": f"Final check: lockbox passes {pct(sim.get('pass_rate'))} (coin flip {pct(twin.get('pass_rate'))})"
                       f" at {size} contract{'s' if size != 1 else ''}", "model_id": model_id}
