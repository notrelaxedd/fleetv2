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
    by construction: the backtester is flat every night, and it is checked anyway)."""
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
    out["gates"] = gates
    out["passes"] = all(gates.values())
    return out


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
