"""The Topstep rules simulator: replays a futures model's days through the Combine, then
the Express Funded account, starting from every day of a period, and says how often it
would pass, how long that takes, and what it would pay after fees.

Every rule comes from config/topstep.toml (load_rules); nothing is hard-coded. A value
the file leaves unset is never guessed: whatever needs it comes back as None, and the
dashboard says "Set the fee in config/topstep.toml" (or names the value) instead.

The Combine, for one attempt starting on day s (balance = account size):
- Loss floor: starts max_loss_limit below the starting balance. At the end of each day
  it moves up to (highest end-of-day balance so far - max_loss_limit), never down, and
  never above the starting balance.
- The floor is enforced during the day: if the balance plus the day's worst dip (open
  trades included) reaches the floor, the attempt fails, even if the day closes green.
- Daily loss limit (when set): a day that falls that far ends at that loss.
- Passes at the end of a day when profit >= profit_target and the best day passes the
  consistency rule (best day <= best_day_share x the target, or x the total profit).
- An attempt still running when the period ends is "unfinished" and left out of every
  rate (counted separately), rather than guessed.

The Express Funded account (only after a pass, from the next day, for at most
express_days trading days): the same floor; a payout once there have been
payout_winning_days days of at least payout_min_day_profit since the start or the last
payout, of payout_share_of_balance x the profit, at most max_payout; after a payout the
floor moves to the starting balance. The trader keeps profit_split of each payout.

Money per attempt: profit_split x payouts - combine_monthly x months paid -
activation x (passed).
"""
from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np

FEE_MESSAGE = "Set the fee in config/topstep.toml"
PAYOUT_CAP_MESSAGE = "Set the payout cap (max_payout) in config/topstep.toml"


class RulesError(ValueError):
    """config/topstep.toml has a value of the wrong kind."""


@dataclass(frozen=True)
class Rules:
    account_size: float = 50_000.0
    profit_target: float = 3_000.0
    max_loss_limit: float = 2_000.0
    daily_loss_limit: float | None = None
    max_micro_contracts: int = 50
    flat_by: str = "15:10"
    consistency_kind: str = "target"
    best_day_share: float = 0.5
    payout_winning_days: int = 5
    payout_min_day_profit: float = 150.0
    payout_share_of_balance: float = 0.5
    max_payout: float | None = None
    profit_split: float = 0.9
    floor_to_start_after_payout: bool = True
    commission_per_side: dict[str, float | None] = field(default_factory=lambda: {"MES": None, "MNQ": None})
    combine_monthly: float | None = None
    activation: float | None = None
    slippage_ticks: float = 1.0
    express_days: int = 120
    days_per_month: int = 21
    twin_seeds: int = 10
    max_sizes: int = 10
    sizing_share: float = 0.5
    min_pass_rate_edge: float = 0.10
    max_lockbox_drop: float = 0.15
    min_shadow_days: int = 20
    max_download_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Rules":
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def sim_key(self) -> str:
        """A short fingerprint of every rule that changes the simulation (fees and the
        ready thresholds only change the money and the verdict, computed at display)."""
        skip = {"commission_per_side", "combine_monthly", "activation", "min_pass_rate_edge", "max_lockbox_drop",
                "min_shadow_days", "max_download_usd"}
        doc = {k: v for k, v in asdict(self).items() if k not in skip}
        return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:12]

    def commission(self, symbol: str) -> float | None:
        return self.commission_per_side.get(symbol)

    def missing_trading_fees(self) -> list[str]:
        return [f"commission_per_side_{s}" for s, v in sorted(self.commission_per_side.items()) if v is None]

    def account_fees_set(self) -> bool:
        return self.combine_monthly is not None and self.activation is not None

    def flat_minutes_before_close(self) -> int:
        """How many minutes before 15:00 Chicago time positions must be closed (0 when
        flat_by is 15:00 or later)."""
        hh, mm = (int(x) for x in self.flat_by.split(":"))
        return max(0, 15 * 60 - (hh * 60 + mm))


def _number(section: dict[str, Any], key: str, where: str, low: float, high: float, default: Any = ..., integer: bool = False):
    raw = section.get(key, default)
    if raw is ...:
        raise RulesError(f"{where} {key} is missing")
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise RulesError(f"{where} {key} must be a number, got {raw!r}")
    if not low <= float(raw) <= high:
        raise RulesError(f"{where} {key} must be between {low:g} and {high:g}, got {raw!r}")
    return int(raw) if integer else float(raw)


def load_rules(path: Path) -> Rules:
    """Read config/topstep.toml and check every value. Unset fees and the unset payout
    cap stay None; a value of the wrong kind stops the coordinator with its name."""
    if not path.is_file():
        raise RulesError(f"{path} is missing: it holds Topstep's rules for the futures simulator")
    with path.open("rb") as fh:
        doc = tomllib.load(fh)
    where = str(path) + ":"
    acc, con, exp = doc.get("account") or {}, doc.get("consistency") or {}, doc.get("express") or {}
    fees, costs, sim, ready = doc.get("fees") or {}, doc.get("costs") or {}, doc.get("simulator") or {}, doc.get("ready") or {}
    kind = con.get("kind", "target")
    if kind not in ("target", "total_profit"):
        raise RulesError(f"{where} [consistency] kind must be \"target\" or \"total_profit\", got {kind!r}")
    flat_by = str(acc.get("flat_by", "15:10"))
    try:
        hh, mm = (int(x) for x in flat_by.split(":"))
        assert 0 <= hh < 24 and 0 <= mm < 60
    except (ValueError, AssertionError):
        raise RulesError(f"{where} [account] flat_by must look like \"15:10\", got {flat_by!r}") from None
    daily = _number(acc, "daily_loss_limit", f"{where} [account]", 0, 1e7, default=0)
    floor_reset = exp.get("floor_to_start_after_payout", True)
    if not isinstance(floor_reset, bool):
        raise RulesError(f"{where} [express] floor_to_start_after_payout must be true or false")
    return Rules(
        account_size=_number(acc, "size", f"{where} [account]", 1_000, 1e7),
        profit_target=_number(acc, "profit_target", f"{where} [account]", 1, 1e7),
        max_loss_limit=_number(acc, "max_loss_limit", f"{where} [account]", 1, 1e7),
        daily_loss_limit=daily or None,
        max_micro_contracts=_number(acc, "max_micro_contracts", f"{where} [account]", 1, 1000, integer=True),
        flat_by=flat_by,
        consistency_kind=kind,
        best_day_share=_number(con, "best_day_share", f"{where} [consistency]", 0.01, 1.0),
        payout_winning_days=_number(exp, "payout_winning_days", f"{where} [express]", 1, 1000, integer=True),
        payout_min_day_profit=_number(exp, "payout_min_day_profit", f"{where} [express]", 0, 1e6),
        payout_share_of_balance=_number(exp, "payout_share_of_balance", f"{where} [express]", 0.01, 1.0),
        max_payout=_number(exp, "max_payout", f"{where} [express]", 1, 1e7, default=None),
        profit_split=_number(exp, "profit_split", f"{where} [express]", 0.0, 1.0),
        floor_to_start_after_payout=floor_reset,
        commission_per_side={s: _number(fees, f"commission_per_side_{s}", f"{where} [fees]", 0, 100, default=None)
                             for s in ("MES", "MNQ")},
        combine_monthly=_number(fees, "combine_monthly", f"{where} [fees]", 0, 1e5, default=None),
        activation=_number(fees, "activation", f"{where} [fees]", 0, 1e5, default=None),
        slippage_ticks=_number(costs, "slippage_ticks", f"{where} [costs]", 0, 100, default=1.0),
        express_days=_number(sim, "express_days", f"{where} [simulator]", 1, 2000, default=120, integer=True),
        days_per_month=_number(sim, "days_per_month", f"{where} [simulator]", 1, 31, default=21, integer=True),
        twin_seeds=_number(sim, "twin_seeds", f"{where} [simulator]", 1, 1000, default=10, integer=True),
        max_sizes=_number(sim, "max_sizes", f"{where} [simulator]", 1, 1000, default=10, integer=True),
        sizing_share=_number(sim, "sizing_share", f"{where} [simulator]", 0.01, 1.0, default=0.5),
        min_pass_rate_edge=_number(ready, "min_pass_rate_edge", f"{where} [ready]", 0, 1, default=0.10),
        max_lockbox_drop=_number(ready, "max_lockbox_drop", f"{where} [ready]", 0, 1, default=0.15),
        min_shadow_days=_number(ready, "min_shadow_days", f"{where} [ready]", 0, 1000, default=20, integer=True),
        max_download_usd=_number(doc.get("data") or {}, "max_download_usd", f"{where} [data]", 0, 1e6, default=0.0),
    )


# ------------------------------------------------------------------ the simulator


def _daily_limit(pnl: np.ndarray, dip: np.ndarray, rules: Rules) -> tuple[np.ndarray, np.ndarray]:
    """With a daily loss limit, a day whose dip reaches it ends at that loss."""
    if not rules.daily_loss_limit:
        return pnl, dip
    hit = dip <= -rules.daily_loss_limit
    return np.where(hit, -rules.daily_loss_limit, pnl), np.where(hit, -rules.daily_loss_limit, dip)


def trace(pnl: list[float], dip: list[float], rules: Rules) -> list[dict[str, Any]]:
    """One Combine attempt from the first day, day by day, written out plainly (the
    tests check the fast combine() below against it): the balance, the loss floor in
    force during the day, and the outcome."""
    start = rules.account_size
    bal, high, floor, best = start, start, start - rules.max_loss_limit, 0.0
    out = []
    for p, d in zip(*_daily_limit(np.asarray(pnl, float), np.asarray(dip, float), rules)):
        row = {"floor": floor, "low": bal + d}
        if bal + d <= floor:
            out.append({**row, "balance": bal + d, "outcome": "failed"})
            break
        bal += p
        best = max(best, p)
        high = max(high, bal)
        floor = min(max(floor, high - rules.max_loss_limit), start)
        allowed = rules.best_day_share * (rules.profit_target if rules.consistency_kind == "target" else bal - start)
        passed = bal - start >= rules.profit_target and best <= allowed + 1e-9
        out.append({**row, "balance": bal, "floor_after": floor, "outcome": "passed" if passed else "running"})
        if passed:
            break
    return out


def combine(pnl: np.ndarray, dip: np.ndarray, rules: Rules) -> dict[str, np.ndarray]:
    """One Combine attempt starting on every day. Returns per start day: state (1 passed,
    2 failed, 0 unfinished), days (trading days used) and end (the day it ended, -1)."""
    pnl, dip = _daily_limit(np.asarray(pnl, float), np.asarray(dip, float), rules)
    n = pnl.shape[0]
    start = float(rules.account_size)
    bal = np.full(n, start)
    high = np.full(n, start)
    floor = np.full(n, start - rules.max_loss_limit)
    best = np.zeros(n)
    state = np.zeros(n, dtype=np.int64)
    days = np.zeros(n, dtype=np.int64)
    end = np.full(n, -1, dtype=np.int64)
    first = np.arange(n)
    for k in range(n):
        a = first[(state == 0) & (first + k < n)]
        if a.size == 0:
            break
        day = a + k
        days[a] += 1
        broke = bal[a] + dip[day] <= floor[a]
        state[a[broke]], end[a[broke]] = 2, day[broke]
        a, day = a[~broke], day[~broke]
        bal[a] += pnl[day]
        best[a] = np.maximum(best[a], pnl[day])
        high[a] = np.maximum(high[a], bal[a])
        floor[a] = np.minimum(np.maximum(floor[a], high[a] - rules.max_loss_limit), start)
        profit = bal[a] - start
        allowed = rules.best_day_share * (rules.profit_target if rules.consistency_kind == "target" else profit)
        passed = (profit >= rules.profit_target) & (best[a] <= allowed + 1e-9)
        state[a[passed]], end[a[passed]] = 1, day[passed]
    return {"state": state, "days": days, "end": end}


def express(pnl: np.ndarray, dip: np.ndarray, begin: np.ndarray, rules: Rules) -> dict[str, np.ndarray]:
    """The Express Funded account opened on each day in `begin`. Returns per account:
    paid (dollars paid out, before the split), payouts (how many), days, breached
    (the loss floor was hit), cut (the period ended before express_days)."""
    pnl, dip = _daily_limit(np.asarray(pnl, float), np.asarray(dip, float), rules)
    if rules.max_payout is None:
        raise ValueError(PAYOUT_CAP_MESSAGE)
    n, m = pnl.shape[0], begin.shape[0]
    start = float(rules.account_size)
    bal, high = np.full(m, start), np.full(m, start)
    floor = np.full(m, start - rules.max_loss_limit)
    wins = np.zeros(m, dtype=np.int64)
    paid, count, days = np.zeros(m), np.zeros(m, dtype=np.int64), np.zeros(m, dtype=np.int64)
    running = np.ones(m, dtype=bool)
    breached = np.zeros(m, dtype=bool)
    for k in range(rules.express_days):
        a = np.flatnonzero(running & (begin + k < n))
        if a.size == 0:
            break
        day = begin[a] + k
        days[a] += 1
        broke = bal[a] + dip[day] <= floor[a]
        breached[a[broke]] = True
        running[a[broke]] = False
        a, day = a[~broke], day[~broke]
        bal[a] += pnl[day]
        high[a] = np.maximum(high[a], bal[a])
        floor[a] = np.minimum(np.maximum(floor[a], high[a] - rules.max_loss_limit), start)
        wins[a] += pnl[day] >= rules.payout_min_day_profit
        due = a[(wins[a] >= rules.payout_winning_days) & (bal[a] > start)]
        amount = np.minimum(rules.payout_share_of_balance * (bal[due] - start), rules.max_payout)
        bal[due] -= amount
        paid[due] += amount
        count[due] += 1
        wins[due] = 0
        high[due] = bal[due]
        if rules.floor_to_start_after_payout:
            floor[due] = start
    cut = running & (days < rules.express_days)
    return {"paid": paid, "payouts": count, "days": days, "breached": breached, "cut": cut}


def simulate(pnl: np.ndarray, dip: np.ndarray, rules: Rules) -> dict[str, Any]:
    """Every attempt that can start in the period: the Combine from each day, and the
    Express Funded account after each pass. Plain numbers (None where a rule value is
    unset); money is computed by net_per_attempt so fees can be filled in later."""
    pnl, dip = np.asarray(pnl, float), np.asarray(dip, float)
    c = combine(pnl, dip, rules)
    finished = c["state"] != 0
    n_fin = int(finished.sum())
    passed = c["state"] == 1
    out: dict[str, Any] = {
        "attempts": int(pnl.shape[0]),
        "finished": n_fin,
        "unfinished": int((~finished).sum()),
        "passed": int(passed.sum()),
        "failed": int((c["state"] == 2).sum()),
        "pass_rate": float(passed.sum() / n_fin) if n_fin else None,
        "fail_rate": float((c["state"] == 2).sum() / n_fin) if n_fin else None,
        "median_days_to_pass": float(np.median(c["days"][passed])) if passed.any() else None,
        "mean_months": float(np.mean(np.ceil(c["days"][finished] / rules.days_per_month))) if n_fin else None,
        "mean_paid": None, "mean_payouts": None, "express_breached": None, "express_cut": None,
        "payout_note": None,
    }
    if rules.max_payout is None:
        out["payout_note"] = PAYOUT_CAP_MESSAGE
        return out
    if n_fin:
        e = express(pnl, dip, c["end"][passed] + 1, rules)
        out["mean_paid"] = float(e["paid"].sum() / n_fin)
        out["mean_payouts"] = float(e["payouts"].sum() / n_fin)
        out["express_breached"] = float(e["breached"].mean()) if passed.any() else None
        out["express_cut"] = int(e["cut"].sum())
    return out


def net_per_attempt(sim: dict[str, Any], rules: Rules) -> float | None:
    """Expected dollars to you per Combine attempt: your share of the payouts, minus the
    monthly Combine fees, minus the activation fee when it passes. None while a fee or
    the payout cap is unset, or the period had no finished attempt."""
    if not rules.account_fees_set() or sim.get("mean_paid") is None or sim.get("mean_months") is None:
        return None
    return float(rules.profit_split * sim["mean_paid"] - rules.combine_monthly * sim["mean_months"]
                 - rules.activation * (sim["pass_rate"] or 0.0))


def money_note(rules: Rules) -> str | None:
    """Why the money per attempt cannot be shown yet (None when it can)."""
    if not rules.account_fees_set():
        return FEE_MESSAGE
    if rules.max_payout is None:
        return PAYOUT_CAP_MESSAGE
    return None


def average(sims: list[dict[str, Any]]) -> dict[str, Any]:
    """The mean of several simulations (the coin-flip twins), None where any is None."""
    out: dict[str, Any] = {}
    for key in sims[0]:
        values = [s[key] for s in sims]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            out[key] = float(np.mean(values))
        else:
            out[key] = values[0] if all(v == values[0] for v in values) else None
    return out


# ------------------------------------------------------------------ a model through the rules


def size_limit(worst_stretch_one: float, rules: Rules) -> tuple[int, bool]:
    """(largest contract size to try, too_risky). The size stops where the worst stretch
    at one contract, times the size, would use more than sizing_share of the loss limit.
    too_risky: even one contract goes past it (then only one is tried)."""
    cap = min(rules.max_micro_contracts, rules.max_sizes)
    if worst_stretch_one >= 0:
        return cap, False
    fits = math.floor(rules.sizing_share * rules.max_loss_limit / -worst_stretch_one)
    return max(1, min(cap, fits)), fits < 1


def evaluate(data: Any, module: ModuleType, params: dict[str, Any], rules: Rules, first_day: int,
             last_day: int | None, max_size: int, should_stop: Callable[[], bool] | None = None,
             slippage_ticks: float | None = None) -> dict[str, Any]:
    """Backtest the model at 1..max_size contracts over days [first_day, last_day) and
    put each through the simulator, with the coin-flip twins beside it. Also the
    period's numbers at one contract, at normal and at double slippage."""
    from fleet2.sim import futures_backtest as fb

    if rules.missing_trading_fees():
        raise ValueError(FEE_MESSAGE)
    slip = rules.slippage_ticks if slippage_ticks is None else slippage_ticks
    costs = fb.FuturesCosts(slip, {s: float(v) for s, v in rules.commission_per_side.items()})
    day_rules = fb.DayRules(cutoff_before_close=10 + rules.flat_minutes_before_close(),
                            flat_before_close=rules.flat_minutes_before_close())
    targets = fb.model_targets(data, module, params)
    sizes: list[dict[str, Any]] = []
    one: Any = None
    for size in range(1, max(1, min(max_size, rules.max_micro_contracts)) + 1):
        run = fb.run(data, module, params, costs, size, day_rules, first_day, last_day, None, should_stop, targets)
        if size == 1:
            one = run
        twins = [fb.run(data, module, params, costs, size, day_rules, first_day, last_day, seed, should_stop, targets)
                 for seed in range(1, rules.twin_seeds + 1)]
        sizes.append({"contracts": size, "sim": simulate(run.pnl, run.dip, rules),
                      "twin": average([simulate(t.pnl, t.dip, rules) for t in twins]),
                      "twin_pnl": float(np.mean([t.pnl.sum() for t in twins])),
                      "net_pnl": float(run.pnl.sum()), "worst_stretch": fb.worst_stretch(run.pnl, run.dip)})
    double = fb.run(data, module, params, costs.doubled(), 1, day_rules, first_day, last_day, None, should_stop, targets)
    twin_one = fb.run(data, module, params, costs, 1, day_rules, first_day, last_day, 1, should_stop, targets)
    summary = fb.summarize(one)
    return {
        "summary": summary,
        "double_slippage_pnl": round(float(double.pnl.sum()), 2),
        "twin_curve": fb.summarize(twin_one)["curve"],
        "sizes": sizes,
        "sim_key": rules.sim_key(),
        "feed": one.feed,
    }


def best_size(result: dict[str, Any], rules: Rules) -> dict[str, Any] | None:
    """The tested size with the most expected money per attempt (None while a fee or
    the payout cap is unset)."""
    best = None
    for entry in result.get("sizes") or []:
        net = net_per_attempt(entry["sim"], rules)
        if net is None:
            return None
        if best is None or net > best[0]:
            best = (net, entry)
    return None if best is None else {**best[1], "net": best[0], "twin_net": net_per_attempt(best[1]["twin"], rules)}
