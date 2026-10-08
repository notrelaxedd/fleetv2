"""The futures backtester: replay one day-trading model over 1-minute bars of MES or MNQ,
the way it would trade for real. It sits beside the stock and crypto backtester
(fleet2/sim/backtest.py), which it does not use or change.

Honesty rules, each enforced here:
- No lookahead. The model answers for every decision bar at once, but the answer for
  bar k only fills at the open of the next 1-minute bar after bar k closes, never at a
  price the model has seen. tests/test_futures_cutoff.py proves each model's answers
  up to bar k do not change when the prices after it are cut off.
- Whole contracts, long or short. A target of +1 is `contracts` contracts long, -0.5
  half of them (rounded), -1 all of them short.
- Costs on every fill: slippage in ticks on each side (a buy fills that many ticks
  above the price, a sell below) and the commission and exchange fees per contract
  per side, both from config/topstep.toml.
- Stops and targets (optional settings of the model) assume the worst. A stop and a
  target inside the same minute: the stop fills. A minute that opens past the stop
  fills at that open, not at the stop. A target only fills when the price trades
  through it, at the target price.
- Flat every day. No new trade (or added contract) from `cutoff_before_close`
  minutes before the close (14:50 Chicago time on a normal day), and everything is
  closed at the close of the session's last minute (15:00), well before Topstep's
  15:10. A position can never be carried overnight.
- A day on which the symbol had no prices is not traded.

run() returns a FuturesRun: one record per trading day (P&L in dollars, the worst dip
below the day's start including open trades, trades opened, minutes held, slippage and
fees paid) and the list of trades. summarize() turns it into the dashboard's numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable

import numpy as np

from fleet2.models.futures.base import clean_targets, params_with_defaults
from fleet2.sim.control import JobStopped
from fleet2.sim.futures_data import Bars, FuturesData, resample

TRADING_DAYS_PER_YEAR = 252
REASONS = ("model", "stop", "target", "close")


@dataclass(frozen=True)
class FuturesCosts:
    """Slippage in ticks per side, and commission plus exchange fees in dollars per
    contract per side, by symbol."""

    slippage_ticks: float
    fee_per_side: dict[str, float]

    def fee(self, symbol: str) -> float:
        return float(self.fee_per_side[symbol])

    def doubled(self) -> "FuturesCosts":
        """The same costs with twice the slippage (model search scores at double)."""
        return FuturesCosts(self.slippage_ticks * 2.0, dict(self.fee_per_side))


@dataclass(frozen=True)
class DayRules:
    """When the day ends for the backtester: no new trades from `cutoff_before_close`
    minutes before the session closes; everything closed at the close of the minute
    ending `flat_before_close` minutes before it (0: the session's last minute)."""

    cutoff_before_close: int = 10
    flat_before_close: int = 0


@dataclass
class FuturesRun:
    """What a futures backtest produced, one entry per trading day of the period."""

    days: np.ndarray
    pnl: np.ndarray
    dip: np.ndarray
    trades: np.ndarray
    minutes: np.ndarray
    slippage: np.ndarray
    fees: np.ndarray
    contracts: int
    feed: str
    trade_list: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_trades(self) -> int:
        return int(self.trades.sum())


def model_targets(data: FuturesData, module: ModuleType, params: dict[str, Any]) -> tuple[Bars, dict[str, np.ndarray]]:
    """The model's decision bars and its checked targets for every one of them."""
    params = params_with_defaults(module, params)
    bars = resample(data, int(params["bar_minutes"]))
    return bars, clean_targets(module.targets(bars, params), tuple(module.SYMBOLS), bars.n)


class _Book:
    """Bookkeeping of one symbol through one backtest (realised P&L per minute, the
    worst open-trade mark per minute, per-day counters and the trade list)."""

    def __init__(self, data: FuturesData, k: int, symbol: str, costs: FuturesCosts, real: np.ndarray,
                 unreal: np.ndarray, per_day: dict[str, np.ndarray], trades: list[tuple]) -> None:
        self.d, self.k, self.symbol = data, k, symbol
        self.pv = data.point_value(symbol)
        self.tick = data.tick(symbol)
        self.slip = costs.slippage_ticks * self.tick
        self.fee = costs.fee(symbol)
        self.real, self.unreal, self.per_day, self.trades = real, unreal, per_day, trades
        self.pos = 0
        self.avg = 0.0
        self.since = 0       # first minute of the current open stretch (for the worst mark)
        self.opened = 0      # minute the current trade was opened
        self.trade_pnl = 0.0
        self.trade_max = 0
        self.stop_px: float | None = None
        self.target_px: float | None = None

    # -------------------------------------------------------------- marks and fills

    def _mark(self, a: int, b: int) -> None:
        """The worst open-trade value in minutes [a, b): the low for a long, the high for a short."""
        if self.pos == 0 or b <= a:
            return
        adverse = self.d.low[self.k, a:b] if self.pos > 0 else self.d.high[self.k, a:b]
        self.unreal[a:b] += self.pos * self.pv * (adverse - self.avg)

    def _cost(self, minute: int, contracts: int) -> float:
        day = self.d.day[minute]
        fee = self.fee * contracts
        self.per_day["fees"][day] += fee
        self.per_day["slippage"][day] += self.slip * self.pv * contracts
        self.real[minute] -= fee
        return fee

    def _close(self, minute: int, price: float, reason: str, held_until: int) -> None:
        """Close the whole position at `price` (slippage already in it)."""
        gain = self.pos * self.pv * (price - self.avg)
        self.real[minute] += gain
        self.trade_pnl += gain - self._cost(minute, abs(self.pos))
        day = self.d.day[self.opened]
        self.per_day["minutes"][day] += max(1, held_until - self.opened)
        self.trades.append((self.symbol, int(self.d.times[self.opened]), int(self.d.times[minute]),
                            1 if self.pos > 0 else -1, self.trade_max, self.trade_pnl, reason))
        self.pos, self.avg, self.stop_px, self.target_px = 0, 0.0, None, None

    def _open(self, minute: int, contracts: int, price: float, stop_ticks: float, target_ticks: float) -> None:
        self.pos, self.avg = contracts, price
        self.since = self.opened = minute
        self.trade_max = abs(contracts)
        self.trade_pnl = -self._cost(minute, abs(contracts))
        self.per_day["trades"][self.d.day[minute]] += 1
        side = 1 if contracts > 0 else -1
        self.stop_px = price - side * stop_ticks * self.tick if stop_ticks > 0 else None
        self.target_px = price + side * target_ticks * self.tick if target_ticks > 0 else None

    def move_to(self, minute: int, want: int, stop_ticks: float, target_ticks: float) -> None:
        """Trade at the open of `minute` to hold `want` contracts."""
        if want == self.pos:
            return
        self._mark(self.since, minute)
        self.since = minute
        side = 1 if want > self.pos else -1
        price = float(self.d.open[self.k, minute]) + side * self.slip
        if self.pos != 0 and (want == 0 or (want > 0) != (self.pos > 0)):
            self._close(minute, price, "model", minute)
        if self.pos == 0:
            if want != 0:
                self._open(minute, want, price, stop_ticks, target_ticks)
            return
        if abs(want) > abs(self.pos):  # add to the position at a new average price
            extra = abs(want) - abs(self.pos)
            self.avg = (self.avg * abs(self.pos) + price * extra) / abs(want)
            self.trade_pnl -= self._cost(minute, extra)
            self.trade_max = max(self.trade_max, abs(want))
        else:  # take part of it off
            off = self.pos - want
            gain = off * self.pv * (price - self.avg)
            self.real[minute] += gain
            self.trade_pnl += gain - self._cost(minute, abs(off))
        self.pos = want

    def watch(self, a: int, b: int) -> bool:
        """Check the stop and target over minutes [a, b); close the trade at the first
        one hit (the stop when both are in the same minute). True when it closed."""
        if self.pos == 0 or b <= a or (self.stop_px is None and self.target_px is None):
            return False
        long = self.pos > 0
        lo, hi, op = self.d.low[self.k, a:b], self.d.high[self.k, a:b], self.d.open[self.k, a:b]
        never = b - a
        js = jt = never
        if self.stop_px is not None:
            hit = np.flatnonzero(lo <= self.stop_px) if long else np.flatnonzero(hi >= self.stop_px)
            js = int(hit[0]) if hit.size else never
        if self.target_px is not None:
            hit = np.flatnonzero(hi > self.target_px) if long else np.flatnonzero(lo < self.target_px)
            jt = int(hit[0]) if hit.size else never
        j = min(js, jt)
        if j == never:
            return False
        minute = a + j
        self._mark(self.since, minute)
        if js <= jt:
            gap = min(self.stop_px, float(op[j])) if long else max(self.stop_px, float(op[j]))
            price = gap - self.slip if long else gap + self.slip
            reason = "stop"
        else:
            price, reason = float(self.target_px), "target"
            self.per_day["slippage"][self.d.day[minute]] -= self.slip * self.pv * abs(self.pos)  # a limit order: no slippage
        self._close(minute, price, reason, minute + 1)
        return True

    def flatten(self, minute: int) -> None:
        """Close at the close of `minute` (the end of the day)."""
        if self.pos == 0:
            return
        self._mark(self.since, minute)
        long = self.pos > 0
        price = float(self.d.close[self.k, minute]) + (-self.slip if long else self.slip)
        adverse = float(self.d.low[self.k, minute] if long else self.d.high[self.k, minute])
        worst = self.pos * self.pv * (adverse - price)  # how much worse the minute got before its close
        self.unreal[minute] += min(0.0, worst)
        self._close(minute, price, "close", minute + 1)


def run(data: FuturesData, module: ModuleType, params: dict[str, Any] | None, costs: FuturesCosts,
        contracts: int = 1, rules: DayRules = DayRules(), first_day: int = 0, last_day: int | None = None,
        flip_seed: int | None = None, should_stop: Callable[[], bool] | None = None,
        targets: tuple[Bars, dict[str, np.ndarray]] | None = None) -> FuturesRun:
    """Trade days [first_day, last_day) of `data` with `contracts` contracts at full size.

    Days before first_day are history the model may look at, never traded; `data` should
    end where the period ends, so a later period is never even computed. `flip_seed`
    makes the coin-flip twin: the same entry times, each trade's direction decided by a
    seeded coin. `targets` reuses an earlier model_targets() answer."""
    params = params_with_defaults(module, params)
    last_day = data.n_days if last_day is None else min(int(last_day), data.n_days)
    if not 0 <= first_day < last_day:
        raise ValueError(f"empty futures period: days {first_day}..{last_day}")
    contracts = int(contracts)
    if contracts < 1:
        raise ValueError("contracts must be at least 1")
    bars, wanted = targets if targets is not None else model_targets(data, module, params)
    stop_ticks, target_ticks = float(params.get("stop_ticks") or 0), float(params.get("target_ticks") or 0)
    coin = np.random.default_rng(flip_seed) if flip_seed is not None else None

    n = data.n_minutes
    real, unreal = np.zeros(n), np.zeros(n)
    per_day = {k: np.zeros(data.n_days) for k in ("trades", "minutes", "slippage", "fees")}
    trade_rows: list[tuple] = []
    for symbol, target in wanted.items():
        if symbol not in data.symbols:
            continue
        k = data.row(symbol)
        book = _Book(data, k, symbol, costs, real, unreal, per_day, trade_rows)
        want = np.rint(target * contracts).astype(np.int64)
        want[~bars.tradable[k]] = 0
        before = np.r_[0, want[:-1]]
        before[bars.first] = 0
        change = np.flatnonzero(want != before)
        fill_at = bars.end[change]
        day_of = bars.day[change]
        keep = (fill_at < data.day_end[day_of]) & (day_of >= first_day) & (day_of < last_day)
        change, fill_at, day_of = change[keep], fill_at[keep], day_of[keep]
        bounds = np.flatnonzero(np.r_[True, day_of[1:] != day_of[:-1]]) if day_of.size else np.zeros(0, dtype=int)
        for g, lo in enumerate(bounds):
            hi = bounds[g + 1] if g + 1 < len(bounds) else change.shape[0]
            d = int(day_of[lo])
            last_minute = int(data.day_end[d]) - 1 - rules.flat_before_close
            cutoff = int(data.session[d]) - rules.cutoff_before_close
            model_sign = 0
            mult = 1
            for i in range(lo, hi):
                minute = int(fill_at[i])
                if minute > last_minute:
                    break
                book.watch(book.since, minute)
                q_model = int(want[change[i]])
                sign = (q_model > 0) - (q_model < 0)
                if coin is not None and sign != 0 and (sign != model_sign or book.pos == 0):
                    mult = 1 if coin.random() < 0.5 else -1
                model_sign = sign
                q = q_model * mult if coin is not None else q_model
                if data.minute[minute] >= cutoff:  # near the close: only reduce or close
                    same_side = q != 0 and book.pos != 0 and (q > 0) == (book.pos > 0)
                    q = q if same_side and abs(q) < abs(book.pos) else 0 if not same_side else book.pos
                book.move_to(minute, q, stop_ticks, target_ticks)
            if book.pos != 0:
                if not book.watch(book.since, last_minute + 1):
                    book.flatten(last_minute)
            if should_stop is not None and g % 50 == 0 and should_stop():
                raise JobStopped()

    starts = data.day_start
    cum = np.cumsum(real)
    before_day = np.where(starts > 0, cum[np.maximum(starts - 1, 0)], 0.0)
    equity = cum - np.repeat(before_day, data.day_end - starts) + unreal
    day_pnl = np.add.reduceat(real, starts)
    day_dip = np.minimum(np.minimum.reduceat(equity, starts), 0.0)
    sl = slice(first_day, last_day)
    trade_list = _trade_arrays(trade_rows)
    return FuturesRun(days=data.days[sl].copy(), pnl=day_pnl[sl], dip=day_dip[sl], trades=per_day["trades"][sl],
                      minutes=per_day["minutes"][sl], slippage=per_day["slippage"][sl], fees=per_day["fees"][sl],
                      contracts=contracts, feed=data.feed, trade_list=trade_list)


def _trade_arrays(rows: list[tuple]) -> dict[str, np.ndarray]:
    rows = sorted(rows, key=lambda r: r[1])
    cols = list(zip(*rows)) if rows else [()] * 7
    return {
        "symbol": np.asarray(cols[0], dtype=object),
        "entry_t": np.asarray(cols[1], dtype=np.int64),
        "exit_t": np.asarray(cols[2], dtype=np.int64),
        "side": np.asarray(cols[3], dtype=np.int64),
        "contracts": np.asarray(cols[4], dtype=np.int64),
        "pnl": np.asarray(cols[5], dtype=float),
        "reason": np.asarray(cols[6], dtype=object),
    }


# ------------------------------------------------------------------ the numbers


CURVE_POINTS = 240


def daily_sharpe(pnl: np.ndarray) -> float | None:
    """Mean over spread of daily P&L, scaled to a year (252 trading days). None when the
    days do not move (no trades) or there are too few of them."""
    if pnl.shape[0] < 3:
        return None
    sd = float(np.std(pnl, ddof=1))
    if sd == 0.0 or not np.isfinite(sd):
        return None
    return float(np.mean(pnl) / sd * np.sqrt(TRADING_DAYS_PER_YEAR))


def worst_stretch(pnl: np.ndarray, dip: np.ndarray) -> float:
    """The deepest fall in dollars from a high point of the running total to a later low,
    counting each day's worst moment, not just its close (0 or negative)."""
    total = np.cumsum(pnl)
    before = np.r_[0.0, total[:-1]]
    peak = np.maximum.accumulate(np.r_[0.0, total])[:-1]
    lows = before + dip
    return float(min(0.0, (lows - peak).min(initial=0.0)))


def summarize(run: FuturesRun) -> dict[str, Any]:
    """The futures numbers for the dashboard and for storage (plain floats, or None
    where they cannot be computed honestly)."""
    pnl = run.pnl
    tl = run.trade_list
    trade_pnl = tl.get("pnl", np.zeros(0))
    wins, losses = trade_pnl[trade_pnl > 0], trade_pnl[trade_pnl < 0]
    total = float(pnl.sum())
    traded = int(np.count_nonzero(run.trades))
    best = float(pnl.max(initial=0.0))
    holds = (tl.get("exit_t", np.zeros(0)) - tl.get("entry_t", np.zeros(0))) / 60.0
    n = pnl.shape[0]
    idx = np.unique(np.linspace(0, n - 1, min(n, CURVE_POINTS)).round().astype(int)) if n else np.zeros(0, dtype=int)
    running = np.cumsum(pnl)
    return {
        "contracts": run.contracts,
        "net_pnl": round(total, 2),
        "days": int(n),
        "days_traded": traded,
        "trades": int(run.n_trades),
        "trades_per_day": (run.n_trades / traded) if traded else None,
        "win_rate": float(wins.size / trade_pnl.size) if trade_pnl.size else None,
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.size else None,
        "avg_hold_minutes": float(holds.mean()) if holds.size else None,
        "daily_sharpe": daily_sharpe(pnl),
        "best_day": best,
        "worst_day": float(pnl.min(initial=0.0)),
        "worst_dip": float(run.dip.min(initial=0.0)),
        "best_day_share": (best / total) if total > 0 else None,
        "worst_stretch": worst_stretch(pnl, run.dip),
        "slippage_paid": round(float(run.slippage.sum()), 2),
        "fees_paid": round(float(run.fees.sum()), 2),
        "start": int(run.days[0]) if n else None,
        "end": int(run.days[-1]) if n else None,
        "feed": run.feed,
        "curve": {"d": [int(run.days[i]) for i in idx], "pnl": [round(float(running[i]), 2) for i in idx]},
    }
