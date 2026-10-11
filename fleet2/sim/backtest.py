"""The backtester: replay one model over past bars, the way it would trade for real.

Honesty rules (owner's spec), each enforced here:
- No lookahead. The model decides at bar i through History(data, i), which shows only
  bars 0..i-1 (closed before bar i opens). Orders fill at bar i's open, never at a
  price the model has already seen. tests/test_lookahead.py proves a decision does not
  change when every later bar is altered.
- Realistic costs. Every fill pays slippage (it buys a little above and sells a little
  below the open); crypto fills also pay Alpaca's crypto taker fee.
- Same limits as paper trading: the model's starting money, the most per position and
  the most in total (config/limits.toml), no short selling, no borrowing.
- Same orders as paper trading: rebalance_trades() decides what to trade for both, so
  a held position is only resized when it is at least REBALANCE_BAND (20%) of its
  target and MIN_TRADE_DOLLARS away from it. Price drift alone does not pay for a
  stream of small top-ups and trims; buying into a new position and selling one out
  entirely are never held back by the band.
- Positions still open at the end are valued as if sold at the last close, costs
  included, so ROI never counts money the model could not have taken out.

run_backtest() returns a Run (equity per bar, trades, benchmark); fleet2.sim.metrics
turns it into the eight numbers on the dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable

import numpy as np

from fleet2.models.base import clean_targets, params_with_defaults
from fleet2.sim.control import JobStopped
from fleet2.sim.marketdata import History, MarketData


@dataclass(frozen=True)
class Costs:
    """Cost of every fill, in basis points (1 bp = 0.01%)."""

    slippage_bps: float
    fee_bps: float

    @property
    def rate(self) -> float:
        return (self.slippage_bps + self.fee_bps) / 10_000.0


# Stocks: Alpaca charges no commission, so only slippage (5 bp on liquid large caps).
# Crypto: 10 bp slippage plus Alpaca's 25 bp taker fee (lowest volume tier, looked up
# in Alpaca's crypto fee schedule).
COSTS = {"stocks": Costs(slippage_bps=5.0, fee_bps=0.0), "crypto": Costs(slippage_bps=10.0, fee_bps=25.0)}
# No order under this many dollars (except selling a position out entirely).
MIN_TRADE_DOLLARS = 25.0
# A held position the model still wants is resized only when it is at least this
# fraction of its target away from it (and at least MIN_TRADE_DOLLARS).
REBALANCE_BAND = 0.20


@dataclass(frozen=True)
class Limits:
    """The money a model gets and the caps that apply to it (config/limits.toml)."""

    money: float = 10_000.0
    max_per_position: float = 1_000.0
    max_per_model: float = 10_000.0


@dataclass
class Trade:
    """One round trip in one symbol: from the first buy to the sale that empties it."""

    symbol: str
    entry_t: int
    exit_t: int = 0
    cost: float = 0.0       # dollars paid, costs included
    proceeds: float = 0.0   # dollars received, costs taken off

    @property
    def pnl(self) -> float:
        return self.proceeds - self.cost


@dataclass
class Run:
    """What a backtest produced, before it is summarised."""

    times: np.ndarray
    equity: np.ndarray
    benchmark: np.ndarray
    trades: list[Trade] = field(default_factory=list)
    money: float = 0.0
    bars_per_year: int = 252
    costs_paid: float = 0.0


Progress = Callable[[float], None]


def _budget(equity: float, limits: Limits, reserved: float) -> float:
    """The most the model's tradable positions may be worth together."""
    return min(limits.max_per_model, max(equity, 0.0)) - reserved


def _cap_targets(weights: dict[str, float], equity: float, limits: Limits, reserved: float = 0.0) -> dict[str, float]:
    """Dollar targets: weight x equity, each at most max_per_position, all together at
    most max_per_model (scaled down evenly when over). `reserved` is money already
    tied up in positions that cannot trade right now (no bar): it counts against the
    model's total, so a stuck position never lets the model invest past its cap."""
    dollars = {s: min(w * equity, limits.max_per_position) for s, w in weights.items()}
    total = sum(dollars.values())
    budget = _budget(equity, limits, reserved)
    if total > budget > 0:
        dollars = {s: d * budget / total for s, d in dollars.items()}
    elif budget <= 0:
        dollars = {}
    return dollars


def rebalance_trades(held: dict[str, float], weights: dict[str, float], equity: float, limits: Limits,
                     reserved: float = 0.0) -> dict[str, float]:
    """The trades that move positions worth `held` dollars towards `weights` x `equity`
    (capped by _cap_targets): {symbol: dollars}, > 0 buy, < 0 sell, absent = leave it.
    The backtester and paper trading (coordinator.trading.plan_orders) both size their
    orders here, so they always agree. Both pass only symbols that can trade now.

    - Not held, now wanted: buy the target, if it is at least MIN_TRADE_DOLLARS.
    - Held, no longer wanted: sell all of it, however small (exactly -held[symbol]).
    - Held and still wanted: resize to the target only when the gap is at least
      max(MIN_TRADE_DOLLARS, REBALANCE_BAND x target), so price drift alone does not
      trade. A position whose target shrinks a lot (held $1,000, target $300) is trimmed.
    - Caps: a held position never gets a buy past its target, and when this
      decision's buys would take the positions past the model's budget (most in total,
      or the money it has), the trims the band held back are made after all (as long
      as each is at least MIN_TRADE_DOLLARS), so buys never spend past the cap.
    """
    wanted = _cap_targets(weights, equity, limits, reserved)
    trades: dict[str, float] = {}
    held_back: dict[str, float] = {}
    for symbol in set(held) | set(wanted):
        have, want = held.get(symbol, 0.0), wanted.get(symbol, 0.0)
        if want <= 0.0:
            if have > 0.0:
                trades[symbol] = -have  # sell out entirely
        elif have <= 0.0:
            if want >= MIN_TRADE_DOLLARS:
                trades[symbol] = want  # a new position
        elif abs(want - have) >= max(MIN_TRADE_DOLLARS, REBALANCE_BAND * want):
            trades[symbol] = want - have
        elif have - want >= MIN_TRADE_DOLLARS:
            held_back[symbol] = want - have  # a trim inside the band
    if held_back and any(d > 0 for d in trades.values()):
        after = sum(held.values()) + sum(trades.values())
        if after > _budget(equity, limits, reserved):
            trades.update(held_back)
    return trades


def run_backtest(
    data: MarketData,
    model: ModuleType,
    params: dict[str, Any] | None,
    start: int,
    stop: int,
    limits: Limits = Limits(),
    benchmark: str | None = None,
    bars_per_year: int = 252,
    progress: Progress | None = None,
    should_stop: Callable[[], bool] | None = None,
    cost_multiplier: float = 1.0,
) -> Run:
    """Trade `model` from bar `start` to bar `stop - 1`. Bars before `start` are history
    the model may look at (its warm-up), never traded. `cost_multiplier` scales every
    fill's slippage and fees (2.0 = a stress test at double the usual costs)."""
    params = params_with_defaults(model, params)
    base = COSTS[data.market]
    costs = Costs(base.slippage_bps * cost_multiplier, base.fee_bps * cost_multiplier)
    every = max(1, int(model.rebalance_every(params)))
    # A model symbol with no data at all is simply never tradable (no bar to fill at).
    rows = {s: data.row(s) for s in model.SYMBOLS if s in data.symbols}
    value_px = data.last_close()
    stop = min(stop, data.n_bars)
    if not 0 <= start < stop:
        raise ValueError(f"empty backtest period {start}..{stop}")

    cash = limits.money
    shares: dict[str, float] = {}
    open_trades: dict[str, Trade] = {}
    trades: list[Trade] = []
    equity = np.empty(stop - start)
    paid = 0.0

    def fill(symbol: str, dollars: float, price: float, t: int) -> None:
        """Buy (dollars > 0) or sell (dollars < 0) at `price` plus or minus costs."""
        nonlocal cash, paid
        if dollars > 0:
            dollars = min(dollars, cash)
            if dollars < 1.0:
                return
            qty = dollars / (price * (1.0 + costs.rate))
            cash -= dollars
            paid += dollars - qty * price
            shares[symbol] = shares.get(symbol, 0.0) + qty
            trade = open_trades.get(symbol)
            if trade is None:
                trade = open_trades[symbol] = Trade(symbol, entry_t=t)
            trade.cost += dollars
        else:
            qty = min(shares.get(symbol, 0.0), -dollars / price)
            if qty <= 0:
                return
            got = qty * price * (1.0 - costs.rate)
            paid += qty * price - got
            cash += got
            left = shares.get(symbol, 0.0) - qty
            trade = open_trades[symbol]
            trade.proceeds += got
            if left * price < 0.01:
                shares.pop(symbol, None)
                trade.exit_t = t
                trades.append(open_trades.pop(symbol))
            else:
                shares[symbol] = left

    def marked(i: int) -> float:
        total = cash
        for symbol, qty in shares.items():
            px = value_px[rows[symbol], i]
            total += qty * (0.0 if np.isnan(px) else px)
        return total

    report_every = max(1, (stop - start) // 50)
    for i in range(start, stop):
        if (i - start) % every == 0:
            weights = clean_targets(model.target_positions(History(data, i), params), model.SYMBOLS)
            prev_equity = marked(i - 1) if i > 0 else cash
            tradable = {s for s in rows if not np.isnan(data.open[rows[s], i]) and data.open[rows[s], i] > 0}
            stuck = sum(qty * np.nan_to_num(value_px[rows[s], i - 1] if i > 0 else 0.0)
                        for s, qty in shares.items() if s not in tradable)
            # A held symbol with no bar now is left alone (stuck): try again at the next decision.
            held = {s: qty * data.open[rows[s], i] for s, qty in shares.items() if s in tradable}
            orders = rebalance_trades(held, {s: w for s, w in weights.items() if s in tradable},
                                      prev_equity, limits, stuck)
            t = int(data.times[i])
            for symbol, delta in sorted(orders.items(), key=lambda o: (o[1], o[0])):  # sells first, then buys
                if delta <= -held.get(symbol, 0.0):
                    delta *= 2  # sell everything (fill() never sells more than is held)
                fill(symbol, delta, data.open[rows[symbol], i], t)
        equity[i - start] = marked(i)
        if should_stop is not None and should_stop():
            raise JobStopped()
        if progress is not None and (i - start) % report_every == 0:
            progress((i - start + 1) / (stop - start))
    if progress is not None:
        progress(1.0)

    last = stop - 1
    end_t = int(data.times[last])
    for symbol in list(shares):
        px = value_px[rows[symbol], last]
        if not np.isnan(px):
            fill(symbol, -shares[symbol] * px * 2, px, end_t)
    equity[-1] = cash + sum(qty * value_px[rows[s], last] for s, qty in shares.items())

    return Run(
        times=data.times[start:stop].copy(),
        equity=equity,
        benchmark=buy_and_hold(data, benchmark, start, stop, limits.money, costs) if benchmark else np.full(stop - start, np.nan),
        trades=trades,
        money=limits.money,
        bars_per_year=bars_per_year,
        costs_paid=paid,
    )


def buy_and_hold(data: MarketData, symbol: str, start: int, stop: int, money: float, costs: Costs) -> np.ndarray:
    """Equity of putting all the money into `symbol` at the first open of the period and
    holding it (one buy, with the same costs; the final value is net of selling costs)."""
    row = data.row(symbol)
    opens = data.open[row, start:stop]
    first = int(np.argmax(~np.isnan(opens))) if (~np.isnan(opens)).any() else None
    out = np.full(stop - start, float(money))
    if first is None:
        return out
    qty = money / (opens[first] * (1.0 + costs.rate))
    out[first:] = qty * data.last_close()[row, start + first:stop]
    out[-1] *= 1.0 - costs.rate
    return out


def split_index(data: MarketData, held_out_fraction: float) -> int:
    """The first bar of the held-out period: the last `held_out_fraction` of the bars."""
    return int(round(data.n_bars * (1.0 - held_out_fraction)))


def split_at(data: MarketData, start_t: int | None, held_out_fraction: float) -> int:
    """The first bar of the held-out period: the first bar at or after the coordinator's
    fixed held-out start date, or (with no date) the last held_out_fraction of the bars."""
    if start_t is None:
        return split_index(data, held_out_fraction)
    return int(np.searchsorted(data.times, int(start_t), side="left"))
