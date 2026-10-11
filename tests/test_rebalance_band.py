"""The rebalance band: a held position the model still wants is resized only when it is
at least REBALANCE_BAND (20%) of its target and MIN_TRADE_DOLLARS away from it. New
positions and full exits are never held back. The backtester and paper trading size
their orders with the same helper (rebalance_trades), checked here to agree.
"""
from __future__ import annotations

import sys

import pytest

from coordinator import trading
from coordinator.limits import Limits as CoordLimits
from fleet2.sim.backtest import COSTS, MIN_TRADE_DOLLARS, REBALANCE_BAND, Limits, rebalance_trades, run_backtest
from tests.test_backtest import fake_model, make_data

STOCK_RATE = COSTS["stocks"].rate
# No cap binds: weight x 10,000 is the target.
OPEN = Limits(money=10_000, max_per_position=10_000, max_per_model=10_000)


def trades(held, weights, equity=10_000.0, limits=OPEN, reserved=0.0):
    return rebalance_trades(held, weights, equity, limits, reserved)


# ------------------------------------------------------------------ the rule itself


@pytest.mark.parametrize("held", [801.0, 850.0, 1000.0, 1150.0, 1199.0])
def test_price_drift_inside_the_band_is_not_traded(held):
    assert trades({"A": held}, {"A": 0.1}) == {}  # target $1,000, band $200


@pytest.mark.parametrize("held,trade", [(1250.0, -250.0), (1200.0, -200.0), (790.0, 210.0), (800.0, 200.0)])
def test_a_gap_of_the_band_or_more_is_traded_to_the_target(held, trade):
    assert trades({"A": held}, {"A": 0.1}) == {"A": pytest.approx(trade)}


def test_a_target_that_shrinks_a_lot_is_still_trimmed():
    assert trades({"A": 1000.0}, {"A": 0.03}) == {"A": pytest.approx(-700.0)}  # target $300


def test_small_targets_fall_back_to_the_25_dollar_minimum():
    # Target $100: 20% is $20, under the $25 minimum, so the minimum decides.
    assert trades({"A": 76.0}, {"A": 0.01}) == {}
    assert trades({"A": 124.0}, {"A": 0.01}) == {}
    assert trades({"A": 74.0}, {"A": 0.01}) == {"A": pytest.approx(26.0)}
    assert trades({"A": 126.0}, {"A": 0.01}) == {"A": pytest.approx(-26.0)}


@pytest.mark.parametrize("want", [0.003, 0.01, 0.1])
def test_new_positions_are_never_held_back_by_the_band(want):
    assert trades({}, {"A": want}) == {"A": pytest.approx(want * 10_000)}


def test_new_positions_under_25_dollars_are_still_skipped():
    assert trades({}, {"A": 0.002}) == {}


@pytest.mark.parametrize("held", [0.5, 10.0, 1000.0, 5000.0])
def test_full_exits_are_never_held_back(held):
    assert trades({"A": held}, {}) == {"A": -held}
    assert trades({"A": held}, {"A": 0.0}) == {"A": -held}


def test_a_position_over_its_cap_from_drift_is_left_alone_inside_the_band():
    limits = Limits(money=10_000, max_per_position=1_000, max_per_model=10_000)
    assert trades({"A": 1150.0}, {"A": 0.5}, limits=limits) == {}  # target capped at $1,000
    assert trades({"A": 1250.0}, {"A": 0.5}, limits=limits) == {"A": pytest.approx(-250.0)}


def test_trims_held_back_by_the_band_are_made_when_buys_would_pass_the_model_cap():
    limits = Limits(money=4_000, max_per_position=1_000, max_per_model=3_000)
    held = {"A": 1150.0, "B": 1150.0}  # each $150 over a $1,000 target: inside the band
    assert trades(held, {"A": 0.25, "B": 0.25}, 4_000.0, limits) == {}
    # Buying C's $1,000 would leave 3,300 invested against a 3,000 cap: trim A and B first.
    got = trades(held, {"A": 0.25, "B": 0.25, "C": 0.25}, 4_000.0, limits)
    assert got == {"A": pytest.approx(-150.0), "B": pytest.approx(-150.0), "C": pytest.approx(1000.0)}
    assert sum(held.values()) + sum(got.values()) == pytest.approx(3_000.0)


def test_the_cap_check_counts_stuck_positions():
    limits = Limits(money=4_000, max_per_position=1_000, max_per_model=3_000)
    # $1,000 stuck (no bar now): the budget for tradable positions is 2,000. A is $100
    # over its $1,000 target (inside the band), but buying C's $1,000 needs the room.
    got = trades({"A": 1100.0}, {"A": 0.25, "C": 0.25}, 4_000.0, limits, reserved=1_000.0)
    assert got == {"A": pytest.approx(-100.0), "C": pytest.approx(1000.0)}
    assert trades({"A": 1100.0}, {"A": 0.25, "C": 0.25}, 4_000.0, limits, reserved=700.0) == {"C": pytest.approx(1000.0)}


# ------------------------------------------------------------------ in the backtest


def _small():
    return Limits(money=1000, max_per_position=1000, max_per_model=1000)


def test_backtest_does_not_trade_drift_inside_the_band():
    # Half the money in AAPL at 100; the price rises 15%: the position is worth ~$575
    # against a target of $500-$537 (past the old $25 rule but inside the band).
    data = make_data("stocks", {"AAPL": [100.0, 115.0, 115.0, 115.0]})
    model = fake_model(("AAPL",), {0: {"AAPL": 0.5}}, every=1)
    run = run_backtest(data, model, None, 0, 4, _small())
    (trade,) = run.trades
    assert trade.cost == pytest.approx(500.0)  # one buy ...
    qty = 500.0 / (100.0 * (1 + STOCK_RATE))
    assert trade.proceeds == pytest.approx(qty * 115.0 * (1 - STOCK_RATE))  # ... and only the final sale


def test_backtest_trims_drift_beyond_the_band():
    data = make_data("stocks", {"AAPL": [100.0, 160.0, 160.0, 160.0]}, {"AAPL": [160.0] * 4})
    model = fake_model(("AAPL",), {0: {"AAPL": 0.5}}, every=1)
    run = run_backtest(data, model, None, 0, 4, _small())
    qty = 500.0 / (100.0 * (1 + STOCK_RATE))
    equity = 500.0 + qty * 160.0  # marked at bar 0's close (160)
    trim = qty * 160.0 - 0.5 * equity
    assert trim >= REBALANCE_BAND * 0.5 * equity
    # proceeds: the trim at bar 1, then the rest sold at the end (target unchanged after).
    (trade,) = run.trades
    assert trade.cost == pytest.approx(500.0)
    assert trade.proceeds == pytest.approx(qty * 160.0 * (1 - STOCK_RATE))
    assert run.costs_paid == pytest.approx(500.0 - qty * 100.0 + qty * 160.0 * STOCK_RATE)


def test_backtest_exits_a_tiny_position_and_enters_a_new_one_at_once():
    data = make_data("stocks", {"AAPL": [100.0, 2.0, 2.0, 2.0], "MSFT": [50.0] * 4})
    model = fake_model(("AAPL", "MSFT"), {0: {"AAPL": 0.1}, 1: {"MSFT": 0.03}}, every=1)
    run = run_backtest(data, model, None, 0, 4, Limits(money=1000, max_per_position=1000, max_per_model=1000))
    by = {t.symbol: t for t in run.trades}
    assert by["AAPL"].exit_t == int(data.times[1])     # ~$2 left, sold anyway
    assert by["MSFT"].entry_t == int(data.times[1])    # a ~$30 new position, bought at once


# ------------------------------------------------------------------ backtest and paper trading agree


SCENARIOS = [
    # (bar-0 weights at 100 each, bar-1 prices, bar-1 weights)
    ({"AAPL": 0.1}, {"AAPL": 110.0}, {"AAPL": 0.1}),                      # drift inside the band
    ({"AAPL": 0.1}, {"AAPL": 130.0}, {"AAPL": 0.1}),                      # drift past it: trim
    ({"AAPL": 0.1}, {"AAPL": 75.0}, {"AAPL": 0.1}),                       # fall past it: top up
    ({"AAPL": 0.1}, {"AAPL": 100.0}, {"AAPL": 0.03}),                     # target shrinks a lot
    ({"AAPL": 0.1}, {"AAPL": 100.0}, {"AAPL": 0.11}),                     # small raise: nothing
    ({"AAPL": 0.1, "MSFT": 0.1}, {"AAPL": 105.0, "MSFT": 90.0}, {"AAPL": 0.1, "KO": 0.05}),  # exit + entry
    ({"AAPL": 0.002}, {"AAPL": 100.0}, {}),                              # never bought ($20): nothing
    # Cap: two positions ~$159 over a $1,000 target (inside the band) and a third
    # wanted; with a $3,000 model cap the band's held-back trims are made first.
    ({"AAPL": 0.1, "MSFT": 0.1}, {"AAPL": 116.0, "MSFT": 116.0}, {"AAPL": 0.1, "MSFT": 0.1, "KO": 0.1}),
]


def _limits_for(scenario_index):
    if scenario_index == len(SCENARIOS) - 1:
        return Limits(money=10_000, max_per_position=1_000, max_per_model=3_000)
    return Limits()


@pytest.mark.parametrize("k", range(len(SCENARIOS)))
def test_backtest_and_plan_orders_trade_the_same_amounts(k):
    w0, p1, w1 = SCENARIOS[k]
    limits = _limits_for(k)
    symbols = ("AAPL", "MSFT", "KO")
    opens = {s: [100.0, p1.get(s, 100.0), p1.get(s, 100.0)] for s in symbols}
    # Bar 0 closes at bar 1's price, so the backtest values the book at bar 1 exactly as
    # paper trading values it at the latest close.
    data = make_data("stocks", opens, {s: [p1.get(s, 100.0)] * 3 for s in symbols})
    seen = []

    def decide(history, params):
        book = sys._getframe(1).f_locals
        seen.append((book["cash"], dict(book["shares"])))
        return dict(w0 if len(history) == 0 else w1)

    run_backtest(data, fake_model(symbols, fn=decide, every=1), None, 0, 3, limits)
    (cash0, shares0), (_, shares1) = seen[1], seen[2]  # the book before and after bar 1's orders

    book = {"cash": cash0}
    coord = CoordLimits(starting_balance_per_model=limits.money, max_per_position=limits.max_per_position,
                        max_per_model=limits.max_per_model)
    prices = {s: p1.get(s, 100.0) for s in symbols}
    plan = trading.plan_orders(book, shares0, prices, w1, coord)
    expected = dict(shares0)
    for order in plan:
        s = order["symbol"]
        if order["side"] == "buy":
            expected[s] = expected.get(s, 0.0) + order["notional"] / (prices[s] * (1 + STOCK_RATE))
        else:
            expected[s] = expected.get(s, 0.0) - order["qty"]
    expected = {s: q for s, q in expected.items() if q * prices[s] >= 0.01}
    assert set(expected) == set(shares1)
    for s, q in shares1.items():
        # in dollars, within the cent plan_orders rounds a buy's notional to
        assert expected[s] * prices[s] == pytest.approx(q * prices[s], abs=0.01), (s, plan)
    # Sells come first in both.
    sides = [o["side"] for o in plan]
    assert sides == sorted(sides, key=lambda side: side != "sell")


def test_plan_orders_skips_drift_inside_the_band_and_keeps_its_order_shape():
    held = {"AAPL": 10.0, "MSFT": 10.0, "KO": 5.0}
    prices = {"AAPL": 115.0, "MSFT": 130.0, "KO": 50.0, "XOM": 40.0}
    book = {"cash": 7_250.0}  # equity 1,150 + 1,300 + 250 + 7,250 = 9,950
    plan = trading.plan_orders(book, held, prices, {"AAPL": 0.1, "MSFT": 0.1, "XOM": 0.05}, CoordLimits())
    by = {o["symbol"]: o for o in plan}
    assert set(by) == {"MSFT", "KO", "XOM"}            # AAPL: $1,150 vs $995, inside the band
    assert by["KO"] == {"symbol": "KO", "side": "sell", "qty": 5.0, "delta": -250.0}
    assert by["MSFT"]["side"] == "sell" and by["MSFT"]["qty"] == pytest.approx(10.0 * 305.0 / 1300.0)
    assert by["XOM"]["side"] == "buy" and by["XOM"]["notional"] == pytest.approx(497.5)
    assert [o["symbol"] for o in plan] == ["MSFT", "KO", "XOM"]  # biggest sale first, buys last
