"""The backtester and its metrics, checked on tiny hand-made data whose answers are
worked out by hand in each test (tests/test_lookahead.py already proves "no lookahead"
for the real models; here the rules of the owner's spec are checked number by number).

Conventions in the hand-made data: bar i opens at times[i] (base + i * step); stocks
cost 5 bp per fill (rate 0.0005), crypto 35 bp (10 slippage + 25 fee, rate 0.0035).
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from fleet2.models import REGISTRY
from fleet2.models.base import BadTargets, clean_targets
from fleet2.sim.backtest import (COSTS, MIN_TRADE_DOLLARS, Costs, Limits, Run, Trade, buy_and_hold,
                                 run_backtest, split_index)
from fleet2.sim.control import JobStopped
from fleet2.sim.marketdata import History, MarketData
from fleet2.sim.metrics import MIN_TRADES, beta_alpha, curve, max_drawdown, sharpe, summarize, t_stat
from fleet2.universe import MARKETS
from fleet2.worker.backtest_job import backtest_periods, signed_pct, summary_line

from tests.test_lookahead import synthetic

BASE = 1_600_000_000
STEP = {"stocks": 86_400, "crypto": 3_600}
NAN = float("nan")
MINUS = "−"


# ---------------------------------------------------------------------------------------
# helpers: tiny data and tiny fake models
# ---------------------------------------------------------------------------------------

def make_data(market: str, opens: dict[str, list[float]], closes: dict[str, list[float]] | None = None) -> MarketData:
    """MarketData from per-symbol open (and close) lists; NaN means "no bar"."""
    symbols = tuple(opens)
    o = np.array([opens[s] for s in symbols], dtype=float)
    c = np.array([(closes or opens)[s] for s in symbols], dtype=float)
    n = o.shape[1]
    times = BASE + np.arange(n, dtype=np.int64) * STEP[market]
    return MarketData(market, "1Day" if market == "stocks" else "1Hour", "test", symbols, times,
                      o, np.fmax(o, c), np.fmin(o, c), c, np.where(np.isnan(c), NAN, 1e6))


def fake_model(symbols, schedule=None, every=1, warmup=0, fn=None, defaults=None):
    """A model module. `schedule` maps a history length (= decision bar) to the weights
    wanted from then on; or pass `fn(history, params) -> weights` directly."""
    steps = sorted((schedule or {}).items())

    def scheduled(history, params):
        wanted: dict[str, float] = {}
        for at, weights in steps:
            if len(history) >= at:
                wanted = weights
        return dict(wanted)

    return SimpleNamespace(
        NAME="fake", MARKET="stocks", SYMBOLS=tuple(symbols), DEFAULT_PARAMS=dict(defaults or {}),
        rebalance_every=lambda params: every, warmup=lambda params: warmup,
        target_positions=fn or scheduled)


def always(symbol: str, weight: float = 1.0, **kw):
    return fake_model((symbol,), {0: {symbol: weight}}, **kw)


STOCK_RATE = COSTS["stocks"].rate
CRYPTO_RATE = COSTS["crypto"].rate


def test_cost_rates_match_the_owners_spec():
    assert COSTS["stocks"] == Costs(slippage_bps=5.0, fee_bps=0.0)
    assert COSTS["crypto"] == Costs(slippage_bps=10.0, fee_bps=25.0)
    assert STOCK_RATE == pytest.approx(0.0005)
    assert CRYPTO_RATE == pytest.approx(0.0035)
    assert MIN_TRADE_DOLLARS == 25.0
    assert Limits() == Limits(money=10_000.0, max_per_position=1_000.0, max_per_model=10_000.0)


# ---------------------------------------------------------------------------------------
# 1. fill price and costs
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("market,symbol,rate", [("stocks", "AAPL", STOCK_RATE), ("crypto", "BTC/USD", CRYPTO_RATE)])
def test_buy_fills_at_the_decision_bars_open_and_final_mark_pays_costs(market, symbol, rate):
    # bar:     0    1    2    3
    opens = {symbol: [100.0, 110.0, 120.0, 130.0]}
    closes = {symbol: [105.0, 115.0, 125.0, 135.0]}
    data = make_data(market, opens, closes)
    model = always(symbol, every=100)  # one decision only: at the first bar traded (bar 1)
    run = run_backtest(data, model, None, 1, 4)

    # Bought $1,000 (the position cap) at bar 1's OPEN 110 (not its close 115, not bar 0's 105).
    qty = 1000.0 / (110.0 * (1 + rate))
    assert qty * 110.0 * (1 + rate) == pytest.approx(1000.0)
    assert run.equity.shape == (3,)
    assert run.equity[0] == pytest.approx(9000.0 + qty * 115.0)   # marked at bar 1's close
    assert run.equity[1] == pytest.approx(9000.0 + qty * 125.0)
    # The last bar: valued as sold at the last close (135) with the selling cost.
    final = 9000.0 + qty * 135.0 * (1 - rate)
    assert run.equity[2] == pytest.approx(final)
    assert run.costs_paid == pytest.approx(qty * 110.0 * rate + qty * 135.0 * rate)
    assert run.money == 10_000.0
    np.testing.assert_array_equal(run.times, data.times[1:4])

    (trade,) = run.trades
    assert trade.symbol == symbol
    assert trade.cost == pytest.approx(1000.0)
    assert trade.proceeds == pytest.approx(qty * 135.0 * (1 - rate))
    assert trade.pnl == pytest.approx(trade.proceeds - 1000.0)
    assert trade.entry_t == int(data.times[1]) and trade.exit_t == int(data.times[3])


def test_a_decision_at_the_first_bar_fills_at_that_bars_open():
    data = make_data("stocks", {"AAPL": [100.0, 90.0, 80.0]}, {"AAPL": [50.0, 60.0, 70.0]})
    run = run_backtest(data, always("AAPL", every=100), None, 0, 3)
    qty = 1000.0 / (100.0 * (1 + STOCK_RATE))  # open 100, not close 50
    assert run.equity[0] == pytest.approx(9000.0 + qty * 50.0)
    assert run.trades[0].entry_t == int(data.times[0])


def test_a_sell_receives_price_times_one_minus_rate_and_a_buy_pays_one_plus_rate():
    # Buy at bar 1 (open 100), sell everything at bar 3 (open 200) on a crypto market.
    sym = "ETH/USD"
    data = make_data("crypto", {sym: [100.0, 100.0, 100.0, 200.0, 200.0]})
    model = fake_model((sym,), {1: {sym: 1.0}, 3: {}}, every=2)
    run = run_backtest(data, model, None, 1, 5)
    qty = 1000.0 / (100.0 * (1 + CRYPTO_RATE))
    got = qty * 200.0 * (1 - CRYPTO_RATE)
    (trade,) = run.trades
    assert trade.cost == pytest.approx(1000.0)
    assert trade.proceeds == pytest.approx(got)
    assert run.equity[-1] == pytest.approx(9000.0 + got)
    assert run.costs_paid == pytest.approx(qty * 100.0 * CRYPTO_RATE + qty * 200.0 * CRYPTO_RATE)


def test_decisions_see_only_bars_before_the_decision_bar():
    seen = []

    def probe(history, params):
        seen.append((len(history), history.now, history.close("AAPL")[-1], history.open("AAPL")[-1]))
        return {}

    data = make_data("stocks", {"AAPL": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]},
                     {"AAPL": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0]})
    run_backtest(data, fake_model(("AAPL",), fn=probe, every=2), None, 2, 6)
    assert [s[0] for s in seen] == [2, 4]
    assert [s[1] for s in seen] == [int(data.times[2]), int(data.times[4])]
    assert [s[2] for s in seen] == [21.0, 23.0]  # the previous bar's close
    assert [s[3] for s in seen] == [11.0, 13.0]  # the previous bar's open, never the current one


def test_model_params_default_and_unknown_names_are_dropped():
    got = []
    model = fake_model(("AAPL",), fn=lambda h, p: got.append(dict(p)) or {}, defaults={"a": 1, "b": 2})
    data = make_data("stocks", {"AAPL": [10.0, 10.0, 10.0]})
    run_backtest(data, model, {"b": 5, "nonsense": 9}, 0, 3)
    assert got[0] == {"a": 1, "b": 5}


# ---------------------------------------------------------------------------------------
# 2. position and model caps
# ---------------------------------------------------------------------------------------

def test_a_full_weight_buys_only_the_per_position_cap():
    data = make_data("stocks", {"AAPL": [100.0, 100.0, 100.0]})
    run = run_backtest(data, always("AAPL", every=100), None, 0, 3)
    assert run.trades[0].cost == pytest.approx(1000.0)
    qty = 1000.0 / (100.0 * (1 + STOCK_RATE))
    assert run.equity[0] == pytest.approx(9000.0 + qty * 100.0)


def test_a_position_cap_that_is_smaller_than_the_money_limits_each_symbol():
    data = make_data("stocks", {"AAPL": [100.0] * 3, "MSFT": [50.0] * 3})
    model = fake_model(("AAPL", "MSFT"), {0: {"AAPL": 0.5, "MSFT": 0.5}}, every=100)
    run = run_backtest(data, model, None, 0, 3, Limits(money=10_000, max_per_position=300, max_per_model=10_000))
    assert {t.symbol: round(t.cost, 6) for t in run.trades} == {"AAPL": 300.0, "MSFT": 300.0}


def test_money_below_the_cap_is_all_that_can_be_spent():
    data = make_data("stocks", {"AAPL": [100.0] * 3})
    run = run_backtest(data, always("AAPL", every=100), None, 0, 3, Limits(money=400, max_per_position=1000, max_per_model=1000))
    assert run.trades[0].cost == pytest.approx(400.0)
    assert run.equity[0] == pytest.approx(400.0 / (1 + STOCK_RATE))  # cash is 0, never negative
    assert run.money == 400


def test_targets_over_the_model_cap_scale_down_evenly():
    data = make_data("stocks", {"AAPL": [100.0] * 3, "MSFT": [50.0] * 3})
    model = fake_model(("AAPL", "MSFT"), {0: {"AAPL": 0.6, "MSFT": 0.4}}, every=100)
    # weight x equity = $6,000 and $4,000: both under the position cap, together over
    # the $500 model cap, so each is scaled by 500 / 10,000 and the 60/40 ratio is kept.
    run = run_backtest(data, model, None, 0, 3, Limits(money=10_000, max_per_position=10_000, max_per_model=500))
    cost = {t.symbol: t.cost for t in run.trades}
    assert cost["AAPL"] == pytest.approx(300.0)
    assert cost["MSFT"] == pytest.approx(200.0)


def test_both_caps_apply_together_and_scale_the_capped_dollars():
    data = make_data("stocks", {"AAPL": [100.0] * 3, "MSFT": [50.0] * 3})
    model = fake_model(("AAPL", "MSFT"), {0: {"AAPL": 0.6, "MSFT": 0.4}}, every=100)
    # position cap 1,000 turns $6,000/$4,000 into $1,000/$1,000; model cap 500 halves... to 250 each.
    run = run_backtest(data, model, None, 0, 3, Limits(money=10_000, max_per_position=1_000, max_per_model=500))
    cost = {t.symbol: t.cost for t in run.trades}
    assert cost["AAPL"] == pytest.approx(250.0)
    assert cost["MSFT"] == pytest.approx(250.0)


def test_total_under_the_model_cap_is_left_alone():
    data = make_data("stocks", {"AAPL": [100.0] * 3, "MSFT": [50.0] * 3})
    model = fake_model(("AAPL", "MSFT"), {0: {"AAPL": 0.05, "MSFT": 0.04}}, every=100)
    run = run_backtest(data, model, None, 0, 3, Limits(money=10_000, max_per_position=1_000, max_per_model=1_000))
    cost = {t.symbol: t.cost for t in run.trades}
    assert cost["AAPL"] == pytest.approx(500.0)  # $500 + $400 = $900, under the $1,000 model cap
    assert cost["MSFT"] == pytest.approx(400.0)


def test_a_position_that_grew_past_the_cap_is_trimmed_back_to_it():
    # Bought $1,000 at 100; the price doubles; the next decision wants $1,000 again.
    data = make_data("stocks", {"AAPL": [100.0, 200.0, 200.0]})
    run = run_backtest(data, always("AAPL", every=1), None, 0, 3)
    qty0 = 1000.0 / (100.0 * (1 + STOCK_RATE))
    # equity before the decision at bar 1 uses bar 0's close (100): the target is $1,000
    # while the position is worth qty0 * 200 = ~$1,999, so ~$999 is sold at 200.
    sold_qty = (qty0 * 200.0 - 1000.0) / 200.0
    left = qty0 - sold_qty
    cash = 9000.0 + sold_qty * 200.0 * (1 - STOCK_RATE)
    assert run.equity[1] == pytest.approx(cash + left * 200.0, rel=1e-9)  # marked at bar 1's close (200)
    assert left * 200.0 == pytest.approx(1000.0, abs=1e-6)


# ---------------------------------------------------------------------------------------
# bad targets
# ---------------------------------------------------------------------------------------

def test_weights_adding_up_to_more_than_one_are_refused():
    data = make_data("stocks", {"AAPL": [100.0] * 3, "MSFT": [50.0] * 3})
    model = fake_model(("AAPL", "MSFT"), {0: {"AAPL": 0.7, "MSFT": 0.4}})
    with pytest.raises(BadTargets):
        run_backtest(data, model, None, 0, 3)


def test_unknown_symbol_negative_or_nan_weights_and_non_dicts_are_refused():
    data = make_data("stocks", {"AAPL": [100.0] * 3})
    for bad in ({"TSLA": 0.5}, {"AAPL": -0.1}, {"AAPL": 1.5}, {"AAPL": NAN}, [("AAPL", 1.0)], None):
        model = fake_model(("AAPL",), fn=lambda h, p, bad=bad: bad)
        with pytest.raises(BadTargets):
            run_backtest(data, model, None, 0, 3)


def test_clean_targets_drops_zero_weights_and_allows_exactly_one():
    assert clean_targets({"A": 0.5, "B": 0.0, "C": 0.5}, ("A", "B", "C")) == {"A": 0.5, "C": 0.5}


def test_a_model_symbol_with_no_data_is_never_traded():
    data = make_data("stocks", {"AAPL": [100.0] * 3})
    model = fake_model(("AAPL", "MSFT"), {0: {"MSFT": 1.0}})
    run = run_backtest(data, model, None, 0, 3)
    assert run.trades == [] and run.equity[-1] == run.money


def test_an_empty_period_is_refused():
    data = make_data("stocks", {"AAPL": [100.0] * 3})
    for start, stop in ((2, 2), (3, 3), (-1, 2), (1, 0)):
        with pytest.raises(ValueError):
            run_backtest(data, always("AAPL"), None, start, stop)


def test_stop_beyond_the_data_is_clipped_to_the_last_bar():
    data = make_data("stocks", {"AAPL": [100.0] * 4})
    run = run_backtest(data, always("AAPL", every=100), None, 1, 99)
    assert run.equity.shape == (3,) and run.times[-1] == data.times[-1]


# ---------------------------------------------------------------------------------------
# 3. long only, no borrowing (property style)
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_random_strategy_never_goes_short_or_borrows(seed):
    symbols = ("AAPL", "MSFT", "KO", "XOM", "JPM")
    data = synthetic("stocks", 150, seed=seed)
    n_sym = len(symbols)
    rng = np.random.default_rng(seed + 100)
    limits = Limits(money=5_000, max_per_position=900, max_per_model=2_500)
    observed = []

    def random_weights(history, params):
        # The caller's frame is run_backtest: read its cash and share book to audit it.
        book = sys._getframe(1).f_locals
        cash, shares = book["cash"], dict(book["shares"])
        i = len(history)
        observed.append(1)
        assert cash >= -1e-9, f"cash {cash} below zero at bar {i}"
        assert all(q >= 0 for q in shares.values()), f"negative shares at bar {i}: {shares}"
        if i > 0:  # what is held, at the open prices the last orders were filled at
            held = {s: q * data.open[data.row(s), i - 1] for s, q in shares.items()}
            assert all(v <= limits.max_per_position + MIN_TRADE_DOLLARS for v in held.values()), held
            assert sum(held.values()) <= limits.max_per_model + MIN_TRADE_DOLLARS * n_sym, held
        w = rng.dirichlet(np.ones(n_sym + 1))[:n_sym]  # sums to < 1, some mass kept as cash
        w = np.where(rng.random(n_sym) < 0.3, 0.0, w)
        return {s: float(x) for s, x in zip(symbols, w)}

    model = fake_model(symbols, fn=random_weights, every=1)
    run = run_backtest(data, model, None, 5, 150, limits)

    assert len(observed) == 145
    assert np.isfinite(run.equity).all() and (run.equity > 0).all()
    assert len(run.trades) > 5
    for t in run.trades:
        assert t.cost > 0 and t.proceeds >= 0 and t.exit_t >= t.entry_t
    # Everything is sold at the end, so the final equity is cash alone: the money plus
    # every round trip's profit. A leak of cash (borrowing, free shares) would break it.
    assert run.equity[-1] == pytest.approx(limits.money + sum(t.pnl for t in run.trades), rel=1e-9)
    assert run.costs_paid > 0


def test_only_selling_what_is_held_never_sells_short():
    # The model drops to zero in a symbol it never bought: nothing happens.
    data = make_data("stocks", {"AAPL": [100.0] * 4})
    run = run_backtest(data, fake_model(("AAPL",), {0: {}}, every=1), None, 0, 4)
    assert run.trades == [] and run.costs_paid == 0.0
    np.testing.assert_array_equal(run.equity, np.full(4, 10_000.0))


# ---------------------------------------------------------------------------------------
# 4. round trips
# ---------------------------------------------------------------------------------------

def test_buy_then_sell_is_one_trade_with_cost_proceeds_pnl_and_epoch_times():
    #            0      1      2      3      4      5
    opens = [100.0, 100.0, 105.0, 110.0, 112.0, 120.0]
    closes = [100.0, 101.0, 106.0, 111.0, 113.0, 121.0]
    data = make_data("stocks", {"AAPL": opens}, {"AAPL": closes})
    model = fake_model(("AAPL",), {1: {"AAPL": 1.0}, 3: {}}, every=2)  # decisions at bars 1, 3, 5
    run = run_backtest(data, model, None, 1, 6)

    qty = 1000.0 / (100.0 * (1 + STOCK_RATE))
    proceeds = qty * 110.0 * (1 - STOCK_RATE)  # sold at bar 3's open
    (trade,) = run.trades
    assert trade.symbol == "AAPL"
    assert trade.cost == pytest.approx(1000.0)
    assert trade.proceeds == pytest.approx(proceeds)
    assert trade.pnl == pytest.approx(proceeds - 1000.0)
    assert trade.entry_t == BASE + 1 * 86_400 == int(data.times[1])
    assert trade.exit_t == BASE + 3 * 86_400 == int(data.times[3])
    assert isinstance(trade.entry_t, int) and isinstance(trade.exit_t, int)
    # After the sale everything is cash, so equity stays flat from bar 3's close on.
    assert run.equity[2] == pytest.approx(9000.0 + proceeds)
    assert run.equity[3] == pytest.approx(9000.0 + proceeds)
    assert run.equity[-1] == pytest.approx(9000.0 + proceeds)
    assert run.costs_paid == pytest.approx(qty * 100.0 * STOCK_RATE + qty * 110.0 * STOCK_RATE)


def test_a_position_open_at_the_end_is_sold_at_the_last_close_with_costs_and_counted():
    opens = [100.0, 100.0, 101.0, 102.0, 101.0, 102.0]
    closes = [100.0, 100.5, 101.5, 101.5, 101.0, 103.0]
    data = make_data("stocks", {"AAPL": opens}, {"AAPL": closes})
    run = run_backtest(data, always("AAPL", every=100), None, 1, 6)
    qty = 1000.0 / (100.0 * (1 + STOCK_RATE))
    (trade,) = run.trades
    assert trade.entry_t == int(data.times[1])
    assert trade.exit_t == int(data.times[5])
    assert trade.proceeds == pytest.approx(qty * 103.0 * (1 - STOCK_RATE))  # last close, not last open
    assert run.equity[-1] == pytest.approx(9000.0 + qty * 103.0 * (1 - STOCK_RATE))
    assert summarize(run)["trades"] == 1


def test_adding_to_and_trimming_a_position_stay_one_trade():
    # buy $400, add to $800, trim to $300, sell the rest: one round trip in one symbol.
    data = make_data("stocks", {"AAPL": [100.0] * 7})
    model = fake_model(("AAPL",), {0: {"AAPL": 0.4}, 1: {"AAPL": 0.8}, 2: {"AAPL": 0.3}, 3: {}}, every=1)
    run = run_backtest(data, model, None, 0, 7, Limits(money=1000, max_per_position=1000, max_per_model=1000))
    (trade,) = run.trades
    assert trade.entry_t == int(data.times[0]) and trade.exit_t == int(data.times[3])
    assert trade.cost > 800 and trade.proceeds > 0
    # Flat prices: everything lost is exactly the 5 bp paid on each fill.
    assert trade.pnl == pytest.approx(-run.costs_paid)
    assert run.equity[-1] == pytest.approx(1000.0 - run.costs_paid)


def test_buying_again_after_a_full_sale_starts_a_new_trade():
    data = make_data("stocks", {"AAPL": [100.0] * 8})
    model = fake_model(("AAPL",), {0: {"AAPL": 1.0}, 2: {}, 4: {"AAPL": 1.0}, 6: {}}, every=1)
    run = run_backtest(data, model, None, 0, 8)
    assert [(t.entry_t, t.exit_t) for t in run.trades] == [(int(data.times[0]), int(data.times[2])),
                                                          (int(data.times[4]), int(data.times[6]))]


def test_two_symbols_make_two_trades_each_closed_at_the_end():
    data = make_data("stocks", {"AAPL": [100.0, 100.0, 120.0], "MSFT": [50.0, 50.0, 40.0]})
    model = fake_model(("AAPL", "MSFT"), {0: {"AAPL": 0.5, "MSFT": 0.5}}, every=100)
    run = run_backtest(data, model, None, 0, 3, Limits(money=1000, max_per_position=1000, max_per_model=1000))
    by = {t.symbol: t for t in run.trades}
    assert set(by) == {"AAPL", "MSFT"}
    assert by["AAPL"].pnl > 0 > by["MSFT"].pnl
    assert by["AAPL"].exit_t == by["MSFT"].exit_t == int(data.times[2])
    assert run.equity[-1] == pytest.approx(1000.0 + by["AAPL"].pnl + by["MSFT"].pnl)


def test_sells_are_filled_before_buys_so_a_switch_is_funded_by_the_sale():
    data = make_data("stocks", {"AAPL": [100.0] * 4, "MSFT": [100.0] * 4})
    model = fake_model(("AAPL", "MSFT"), {0: {"AAPL": 1.0}, 2: {"MSFT": 1.0}}, every=1)
    limits = Limits(money=1000, max_per_position=1000, max_per_model=1000)
    run = run_backtest(data, model, None, 0, 4, limits)
    by = {t.symbol: t for t in run.trades}
    assert by["AAPL"].exit_t == int(data.times[2]) and by["MSFT"].entry_t == int(data.times[2])
    # MSFT could only be bought with what the AAPL sale brought in: 1000 * (1 - r) / (1 + r).
    assert by["MSFT"].cost == pytest.approx(1000.0 * (1 - STOCK_RATE) / (1 + STOCK_RATE))
    assert by["MSFT"].cost < 1000.0


# ---------------------------------------------------------------------------------------
# 5. MIN_TRADE_DOLLARS
# ---------------------------------------------------------------------------------------

def _small_limits():
    return Limits(money=1000, max_per_position=1000, max_per_model=1000)


def test_a_target_change_under_25_dollars_does_not_trade():
    data = make_data("stocks", {"AAPL": [100.0] * 4})
    # Buy 0.5 (~$500); equity is then 999.75, so 0.52 wants $519.87: only +$20.12 more.
    model = fake_model(("AAPL",), {0: {"AAPL": 0.5}, 1: {"AAPL": 0.52}}, every=1)
    run = run_backtest(data, model, None, 0, 4, _small_limits())
    assert run.trades[0].cost == pytest.approx(500.0)
    # ... and nor does a cut of about $20 (0.48 wants 479.88 against a position of 499.75).
    model = fake_model(("AAPL",), {0: {"AAPL": 0.5}, 1: {"AAPL": 0.48}}, every=1)
    run = run_backtest(data, model, None, 0, 4, _small_limits())
    assert run.trades[0].cost == pytest.approx(500.0)
    assert run.trades[0].proceeds == pytest.approx(500.0 / (1 + STOCK_RATE) * (1 - STOCK_RATE))  # only the final sale
    assert run.costs_paid == pytest.approx(500.0 / (1 + STOCK_RATE) * STOCK_RATE * 2)


def test_a_target_change_of_25_dollars_or_more_does_trade():
    data = make_data("stocks", {"AAPL": [100.0] * 4})
    model = fake_model(("AAPL",), {0: {"AAPL": 0.5}, 1: {"AAPL": 0.55}}, every=1)
    run = run_backtest(data, model, None, 0, 4, _small_limits())
    equity_before = 500.0 + 500.0 / (1 + STOCK_RATE)          # 999.75, marked at bar 0's close
    extra = 0.55 * equity_before - 500.0 / (1 + STOCK_RATE) * 1.0  # target minus what is held (at 100)
    assert extra >= MIN_TRADE_DOLLARS
    assert run.trades[0].cost == pytest.approx(500.0 + extra)


def test_small_first_buys_are_skipped_too():
    data = make_data("stocks", {"AAPL": [100.0] * 3})
    run = run_backtest(data, always("AAPL", 0.02, every=1), None, 0, 3, _small_limits())  # $20 target
    assert run.trades == [] and run.costs_paid == 0.0


def test_going_to_zero_sells_everything_even_a_few_dollars():
    # $1,000 bought at 100; the price falls to 1, so the position is worth ~$10 (under
    # the $25 minimum) and the model now wants nothing: it must still be sold.
    data = make_data("stocks", {"AAPL": [100.0, 1.0, 1.0, 1.0]})
    model = fake_model(("AAPL",), {0: {"AAPL": 1.0}, 1: {}}, every=1)
    run = run_backtest(data, model, None, 0, 4, _small_limits())
    (trade,) = run.trades
    assert trade.exit_t == int(data.times[1])
    qty = 1000.0 / (100.0 * (1 + STOCK_RATE))
    assert trade.proceeds == pytest.approx(qty * 1.0 * (1 - STOCK_RATE))
    assert qty * 1.0 < MIN_TRADE_DOLLARS


def test_going_to_zero_from_a_big_position_sells_all_of_it():
    data = make_data("stocks", {"AAPL": [100.0] * 4})
    model = fake_model(("AAPL",), {0: {"AAPL": 1.0}, 2: {}}, every=1)
    run = run_backtest(data, model, None, 0, 4)
    (trade,) = run.trades
    qty = 1000.0 / (100.0 * (1 + STOCK_RATE))
    assert trade.proceeds == pytest.approx(qty * 100.0 * (1 - STOCK_RATE))
    assert run.equity[-1] == pytest.approx(9000.0 + trade.proceeds)


# ---------------------------------------------------------------------------------------
# 6. missing bars
# ---------------------------------------------------------------------------------------

def test_a_symbol_without_a_bar_is_not_bought_then_but_at_a_later_decision():
    #                 0      1      2      3      4      5
    aapl = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    msft = [50.0, NAN, 50.0, 50.0, 50.0, 50.0]
    data = make_data("stocks", {"AAPL": aapl, "MSFT": msft})
    model = fake_model(("AAPL", "MSFT"), {0: {"AAPL": 0.5, "MSFT": 0.5}}, every=2)  # decisions: 1, 3, 5
    run = run_backtest(data, model, None, 1, 6, Limits(money=2000, max_per_position=1000, max_per_model=2000))
    by = {t.symbol: t for t in run.trades}
    assert by["AAPL"].entry_t == int(data.times[1])
    assert by["MSFT"].entry_t == int(data.times[3])  # no bar at 1: bought at the next decision
    # sized at bar 3 from equity at bar 2's close: half of 1,000 cash + 999.5 of AAPL
    assert by["MSFT"].cost == pytest.approx(0.5 * (1000.0 + 1000.0 / (1 + STOCK_RATE)))


def test_a_missing_bar_at_the_decision_leaves_a_held_symbol_alone_even_if_the_model_wants_out():
    aapl = [100.0, 100.0, NAN, 100.0, 100.0]
    data = make_data("stocks", {"AAPL": aapl})
    model = fake_model(("AAPL",), {1: {"AAPL": 1.0}, 2: {}}, every=1)  # sell wanted at bars 2, 3, 4
    run = run_backtest(data, model, None, 1, 5)
    (trade,) = run.trades
    assert trade.entry_t == int(data.times[1])
    assert trade.exit_t == int(data.times[3])  # bar 2 has no bar to trade at


def test_equity_uses_the_last_known_close_for_a_held_symbol_with_a_missing_bar():
    opens = {"AAPL": [100.0, 100.0, NAN, NAN, 100.0]}
    closes = {"AAPL": [100.0, 110.0, NAN, NAN, 130.0]}
    data = make_data("stocks", opens, closes)
    run = run_backtest(data, always("AAPL", every=100), None, 1, 5)
    qty = 1000.0 / (100.0 * (1 + STOCK_RATE))
    assert run.equity[0] == pytest.approx(9000.0 + qty * 110.0)
    assert run.equity[1] == pytest.approx(9000.0 + qty * 110.0)  # bar 2: no close, carry 110
    assert run.equity[2] == pytest.approx(9000.0 + qty * 110.0)
    assert run.equity[3] == pytest.approx(9000.0 + qty * 130.0 * (1 - STOCK_RATE))
    assert np.isfinite(run.equity).all()


def test_the_final_sale_uses_the_last_known_close_when_the_last_bar_is_missing():
    opens = {"AAPL": [100.0, 100.0, 100.0, NAN]}
    closes = {"AAPL": [100.0, 100.0, 120.0, NAN]}
    data = make_data("stocks", opens, closes)
    run = run_backtest(data, always("AAPL", every=100), None, 0, 4)
    qty = 1000.0 / (100.0 * (1 + STOCK_RATE))
    assert run.equity[-1] == pytest.approx(9000.0 + qty * 120.0 * (1 - STOCK_RATE))
    assert run.trades[0].exit_t == int(data.times[3])


def test_a_symbol_that_never_has_a_bar_is_never_traded_and_nothing_breaks():
    data = make_data("stocks", {"AAPL": [100.0] * 4, "MSFT": [NAN] * 4})
    model = fake_model(("AAPL", "MSFT"), {0: {"AAPL": 0.5, "MSFT": 0.5}}, every=1)
    run = run_backtest(data, model, None, 0, 4, Limits(money=2000, max_per_position=1000, max_per_model=2000))
    assert {t.symbol for t in run.trades} == {"AAPL"}
    assert np.isfinite(run.equity).all()


def test_a_held_position_with_a_missing_bar_still_counts_towards_the_model_cap():
    """Holding AAPL worth $1,000 (no bar at the decision, so it cannot be sold) and
    switching to MSFT must not push the total past max_per_model."""
    limits = Limits(money=10_000, max_per_position=1_000, max_per_model=1_000)
    data = make_data("stocks", {"AAPL": [100.0, 100.0, NAN, 100.0], "MSFT": [100.0, 100.0, 100.0, 100.0]})
    model = fake_model(("AAPL", "MSFT"), {0: {"AAPL": 1.0}, 2: {"MSFT": 1.0}}, every=1)
    invested = []

    def audit(history, params):
        book = sys._getframe(1).f_locals
        i = len(history)
        if i == 3:  # after the bar-2 decision: AAPL could not be sold, MSFT may have been bought
            invested.append(sum(q * 100.0 for q in book["shares"].values()))
        return model.target_positions(history, params)

    run_backtest(data, fake_model(("AAPL", "MSFT"), fn=audit, every=1), None, 0, 4, limits)
    assert invested and invested[0] <= limits.max_per_model + MIN_TRADE_DOLLARS


# ---------------------------------------------------------------------------------------
# 7. benchmark
# ---------------------------------------------------------------------------------------

def test_buy_and_hold_equity_by_hand_with_the_first_open_with_a_bar():
    #                 0     1      2      3
    opens = {"SPY": [NAN, 100.0, 110.0, 120.0]}
    closes = {"SPY": [NAN, 102.0, 105.0, 110.0]}
    data = make_data("stocks", opens, closes)
    out = buy_and_hold(data, "SPY", 0, 4, 10_000.0, COSTS["stocks"])
    qty = 10_000.0 / (100.0 * (1 + STOCK_RATE))  # first bar with an open is bar 1
    assert out[0] == 10_000.0                       # nothing bought yet
    assert out[1] == pytest.approx(qty * 102.0)
    assert out[2] == pytest.approx(qty * 105.0)
    assert out[3] == pytest.approx(qty * 110.0 * (1 - STOCK_RATE))  # net of the selling cost


def test_buy_and_hold_inside_a_later_period_starts_at_that_periods_first_open():
    data = make_data("stocks", {"SPY": [50.0, 60.0, 100.0, 110.0, 120.0]}, {"SPY": [55.0, 65.0, 105.0, 115.0, 125.0]})
    out = buy_and_hold(data, "SPY", 2, 5, 10_000.0, COSTS["stocks"])
    qty = 10_000.0 / (100.0 * (1 + STOCK_RATE))
    assert out.shape == (3,)
    assert out[0] == pytest.approx(qty * 105.0)
    assert out[1] == pytest.approx(qty * 115.0)
    assert out[2] == pytest.approx(qty * 125.0 * (1 - STOCK_RATE))


def test_buy_and_hold_with_crypto_costs_and_a_one_bar_period():
    data = make_data("crypto", {"BTC/USD": [100.0, 200.0]}, {"BTC/USD": [150.0, 300.0]})
    out = buy_and_hold(data, "BTC/USD", 0, 2, 10_000.0, COSTS["crypto"])
    qty = 10_000.0 / (100.0 * (1 + CRYPTO_RATE))
    assert out[0] == pytest.approx(qty * 150.0)
    assert out[1] == pytest.approx(qty * 300.0 * (1 - CRYPTO_RATE))


def test_buy_and_hold_with_no_bars_at_all_stays_at_the_money():
    data = make_data("stocks", {"SPY": [NAN] * 3})
    np.testing.assert_array_equal(buy_and_hold(data, "SPY", 0, 3, 10_000.0, COSTS["stocks"]), np.full(3, 10_000.0))


def test_a_run_carries_the_benchmark_over_the_same_period_with_the_same_costs():
    data = make_data("stocks", {"AAPL": [100.0] * 5, "SPY": [200.0, 200.0, 210.0, 220.0, 230.0]},
                     {"AAPL": [100.0] * 5, "SPY": [200.0, 205.0, 215.0, 225.0, 240.0]})
    run = run_backtest(data, always("AAPL", every=100), None, 1, 5, benchmark="SPY")
    qty = 10_000.0 / (200.0 * (1 + STOCK_RATE))  # period starts at bar 1: SPY open there is 200
    assert run.benchmark.shape == run.equity.shape == (4,)
    assert run.benchmark[0] == pytest.approx(qty * 205.0)
    assert run.benchmark[-1] == pytest.approx(qty * 240.0 * (1 - STOCK_RATE))
    s = summarize(run)
    assert s["benchmark_roi"] == pytest.approx(run.benchmark[-1] / 10_000.0 - 1.0)
    assert s["vs_buy_and_hold"] == pytest.approx(s["roi"] - s["benchmark_roi"])


def test_without_a_benchmark_the_run_has_none_and_the_summary_says_so():
    data = make_data("stocks", {"AAPL": [100.0] * 3})
    run = run_backtest(data, always("AAPL"), None, 0, 3)
    assert np.isnan(run.benchmark).all()
    s = summarize(run)
    assert s["benchmark_roi"] is None and s["vs_buy_and_hold"] is None
    assert curve(run)["benchmark"] == [None, None, None]


def test_holding_the_benchmark_itself_with_the_full_money_matches_buy_and_hold():
    """A model that puts everything into the benchmark at the first open, with caps that
    allow it, ends where buy and hold ends (same fills, same final sale)."""
    data = make_data("stocks", {"SPY": [100.0, 101.0, 103.0, 102.0, 108.0]}, {"SPY": [100.5, 102.0, 104.0, 103.0, 110.0]})
    limits = Limits(money=10_000.0, max_per_position=10_000.0, max_per_model=10_000.0)
    model = fake_model(("SPY",), {0: {"SPY": 1.0}}, every=100)
    run = run_backtest(data, model, None, 0, 5, limits, benchmark="SPY")
    assert run.equity[-1] == pytest.approx(run.benchmark[-1])
    assert run.equity[2] == pytest.approx(run.benchmark[2])


def test_buy_and_hold_with_an_integer_money_is_not_truncated_to_whole_dollars():
    data = make_data("stocks", {"SPY": [100.0, 101.0, 103.0, 102.0, 108.0]}, {"SPY": [100.5, 102.0, 104.0, 103.0, 110.0]})
    out = buy_and_hold(data, "SPY", 0, 5, 10_000, COSTS["stocks"])
    qty = 10_000.0 / (100.0 * (1 + STOCK_RATE))
    assert out[0] == pytest.approx(qty * 100.5, abs=1e-6)
    assert out[-1] == pytest.approx(qty * 110.0 * (1 - STOCK_RATE), abs=1e-6)


# ---------------------------------------------------------------------------------------
# 8. metrics
# ---------------------------------------------------------------------------------------

def make_run(equity, trades=(), benchmark=None, money=100.0, bars_per_year=252) -> Run:
    equity = np.asarray(equity, dtype=float)
    return Run(times=BASE + np.arange(equity.shape[0], dtype=np.int64) * 86_400, equity=equity,
               benchmark=np.full(equity.shape[0], NAN) if benchmark is None else np.asarray(benchmark, dtype=float),
               trades=list(trades), money=money, bars_per_year=bars_per_year, costs_paid=1.2345)


def test_max_drawdown_is_the_worst_fall_from_a_high():
    assert max_drawdown(np.array([100.0, 120.0, 90.0, 130.0, 117.0])) == pytest.approx(0.25)
    assert max_drawdown(np.array([100.0, 110.0, 120.0])) == 0.0
    assert max_drawdown(np.array([100.0, 50.0, 100.0, 75.0])) == pytest.approx(0.5)
    assert max_drawdown(np.array([])) == 0.0


def test_sharpe_is_none_for_constant_equity_and_for_too_few_bars():
    assert sharpe(np.full(10, 100.0), 252) is None
    assert sharpe(np.array([100.0, 110.0]), 252) is None
    assert sharpe(np.array([]), 252) is None


def test_sharpe_by_hand():
    # per-bar returns +10%, -10%, +10%: mean 0.1/3, sample sd 0.2/sqrt(3) -> sqrt(3)/6 per bar.
    equity = np.array([100.0, 110.0, 99.0, 108.9])
    assert sharpe(equity, 252) == pytest.approx(np.sqrt(3) / 6 * np.sqrt(252))
    assert sharpe(equity, 8760) == pytest.approx(np.sqrt(3) / 6 * np.sqrt(8760))
    # returns +10%, -10%: mean 0 -> a Sharpe of exactly 0.
    assert sharpe(np.array([100.0, 110.0, 99.0]), 252) == pytest.approx(0.0, abs=1e-12)


def _trade(pnl: float, entry: int = 0, exit_: int = 100) -> Trade:
    return Trade("AAPL", entry_t=entry, exit_t=exit_, cost=100.0, proceeds=100.0 + pnl)


def test_win_rate_profit_factor_and_hold_from_known_trades():
    trades = [_trade(10, 0, 100), _trade(30, 100, 400), _trade(-20, 400, 500), _trade(-10, 500, 700)]
    s = summarize(make_run([100.0, 101.0, 102.0], trades))
    assert s["trades"] == 4
    assert s["win_rate"] == 0.5
    assert s["profit_factor"] == pytest.approx(40.0 / 30.0)
    assert s["avg_hold_s"] == pytest.approx((100 + 300 + 100 + 200) / 4)
    assert s["enough_trades"] is False


def test_a_break_even_trade_counts_in_the_win_rate_denominator_only():
    s = summarize(make_run([100.0, 101.0, 102.0], [_trade(10), _trade(0)]))
    assert s["win_rate"] == 0.5
    assert s["profit_factor"] is None  # no losses


def test_profit_factor_is_none_without_losses_and_zero_with_only_losses():
    assert summarize(make_run([100.0, 101.0, 102.0], [_trade(5), _trade(7)]))["profit_factor"] is None
    s = summarize(make_run([100.0, 99.0, 98.0], [_trade(-5), _trade(-7)]))
    assert s["profit_factor"] == 0.0 and s["win_rate"] == 0.0


def test_no_trades_gives_none_for_the_trade_statistics():
    s = summarize(make_run([100.0, 100.0, 100.0]))
    assert s["trades"] == 0 and s["win_rate"] is None and s["profit_factor"] is None and s["avg_hold_s"] is None
    assert s["enough_trades"] is False and s["sharpe"] is None


def test_enough_trades_needs_at_least_100():
    assert MIN_TRADES == 100
    assert summarize(make_run([100.0, 101.0], [_trade(1)] * 99))["enough_trades"] is False
    assert summarize(make_run([100.0, 101.0], [_trade(1)] * 100))["enough_trades"] is True
    assert summarize(make_run([100.0, 101.0], [_trade(1)] * 101))["enough_trades"] is True


def test_summary_numbers_roi_benchmark_and_period():
    run = make_run([100.0, 120.0, 90.0, 130.0, 117.0], benchmark=[100.0, 101.0, 102.0, 103.0, 110.0])
    s = summarize(run)
    assert s["roi"] == pytest.approx(0.17)
    assert s["benchmark_roi"] == pytest.approx(0.10)
    assert s["vs_buy_and_hold"] == pytest.approx(0.07)
    assert s["max_drawdown"] == pytest.approx(0.25)
    assert s["costs_paid"] == 1.23
    assert s["start"] == int(run.times[0]) and s["end"] == int(run.times[-1])
    assert all(isinstance(s[k], float) for k in ("roi", "benchmark_roi", "vs_buy_and_hold", "max_drawdown", "sharpe"))
    json.dumps(s)  # the coordinator stores it as JSON


def test_a_losing_model_has_a_negative_roi():
    assert summarize(make_run([100.0, 90.0, 80.0]))["roi"] == pytest.approx(-0.2)


def test_curve_is_thinned_starts_at_100_and_ends_on_the_last_bar():
    n = 1000
    equity = 10_000.0 * (1 + np.arange(n) / n)
    run = make_run(equity, benchmark=np.linspace(10_000.0, 12_000.0, n), money=10_000.0)
    c = curve(run)
    assert len(c["t"]) == len(c["model"]) == len(c["benchmark"]) <= 240
    assert len(c["t"]) > 200
    assert c["model"][0] == 100.0 and c["benchmark"][0] == 100.0
    assert c["t"][0] == int(run.times[0]) and c["t"][-1] == int(run.times[-1])
    assert c["model"][-1] == pytest.approx(round(float(equity[-1]) / 100.0, 3))
    assert c["t"] == sorted(set(c["t"]))


def test_curve_of_a_short_run_has_every_bar():
    c = curve(make_run([100.0, 105.0, 110.0], money=100.0))
    assert c["t"] == [BASE, BASE + 86_400, BASE + 2 * 86_400]
    assert c["model"] == [100.0, 105.0, 110.0]
    assert c["benchmark"] == [None, None, None]
    assert curve(make_run([100.0]))["model"] == [100.0]


def test_a_real_backtests_curve_starts_near_100_for_the_model():
    data = synthetic("stocks", 300, seed=11)
    model = REGISTRY["momentum"]
    run = run_backtest(data, model, None, 130, 300, benchmark="SPY")
    c = summarize(run)["curve"]
    assert len(c["t"]) <= 240
    assert abs(c["model"][0] - 100.0) < 2.0  # the first bar is marked after the first buys' costs
    assert c["benchmark"][0] == pytest.approx(100.0, abs=3.0)


# ---------------------------------------------------------------------------------------
# 9. split_index, backtest_periods, summary_line
# ---------------------------------------------------------------------------------------

def test_split_index_holds_out_the_last_quarter_of_the_bars():
    for n in (100, 400, 20, 1000):
        data = synthetic("stocks", n)
        split = split_index(data, 0.25)
        assert split == int(round(n * 0.75))
        assert n - split == pytest.approx(n * 0.25, abs=0.5)
    assert split_index(synthetic("stocks", 100), 0.25) == 75
    assert split_index(synthetic("stocks", 100), 0.0) == 100
    assert split_index(synthetic("stocks", 100), 0.5) == 50


def _stock_data_with_spy(n=40, seed=5) -> MarketData:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (2, n)), axis=1))
    open_ = close * np.exp(rng.normal(0, 0.003, close.shape))
    times = BASE + np.arange(n, dtype=np.int64) * 86_400
    return MarketData("stocks", "1Day", "test", ("AAPL", "SPY"), times, open_, np.maximum(open_, close),
                      np.minimum(open_, close), close, np.full(close.shape, 1e6))


def test_backtest_periods_trains_before_the_split_and_holds_out_the_last_quarter():
    data = _stock_data_with_spy(40)
    model = always("AAPL", every=5, warmup=3)
    result = backtest_periods(data, model, {}, Limits(), 0.25)

    assert set(result) == {"train", "held_out", "split_t"}
    split = split_index(data, 0.25)
    assert split == 30
    assert result["split_t"] == int(data.times[split])
    train, held = result["train"], result["held_out"]
    assert train["start"] == int(data.times[3])                 # after the warm-up
    assert train["end"] == int(data.times[split - 1])            # ends before the split
    assert train["end"] < result["split_t"]
    assert held["start"] == result["split_t"]                    # the held-out period is the last 25%
    assert held["end"] == int(data.times[-1])
    n_held = round((held["end"] - held["start"]) / 86_400) + 1
    assert n_held == 10 == 40 - split
    for part in (train, held):
        assert part["trades"] >= 1 and part["benchmark_roi"] is not None
        assert part["curve"]["t"][-1] == part["end"]


def test_the_held_out_run_can_use_training_bars_as_history_but_never_trades_them():
    data = _stock_data_with_spy(40)
    lengths = []
    model = fake_model(("AAPL",), fn=lambda h, p: lengths.append(len(h)) or {"AAPL": 1.0}, every=1, warmup=3)
    backtest_periods(data, model, {}, Limits(), 0.25)
    # train decides at bars 3..29, then held-out at bars 30..39: each sees all earlier bars.
    assert lengths == list(range(3, 30)) + list(range(30, 40))


def test_warmup_longer_than_the_training_period_still_leaves_one_training_bar():
    data = _stock_data_with_spy(40)
    result = backtest_periods(data, always("AAPL", every=5, warmup=500), {}, Limits(), 0.25)
    assert result["train"]["start"] == result["train"]["end"] == int(data.times[29])


def test_backtest_periods_reports_progress_up_to_one_and_passes_should_stop_through():
    data = _stock_data_with_spy(40)
    seen: list[tuple[float, str]] = []
    backtest_periods(data, always("AAPL", every=5, warmup=3), {}, Limits(), 0.25, lambda f, d: seen.append((f, d)))
    fracs = [f for f, _ in seen]
    assert fracs == sorted(fracs) and 0 < fracs[0] and fracs[-1] == pytest.approx(1.0)
    assert any("training" in d for _, d in seen) and any("held-out" in d for _, d in seen)
    with pytest.raises(JobStopped):
        backtest_periods(data, always("AAPL", every=5, warmup=3), {}, Limits(), 0.25, should_stop=lambda: True)


def _result(roi, bench, trades, enough, market="stocks"):
    return {"market": market, "held_out": {"roi": roi, "benchmark_roi": bench, "trades": trades, "enough_trades": enough}}


def test_summary_line_text():
    assert summary_line(_result(0.042, 0.031, 124, True)) == "Held-out ROI +4.2% vs SPY +3.1% · 124 trades"


def test_summary_line_uses_a_real_minus_sign_for_losses_but_keeps_the_hyphen_in_held_out():
    line = summary_line(_result(-0.031, -0.012, 150, True))
    assert line == f"Held-out ROI {MINUS}3.1% vs SPY {MINUS}1.2% · 150 trades"
    assert MINUS in line and "-" in line  # the ordinary hyphen of "Held-out" stays
    assert line.startswith("Held-out")
    assert "-3.1" not in line and "-1.2" not in line  # no ASCII minus in front of a number
    assert line.count("-") == 1


def test_summary_line_flags_too_few_trades():
    assert summary_line(_result(0.042, 0.031, 99, False)).endswith(" · 99 trades (not enough trades)")
    assert summary_line(_result(0.042, 0.031, 100, True)).endswith("100 trades")
    assert summary_line(_result(0.0, 0.0, 0, False)) == "Held-out ROI +0.0% vs SPY +0.0% · 0 trades (not enough trades)"


def test_summary_line_names_the_crypto_benchmark_and_omits_a_missing_one():
    assert summary_line(_result(0.1, 0.2, 120, True, "crypto")) == "Held-out ROI +10.0% vs BTC +20.0% · 120 trades"
    assert summary_line(_result(0.1, None, 120, True)) == "Held-out ROI +10.0% · 120 trades"


def test_signed_pct():
    assert signed_pct(0.042) == "+4.2%"
    assert signed_pct(-0.031) == f"{MINUS}3.1%"
    assert signed_pct(0.0) == "+0.0%"
    assert signed_pct(1.5) == "+150.0%"
    assert MINUS == "−" and MINUS != "-"


# ---------------------------------------------------------------------------------------
# 10. determinism
# ---------------------------------------------------------------------------------------

def test_the_same_inputs_give_identical_runs():
    data = synthetic("stocks", 500, seed=21)
    model = REGISTRY["momentum"]
    a = run_backtest(data, model, {"top_n": 5}, 130, 500, benchmark="SPY")
    b = run_backtest(data, model, {"top_n": 5}, 130, 500, benchmark="SPY")
    np.testing.assert_array_equal(a.equity, b.equity)
    np.testing.assert_array_equal(a.benchmark, b.benchmark)
    np.testing.assert_array_equal(a.times, b.times)
    assert a.trades == b.trades and a.costs_paid == b.costs_paid and len(a.trades) > 0


def test_backtest_periods_is_deterministic_and_does_not_change_the_data():
    data = synthetic("crypto", 900, seed=2)
    before = data.close.copy(), data.open.copy()
    model = REGISTRY["crypto_trend"]
    one = backtest_periods(data, model, dict(model.DEFAULT_PARAMS), Limits(), 0.25)
    two = backtest_periods(data, model, dict(model.DEFAULT_PARAMS), Limits(), 0.25)
    assert json.dumps(one, sort_keys=True) == json.dumps(two, sort_keys=True)
    np.testing.assert_array_equal(data.close, before[0])
    np.testing.assert_array_equal(data.open, before[1])


# ---------------------------------------------------------------------------------------
# 11. the real models end to end
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_real_model_runs_end_to_end_with_valid_weights_and_trades(name):
    model = REGISTRY[name]
    market = model.MARKET
    n = 700 if market == "stocks" else 1500
    data = synthetic(market, n, seed=7)
    params = dict(model.DEFAULT_PARAMS)
    decisions: list[dict[str, float]] = []

    def checked(history, p):
        raw = model.target_positions(history, p)
        decisions.append(clean_targets(raw, model.SYMBOLS))  # raises BadTargets on a bad answer
        return raw

    wrapped = SimpleNamespace(SYMBOLS=model.SYMBOLS, DEFAULT_PARAMS=model.DEFAULT_PARAMS,
                              rebalance_every=model.rebalance_every, warmup=model.warmup, target_positions=checked)
    benchmark = MARKETS[market]["benchmark"]
    start = model.warmup(params)
    run = run_backtest(data, wrapped, params, start, n, Limits(), benchmark, MARKETS[market]["bars_per_year"])

    every = max(1, int(model.rebalance_every(params)))
    assert len(decisions) == len(range(start, n, every))
    assert all(0 <= w <= 1 for d in decisions for w in d.values())
    assert all(sum(d.values()) <= 1 + 1e-9 for d in decisions)
    assert run.equity.shape == (n - start,) and np.isfinite(run.equity).all() and (run.equity > 0).all()
    assert np.isfinite(run.benchmark).all()
    assert len(run.trades) > 0, f"{name} made no trades with default params"
    assert all(t.symbol in model.SYMBOLS and t.cost > 0 and t.exit_t >= t.entry_t for t in run.trades)
    s = summarize(run)
    assert s["trades"] == len(run.trades) > 0
    assert 0.0 <= s["max_drawdown"] < 1.0 and s["costs_paid"] > 0
    json.dumps(s)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_real_model_through_backtest_periods(name):
    model = REGISTRY[name]
    market = model.MARKET
    n = 700 if market == "stocks" else 1500
    data = synthetic(market, n, seed=13)
    result = backtest_periods(data, model, dict(model.DEFAULT_PARAMS), Limits(), 0.25)
    split = split_index(data, 0.25)
    assert result["split_t"] == int(data.times[split])
    assert result["train"]["end"] < result["split_t"] == result["held_out"]["start"]
    assert result["held_out"]["end"] == int(data.times[-1])
    assert result["train"]["trades"] > 0
    line = summary_line({**result, "market": market})
    assert line.startswith("Held-out ROI ") and f" · {result['held_out']['trades']} trades" in line
    assert ("(not enough trades)" in line) == (not result["held_out"]["enough_trades"])


# ---------------------------------------------------------------------------------------
# control: stop and progress
# ---------------------------------------------------------------------------------------

def test_should_stop_raises_job_stopped_and_progress_ends_at_one():
    data = synthetic("stocks", 120)
    fracs: list[float] = []
    run_backtest(data, always("AAPL", every=5), None, 0, 120, progress=fracs.append)
    assert fracs == sorted(fracs) and 0 < fracs[0] and fracs[-1] > 0.95 and max(fracs) <= 1.0
    calls = []
    with pytest.raises(JobStopped):
        run_backtest(data, always("AAPL", every=5), None, 0, 120, should_stop=lambda: calls.append(1) or len(calls) > 10)
    assert len(calls) == 11


def test_t_stat_is_the_sharpe_ratio_times_the_root_of_the_years_tested():
    rng = np.random.default_rng(5)
    equity = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.01, 504))
    s = summarize(make_run(equity))
    assert s["years"] == pytest.approx(2.0)
    assert s["t_stat"] == pytest.approx(s["sharpe"] * np.sqrt(2.0))
    # the same thing as the mean per-bar return over its standard error
    full = np.concatenate(([100.0], equity))
    r = np.diff(full) / full[:-1]
    assert s["t_stat"] == pytest.approx(np.mean(r) / np.std(r, ddof=1) * np.sqrt(r.shape[0]))
    assert t_stat(None, 504, 252) is None and t_stat(1.0, 0, 252) is None
    assert summarize(make_run([100.0, 100.0, 100.0]))["t_stat"] is None  # no ups and downs


def test_beta_and_alpha_recover_a_known_exposure():
    rng = np.random.default_rng(11)
    rb = rng.normal(0.0005, 0.01, 400)
    rm = 0.5 * rb + 0.0002  # half the market's moves, plus 0.02% a bar of its own
    bench = 100.0 * np.cumprod(np.concatenate(([1.0], 1.0 + rb)))
    model = 100.0 * np.cumprod(np.concatenate(([1.0], 1.0 + rm)))
    beta, alpha = beta_alpha(model, bench, 252)
    assert beta == pytest.approx(0.5)
    assert alpha == pytest.approx(0.0002 * 252)
    s = summarize(make_run(model[1:], benchmark=bench[1:]))
    assert s["beta"] == pytest.approx(0.5) and s["alpha"] == pytest.approx(0.0002 * 252)


def test_beta_and_alpha_are_none_without_a_benchmark_or_without_benchmark_moves():
    s = summarize(make_run([100.0, 101.0, 99.0, 102.0, 103.0]))
    assert s["beta"] is None and s["alpha"] is None
    s = summarize(make_run([100.0, 101.0, 99.0, 102.0, 103.0], benchmark=[100.0] * 5))
    assert s["beta"] is None and s["alpha"] is None
    assert beta_alpha(np.array([100.0, 101.0]), np.array([100.0, 102.0]), 252) == (None, None)
