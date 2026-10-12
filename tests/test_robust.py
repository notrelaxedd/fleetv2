"""The stress tests model search runs on its stock and crypto finds (fleet2.sim.robust),
on tiny hand-made prices whose right answer is obvious: a model that only worked in one
stretch, one that only works at one exact setting, one whose edge is smaller than
double costs, and the coin-flip twins' determinism and blindness to later bars."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from fleet2.models import REGISTRY
from fleet2.sim import robust
from fleet2.sim.backtest import Limits, run_backtest
from fleet2.sim.control import JobStopped
from fleet2.sim.marketdata import History
from fleet2.sim.metrics import MIN_TRADES, summarize

from tests.test_backtest import make_data
from tests.test_lookahead import scramble_from, synthetic

ALL_IN = Limits(money=10_000.0, max_per_position=10_000.0, max_per_model=10_000.0)


def model(symbols, fn, space=None, defaults=None, every=1, market="stocks"):
    """A model module from a decision function, with an optional search space."""
    return SimpleNamespace(NAME="fake", MARKET=market, SYMBOLS=tuple(symbols), DEFAULT_PARAMS=dict(defaults or {}),
                           SEARCH_SPACE=dict(space or {}), rebalance_every=lambda params: every,
                           warmup=lambda params: 0, target_positions=fn)


def trend(n, start, per_bar_pct):
    return list(start * np.cumprod(np.full(n, 1.0 + per_bar_pct / 100.0)))


# ---------------------------------------------------------------------------------------
# score and folds
# ---------------------------------------------------------------------------------------

def test_fold_bounds_are_back_to_back_and_cover_training():
    assert robust.fold_bounds(100, 500) == [(100, 200), (200, 300), (300, 400), (400, 500)]
    parts = robust.fold_bounds(7, 30, 4)
    assert parts[0][0] == 7 and parts[-1][1] == 30
    assert all(a[1] == b[0] for a, b in zip(parts, parts[1:]))
    sizes = [b - a for a, b in parts]
    assert max(sizes) - min(sizes) <= 1
    with pytest.raises(ValueError):
        robust.fold_bounds(0, 7, 4)


def test_training_bounds_match_the_search_and_start_after_the_warmup():
    m = REGISTRY["momentum"]
    start, stop = robust.training_bounds(m, None, 600)
    assert (start, stop) == (m.warmup(m.DEFAULT_PARAMS), 600)
    assert robust.fold_bounds(start, stop)[0][0] >= m.warmup(m.DEFAULT_PARAMS)
    assert robust.training_bounds(m, None, 50) == (49, 50)  # too short: as search_job.evaluate


def test_score_is_flat_in_trades_past_min_trades():
    base = {"roi": 0.30, "benchmark_roi": 0.10, "years": 1.0}
    at = {n: robust.score({**base, "trades": n}) for n in (MIN_TRADES, 300, 5_000)}
    assert at[MIN_TRADES] == pytest.approx(0.20)
    assert at[300] == at[MIN_TRADES] == at[5_000]
    # Fewer trades shrink a gain in proportion...
    assert robust.score({**base, "trades": MIN_TRADES // 2}) == pytest.approx(0.10)
    assert robust.score({**base, "trades": 0}) == 0.0
    # ...but never soften a loss.
    loss = {"roi": -0.05, "benchmark_roi": 0.10, "years": 1.0}
    assert robust.score({**loss, "trades": 3}) == robust.score({**loss, "trades": 3_000}) < 0


def test_score_is_yearly_and_robust_to_missing_numbers():
    # 21% over two years is 10% a year; buy and hold flat.
    assert robust.score({"roi": 0.21, "benchmark_roi": 0.0, "years": 2.0, "trades": 500}) == pytest.approx(0.10)
    # A short stretch is not scaled up to a year.
    assert robust.score({"roi": 0.05, "benchmark_roi": 0.0, "years": 0.1, "trades": 500}) == pytest.approx(0.05)
    # Sitting out a crash beats buy and hold.
    assert robust.score({"roi": -0.02, "benchmark_roi": -0.30, "years": 1.0, "trades": 500}) > 0
    assert robust.score(None) == 0.0
    assert robust.score({}) == 0.0
    assert robust.score({"roi": None, "trades": None}) == 0.0
    assert robust.score({"roi": 0.1, "benchmark_roi": None, "years": None, "trades": 500}) == pytest.approx(0.1)
    assert robust.score({"roi": float("nan"), "trades": 500}) == 0.0
    assert robust.score({"roi": -1.5, "benchmark_roi": 0.0, "years": 1.0, "trades": 500}) == -1.0


# ---------------------------------------------------------------------------------------
# the flags
# ---------------------------------------------------------------------------------------

def test_a_one_fold_wonder_is_inconsistent():
    """AAPL soars in the first quarter of training and sinks after; holding it beats a
    flat SPY in one part only. The whole-period ROI still looks fine."""
    n = 400
    data = make_data("stocks", {"SPY": [100.0] * n, "AAPL": trend(100, 100.0, 1.0) + trend(300, 270.0, -0.15)})
    holder = model(("AAPL",), lambda history, params: {"AAPL": 1.0})
    whole = summarize(run_backtest(data, holder, None, 0, n, ALL_IN, "SPY", 252))
    assert whole["vs_buy_and_hold"] > 0
    got = robust.report(data, holder, None, 0, n, ALL_IN, n_draws=3)
    assert got["positive_folds"] == 1
    assert "inconsistent" in got["flags"]
    assert got["score"] < 0 and not got["passed"]
    assert len(got["folds"]) == 4 and all("curve" not in f for f in got["folds"])


def test_a_knife_edge_setting_is_fragile_and_a_smooth_one_is_not():
    """At exactly x = 50 the model holds the riser; at any other x it holds the faller."""
    n = 400
    data = make_data("stocks", {"SPY": [100.0] * n, "AAPL": trend(n, 100.0, 0.3), "MSFT": trend(n, 100.0, -0.3)})
    space = {"x": (0, 100, "int"), "y": (0.0, 1.0, "float")}

    def knife(history, params):
        return {"AAPL": 1.0} if int(params["x"]) == 50 else {"MSFT": 1.0}

    def smooth(history, params):
        return {"AAPL": 1.0}

    sharp = model(("AAPL", "MSFT"), knife, space, {"x": 50, "y": 0.5})
    near = robust.neighbours(sharp, None)
    moves = [(name, sign, p[name]) for name, sign, p in near]
    assert moves == [("x", -1, 40), ("x", 1, 60), ("y", -1, 0.4), ("y", 1, 0.6)]
    got = robust.report(data, sharp, None, 0, n, ALL_IN, n_draws=3)
    assert got["score"] > 0 and got["positive_folds"] == 4
    assert got["neighbours"]["tried"] == 4
    assert "fragile" in got["flags"] and not got["passed"]

    steady = robust.neighbour_check(data, model(("AAPL", "MSFT"), smooth, space, {"x": 50, "y": 0.5}), None,
                                    robust.fold_bounds(0, n), ALL_IN)
    assert not steady["fragile"] and steady["share"] == pytest.approx(1.0)


def test_neighbours_stay_in_range_and_skip_moves_that_change_nothing():
    m = model(("AAPL",), lambda h, p: {}, {"a": (5, 20, "int"), "b": (2.0, 10.0, "float"), "c": (1, 3, "choice")},
              {"a": 5, "b": 9.5, "c": 2})
    got = {(name, sign): p[name] for name, sign, p in robust.neighbours(m, None)}
    assert got == {("a", 1): 7, ("b", -1): 8.7, ("b", 1): 10.0}


def test_an_edge_smaller_than_double_costs_is_cost_sensitive():
    """ETH opens at 100 and 101 in turn; buying the 100s and selling the 101s makes 1%
    a round trip, against 0.7% of crypto costs (1.4% at double costs)."""
    n = 400
    eth = [100.0 if i % 2 == 0 else 101.0 for i in range(n + 1)]
    data = make_data("crypto", {"BTC/USD": [100.0] * n, "ETH/USD": eth[:n]},
                     {"BTC/USD": [100.0] * n, "ETH/USD": eth[1:]})
    flipper = model(("ETH/USD",), lambda history, params: {"ETH/USD": 1.0} if len(history) % 2 == 0 else {},
                    market="crypto")
    got = robust.report(data, flipper, None, 0, n, ALL_IN, n_draws=5)
    assert got["score"] > 0 and got["positive_folds"] == 4
    assert got["cost_stress"]["score"] < 0
    assert got["flags"] == ["cost_sensitive"]
    assert got["coin_flip"]["percentile"] == 100.0  # random timing cannot find the pattern


def test_a_churny_model_on_random_walks_is_cost_sensitive():
    data = synthetic("crypto", 600, seed=5)
    coins = ("ETH/USD", "SOL/USD", "LTC/USD")
    churn = model(coins, lambda history, params: {coins[len(history) % 3]: 1.0}, market="crypto")
    got = robust.cost_stress(data, churn, None, robust.fold_bounds(0, 600), ALL_IN)
    assert got["cost_sensitive"] and got["score"] < 0 and got["multiplier"] == 2.0


# ---------------------------------------------------------------------------------------
# coin-flip twins
# ---------------------------------------------------------------------------------------

HABIT = {"slots": 3, "weight": 0.2, "share": 0.4, "hold_bars": 5}


def _decisions(twin, data, ts):
    return [twin.target_positions(History(data, t), {}) for t in ts]


def test_coin_flip_twin_is_deterministic_and_never_looks_ahead():
    m = REGISTRY["momentum"]
    data = synthetic("stocks", 300)
    ts = range(10, 300, 7)
    twin = robust.coin_flip_twin(m, HABIT, seed=4, draw=2)
    real = _decisions(twin, data, ts)
    assert any(real) and all(sum(d.values()) <= 1.0 + 1e-9 and set(d) <= set(m.SYMBOLS) for d in real)
    for t, d in zip(ts, real):
        assert twin.target_positions(History(scramble_from(data, t), t), {}) == d, f"twin changed at bar {t}"
    assert _decisions(robust.coin_flip_twin(m, HABIT, seed=4, draw=2), data, ts) == real
    assert _decisions(robust.coin_flip_twin(m, HABIT, seed=4, draw=3), data, ts) != real
    # The twin's backtest up to a bar ignores every later bar too.
    cut = 200
    a = run_backtest(data, twin, None, 10, 300, Limits())
    b = run_backtest(scramble_from(data, cut + 1), twin, None, 10, 300, Limits())
    np.testing.assert_array_equal(a.equity[: cut - 10 + 1], b.equity[: cut - 10 + 1])


def test_coin_flip_twin_copies_the_models_habits():
    m = REGISTRY["momentum"]
    data = synthetic("stocks", 900, seed=2)
    start, stop = robust.training_bounds(m, None, 900)
    folds = robust.fold_bounds(start, stop)
    got = robust.coin_flip_baseline(data, m, None, folds, Limits(), n_draws=4, seed=1)
    habit = got["habits"]
    assert habit["slots"] == m.DEFAULT_PARAMS["top_n"] and habit["weight"] == pytest.approx(0.1)
    assert habit["hold_bars"] % m.DEFAULT_PARAMS["rebalance_every"] == 0
    real_trades = robust.fold_score(data, m, None, folds, Limits())["trades"]
    assert 0.6 * real_trades <= got["twin_trades"] <= 1.4 * real_trades
    assert got["draws"] == 4 and len(got["twin_scores"]) == 4
    assert got == robust.coin_flip_baseline(data, m, None, folds, Limits(), n_draws=4, seed=1)
    assert got["twin_scores"] != robust.coin_flip_baseline(data, m, None, folds, Limits(), n_draws=4,
                                                           seed=2)["twin_scores"]


# ---------------------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------------------

def test_report_is_compact_json_and_repeats_exactly():
    m = REGISTRY["momentum"]
    data = synthetic("stocks", 700)
    start, stop = robust.training_bounds(m, None, 700)
    seen = []
    got = robust.report(data, m, None, start, stop, Limits(), lambda: False, n_draws=3, seed=7,
                        progress=seen.append)
    text = json.dumps(got, allow_nan=False)
    assert json.loads(text) == got
    assert len(text) < 5_000 and "curve" not in text
    assert set(got) == {"score", "folds", "trades", "positive_folds", "neighbours", "cost_stress", "coin_flip",
                        "flags", "passed"}
    assert set(got["flags"]) <= set(robust.FLAGS)
    assert got["passed"] == (not got["flags"] and got["score"] > 0)
    # rebalance_every's default sits at the bottom of its range: only the move up is tried.
    assert got["neighbours"]["tried"] == len(robust.neighbours(m, None)) == 2 * len(m.SEARCH_SPACE) - 1
    assert seen and seen == sorted(seen) and seen[-1] == pytest.approx(1.0)
    assert robust.report(data, m, None, start, stop, Limits(), n_draws=3, seed=7) == got


def test_report_stops_when_asked():
    m = REGISTRY["momentum"]
    data = synthetic("stocks", 700)
    calls = []

    def stop_soon():
        calls.append(1)
        return len(calls) > 50

    with pytest.raises(JobStopped):
        robust.report(data, m, None, *robust.training_bounds(m, None, 700), Limits(), stop_soon, n_draws=3)
