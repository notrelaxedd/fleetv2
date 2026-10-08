"""Proof that no model sees the future: a decision at bar T is the same whatever the
bars from T on contain, and a backtest's equity up to bar T is the same too."""
from __future__ import annotations

import numpy as np
import pytest

from fleet2.models import REGISTRY
from fleet2.models.base import clean_targets, params_with_defaults
from fleet2.sim.backtest import Limits, run_backtest
from fleet2.sim.marketdata import History, LookaheadError, MarketData
from fleet2.universe import MARKETS


def synthetic(market: str, n: int, seed: int = 7) -> MarketData:
    spec = MARKETS[market]
    syms = spec["symbols"]
    rng = np.random.default_rng(seed)
    step = 86400 if market == "stocks" else 3600
    times = np.arange(n, dtype=np.int64) * step + 1_600_000_000
    vol = 0.02 if market == "stocks" else 0.01
    close = 100 * np.exp(np.cumsum(rng.normal(0.0, vol, (len(syms), n)), axis=1))
    open_ = close * np.exp(rng.normal(0, vol / 2, close.shape))
    return MarketData(market, spec["timeframe"], "test", syms, times, open_, np.maximum(open_, close),
                      np.minimum(open_, close), close, np.full(close.shape, 1e6))


def scramble_from(data: MarketData, t: int, seed: int = 99) -> MarketData:
    """The same data with every bar from index t on replaced by wild random values."""
    rng = np.random.default_rng(seed)
    arrays = {}
    for name in ("open", "high", "low", "close", "volume"):
        arr = getattr(data, name).copy()
        arr[:, t:] = rng.uniform(1, 1000, arr[:, t:].shape)
        arrays[name] = arr
    return MarketData(data.market, data.timeframe, data.feed, data.symbols, data.times, **arrays)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_decision_at_t_ignores_bars_from_t_on(name):
    model = REGISTRY[name]
    params = params_with_defaults(model, None)
    n = 700 if model.MARKET == "stocks" else 1500
    data = synthetic(model.MARKET, n)
    decided = 0
    for t in range(model.warmup(params), n, 37):
        real = clean_targets(model.target_positions(History(data, t), params), model.SYMBOLS)
        fake = clean_targets(model.target_positions(History(scramble_from(data, t), t), params), model.SYMBOLS)
        assert real == fake, f"{name} decision at bar {t} changed when later bars changed"
        decided += bool(real)
    assert decided > 0, f"{name} never wanted a position, so the test proved nothing"


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_backtest_equity_up_to_t_ignores_later_bars(name):
    model = REGISTRY[name]
    n = 600 if model.MARKET == "stocks" else 1400
    data = synthetic(model.MARKET, n, seed=3)
    cut = n - 120
    start = model.warmup(params_with_defaults(model, None))
    a = run_backtest(data, model, None, start, n, Limits())
    b = run_backtest(scramble_from(data, cut + 1), model, None, start, n, Limits())
    # Bar `cut` closes before bar cut+1 exists, so everything through it must match.
    np.testing.assert_array_equal(a.equity[: cut - start + 1], b.equity[: cut - start + 1])


def test_history_refuses_future_bars_and_hands_out_copies():
    data = synthetic("stocks", 50)
    h = History(data, 20)
    assert len(h) == 20 and h.close("AAPL").shape == (20,)
    with pytest.raises(LookaheadError):
        h.at("AAPL", 20)
    with pytest.raises(LookaheadError):
        History(data, 51)
    view = h.close("AAPL")
    view[:] = -1.0
    assert (data.close[data.row("AAPL"), :20] > 0).all()
    assert h.close("AAPL", bars=5).shape == (5,)
    assert h.now == int(data.times[20])


def test_the_check_catches_a_model_that_peeks():
    """The comparison above would notice a model that reads a bar it should not see."""

    def peeking(history, params):
        data, i = history._data, len(history)  # reaches around the guard on purpose
        row = data.row("AAPL")
        return {"AAPL": 1.0} if data.open[row, i] > data.close[row, i - 1] else {}

    data = synthetic("stocks", 300)
    changed = sum(peeking(History(data, t), {}) != peeking(History(scramble_from(data, t), t), {})
                  for t in range(10, 300, 7))
    assert changed > 0
