"""The three price-only models adapted from published effects (short-term reversal,
52-week high, low volatility): each picks what it says it picks."""
from __future__ import annotations

import numpy as np

from fleet2.models import low_vol, near_high, reversal
from fleet2.models.base import clean_targets, params_with_defaults
from fleet2.sim.marketdata import History, MarketData

BASE = 1_600_000_000


def stocks(closes: dict[str, np.ndarray]) -> MarketData:
    """Daily stock data from per-symbol closes (opens equal closes)."""
    symbols = tuple(closes)
    c = np.array([closes[s] for s in symbols], dtype=float)
    times = BASE + np.arange(c.shape[1], dtype=np.int64) * 86_400
    return MarketData("stocks", "1Day", "test", symbols, times, c, c, c, c, np.full(c.shape, 1e6))


def flat(n: int, level: float = 100.0) -> np.ndarray:
    return np.full(n, level)


def decide(model, data: MarketData, **params) -> dict[str, float]:
    p = params_with_defaults(model, params)
    return clean_targets(model.target_positions(History(data, data.n_bars), p), model.SYMBOLS)


def test_reversal_buys_the_biggest_losers_that_actually_fell():
    n = 10
    closes = {s: flat(n) for s in reversal.SYMBOLS}
    closes["AAPL"] = np.linspace(100, 90, n)   # -10%
    closes["MSFT"] = np.linspace(100, 95, n)   # -5%
    closes["KO"] = np.linspace(100, 99.5, n)   # -0.5%
    closes["NVDA"] = np.linspace(100, 120, n)  # up: never bought
    out = decide(reversal, stocks(closes), lookback=5, top_n=3, min_drop_pct=0.0)
    assert set(out) == {"AAPL", "MSFT", "KO"} and all(w == 1 / 3 for w in out.values())
    out = decide(reversal, stocks(closes), lookback=5, top_n=3, min_drop_pct=1.0)
    assert set(out) == {"AAPL", "MSFT"}  # KO fell less than 1%
    assert decide(reversal, stocks({s: flat(n) for s in reversal.SYMBOLS})) == {}  # nothing fell


def test_near_high_holds_stocks_close_to_their_yearly_high():
    n = 30
    closes = {s: np.concatenate((flat(n - 1, 120.0), [100.0])) for s in near_high.SYMBOLS}  # 17% below
    closes["AAPL"] = np.linspace(100, 130, n)                          # at its high
    closes["MSFT"] = np.concatenate((flat(n - 1, 100.0), [97.0]))      # 3% below
    closes["KO"] = np.concatenate((flat(n - 1, 100.0), [92.0]))        # 8% below
    out = decide(near_high, stocks(closes), lookback=20, top_n=5, within_pct=5.0)
    assert set(out) == {"AAPL", "MSFT"} and all(w == 1 / 5 for w in out.values())
    assert set(decide(near_high, stocks(closes), lookback=20, top_n=1, within_pct=5.0)) == {"AAPL"}
    assert decide(near_high, stocks(closes), lookback=40) == {}  # not enough history yet


def test_low_vol_holds_the_calmest_stocks():
    rng = np.random.default_rng(3)
    n = 80
    closes = {s: 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n))) for s in low_vol.SYMBOLS}
    closes["KO"] = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    closes["PG"] = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    out = decide(low_vol, stocks(closes), lookback=63, top_n=2)
    assert set(out) == {"KO", "PG"} and all(w == 0.5 for w in out.values())
    closes["KO"][-5] = np.nan  # a missing bar in the window: skipped, never guessed
    assert "KO" not in decide(low_vol, stocks(closes), lookback=63, top_n=2)
