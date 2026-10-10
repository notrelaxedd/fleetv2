"""52-week high: hold the stocks trading closest to their highest price of the past year."""
from __future__ import annotations

from typing import Any

import numpy as np

from fleet2.universe import STOCKS

NAME = "Near yearly high"
MARKET = "stocks"
DESCRIPTION = "Holds the stocks trading closest to their highest price of the past year."
HOW_IT_WORKS = (
    "Every few weeks it compares each stock's price with its highest price of the last year. "
    "It buys the ones closest to that high, but only ones within a few percent of it, and sells any that slip further away. "
    "Investors tend to hesitate to push a stock past its old high, so good news often takes a while to show in the price."
)
SYMBOLS = tuple(s for s in STOCKS if s not in ("SPY", "QQQ"))
DEFAULT_PARAMS: dict[str, Any] = {"lookback": 252, "top_n": 10, "within_pct": 5.0, "rebalance_every": 21}
SEARCH_SPACE = {
    "lookback": (126, 252, "int"),
    "top_n": (3, 10, "int"),
    "within_pct": (1.0, 15.0, "float"),
    "rebalance_every": (5, 21, "int"),
}


def rebalance_every(params: dict[str, Any]) -> int:
    return max(1, int(params["rebalance_every"]))


def warmup(params: dict[str, Any]) -> int:
    return int(params["lookback"])


def target_positions(history: Any, params: dict[str, Any]) -> dict[str, float]:
    lookback, top_n = int(params["lookback"]), int(params["top_n"])
    within = float(params["within_pct"]) / 100.0
    if len(history) < lookback:
        return {}
    closeness = []
    for symbol in SYMBOLS:
        c = history.close(symbol, bars=lookback)
        if np.isnan(c).any():
            continue
        high = float(np.max(c))
        if high <= 0:
            continue
        closeness.append((c[-1] / high, symbol))
    picks = [s for r, s in sorted(closeness, reverse=True)[:top_n] if r >= 1.0 - within]
    return {s: 1.0 / top_n for s in picks}
