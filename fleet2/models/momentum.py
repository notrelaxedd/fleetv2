"""Momentum: hold the stocks that have risen most over the past few months."""
from __future__ import annotations

from typing import Any

import numpy as np

from fleet2.universe import STOCKS

NAME = "Momentum"
MARKET = "stocks"
DESCRIPTION = "Buys the stocks that have risen the most over the past few months and holds them while they keep rising."
HOW_IT_WORKS = (
    "Every week it checks how much each stock has gained over the last few months. "
    "It buys the few biggest gainers, but only ones that are actually up, and sells any that drop out of the top. "
    "When nothing is rising it keeps the money in cash."
)
SYMBOLS = tuple(s for s in STOCKS if s != "SPY")
DEFAULT_PARAMS: dict[str, Any] = {"lookback": 126, "skip_recent": 5, "top_n": 10, "rebalance_every": 5}
SEARCH_SPACE = {
    "lookback": (40, 252, "int"),
    "skip_recent": (0, 21, "int"),
    "top_n": (5, 15, "int"),
    "rebalance_every": (5, 21, "int"),
}


def rebalance_every(params: dict[str, Any]) -> int:
    return max(1, int(params["rebalance_every"]))


def warmup(params: dict[str, Any]) -> int:
    return int(params["lookback"]) + int(params["skip_recent"]) + 1


def target_positions(history: Any, params: dict[str, Any]) -> dict[str, float]:
    lookback, skip, top_n = int(params["lookback"]), int(params["skip_recent"]), int(params["top_n"])
    need = lookback + skip + 1
    if len(history) < need:
        return {}
    gains = []
    for symbol in SYMBOLS:
        c = history.close(symbol, bars=need)
        start, end = c[0], c[-1 - skip]
        if np.isnan(start) or np.isnan(end) or start <= 0:
            continue
        gains.append((end / start - 1.0, symbol))
    winners = [s for g, s in sorted(gains, reverse=True)[:top_n] if g > 0]
    return {s: 1.0 / top_n for s in winners}
