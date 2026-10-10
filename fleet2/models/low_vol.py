"""Low volatility: hold the stocks whose prices have moved the least."""
from __future__ import annotations

from typing import Any

import numpy as np

from fleet2.universe import STOCKS

NAME = "Calm stocks"
MARKET = "stocks"
DESCRIPTION = "Holds the stocks whose prices have bounced around the least over the past few months."
HOW_IT_WORKS = (
    "Every few weeks it measures how much each stock's price swung up and down from day to day over the last few months. "
    "It buys the calmest ones and sells any that have become jumpier than the rest. "
    "Calm stocks have often earned about as much as jumpy ones with smaller falls along the way."
)
SYMBOLS = tuple(s for s in STOCKS if s not in ("SPY", "QQQ"))
DEFAULT_PARAMS: dict[str, Any] = {"lookback": 63, "top_n": 10, "rebalance_every": 21}
SEARCH_SPACE = {
    "lookback": (21, 252, "int"),
    "top_n": (3, 10, "int"),
    "rebalance_every": (5, 21, "int"),
}


def rebalance_every(params: dict[str, Any]) -> int:
    return max(1, int(params["rebalance_every"]))


def warmup(params: dict[str, Any]) -> int:
    return int(params["lookback"]) + 1


def target_positions(history: Any, params: dict[str, Any]) -> dict[str, float]:
    lookback, top_n = int(params["lookback"]), int(params["top_n"])
    need = lookback + 1
    if len(history) < need:
        return {}
    swings = []
    for symbol in SYMBOLS:
        c = history.close(symbol, bars=need)
        if np.isnan(c).any() or (c <= 0).any():
            continue
        swings.append((float(np.std(np.diff(np.log(c)), ddof=1)), symbol))
    calmest = [s for _, s in sorted(swings)[:top_n]]
    return {s: 1.0 / top_n for s in calmest}
