"""Short-term reversal: buy the large stocks that fell the most last week."""
from __future__ import annotations

from typing import Any

import numpy as np

from fleet2.universe import STOCKS

NAME = "Weekly losers"
MARKET = "stocks"
DESCRIPTION = "Buys the big stocks that fell the most over the past week, expecting part of the drop to bounce back."
HOW_IT_WORKS = (
    "Every few days it checks how much each stock moved over the last week or so. "
    "It buys the ones that fell the most, but only ones that actually fell, and sells them at the next check unless they are still among the biggest losers. "
    "Large companies often bounce back after a short sharp drop that had no lasting reason behind it."
)
SYMBOLS = tuple(s for s in STOCKS if s not in ("SPY", "QQQ"))
DEFAULT_PARAMS: dict[str, Any] = {"lookback": 5, "top_n": 10, "min_drop_pct": 0.0, "rebalance_every": 5}
SEARCH_SPACE = {
    "lookback": (3, 21, "int"),
    "top_n": (3, 10, "int"),
    "min_drop_pct": (0.0, 5.0, "float"),
    "rebalance_every": (1, 10, "int"),
}


def rebalance_every(params: dict[str, Any]) -> int:
    return max(1, int(params["rebalance_every"]))


def warmup(params: dict[str, Any]) -> int:
    return int(params["lookback"]) + 1


def target_positions(history: Any, params: dict[str, Any]) -> dict[str, float]:
    lookback, top_n = int(params["lookback"]), int(params["top_n"])
    min_drop = float(params["min_drop_pct"]) / 100.0
    need = lookback + 1
    if len(history) < need:
        return {}
    moves = []
    for symbol in SYMBOLS:
        c = history.close(symbol, bars=need)
        start, end = c[0], c[-1]
        if np.isnan(start) or np.isnan(end) or start <= 0:
            continue
        moves.append((end / start - 1.0, symbol))
    losers = [s for m, s in sorted(moves)[:top_n] if m < -min_drop]
    return {s: 1.0 / top_n for s in losers}
