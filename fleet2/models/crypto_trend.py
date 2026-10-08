"""Crypto trend following: hold coins in an uptrend, sit in cash otherwise."""
from __future__ import annotations

from typing import Any

import numpy as np

from fleet2.universe import CRYPTO

NAME = "Crypto trend"
MARKET = "crypto"
DESCRIPTION = "Holds the coins whose prices are trending up and moves to cash when the trend turns down."
HOW_IT_WORKS = (
    "For each coin it compares the average price of the last day or so with the average of the last week or two. "
    "When the short average is above the long one the price is climbing, so it holds that coin. "
    "When the short average drops below the long one, it sells and waits in cash."
)
SYMBOLS = CRYPTO
DEFAULT_PARAMS: dict[str, Any] = {"fast_hours": 24, "slow_hours": 168, "rebalance_every": 4, "max_coins": 8}
SEARCH_SPACE = {
    "fast_hours": (6, 72, "int"),
    "slow_hours": (96, 720, "int"),
    "rebalance_every": (1, 24, "int"),
    "max_coins": (3, 8, "int"),
}


def rebalance_every(params: dict[str, Any]) -> int:
    return max(1, int(params["rebalance_every"]))


def warmup(params: dict[str, Any]) -> int:
    return max(int(params["fast_hours"]), int(params["slow_hours"])) + 1


def target_positions(history: Any, params: dict[str, Any]) -> dict[str, float]:
    fast, slow, slots = int(params["fast_hours"]), int(params["slow_hours"]), int(params["max_coins"])
    if fast >= slow:
        fast, slow = slow, fast
    if len(history) < slow:
        return {}
    rising = []
    for symbol in SYMBOLS:
        c = history.close(symbol, bars=slow)
        if np.isnan(c).sum() > slow // 10:
            continue
        fast_avg, slow_avg = np.nanmean(c[-fast:]), np.nanmean(c)
        if fast_avg > slow_avg:
            rising.append((fast_avg / slow_avg, symbol))
    chosen = [s for _, s in sorted(rising, reverse=True)[:slots]]
    return {s: 1.0 / slots for s in chosen}
