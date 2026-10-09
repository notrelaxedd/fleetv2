"""Liquidity sweep: trade a poke through yesterday's high or low that closes back inside."""
from __future__ import annotations

from typing import Any

import numpy as np

from fleet2.models.futures import features as f
from fleet2.models.futures.recipe import liquidity_sweeps

NAME = "Liquidity sweep"
MARKET = "futures"
DESCRIPTION = "Buys when the price dips below yesterday's low and closes back above it, and sells short the mirror image."
HOW_IT_WORKS = (
    "Many stop orders sit just past yesterday's high and low, and a quick poke through one can be the last "
    "push before the price turns. "
    "When a bar trades below yesterday's low and closes back above it, it buys; when a bar trades above "
    "yesterday's high and closes back below it, it sells short. "
    "It holds until its stop, its target, a sweep the other way or the close."
)
SYMBOLS = ("MES", "MNQ")
DEFAULT_PARAMS: dict[str, Any] = {"symbol": "MES", "bar_minutes": 5, "start_minute": 15,
                                  "last_entry_minute": 300, "stop_ticks": 24, "target_ticks": 48}
SEARCH_SPACE = {
    "symbol": (("MES", "MNQ"), "choice"),
    "bar_minutes": ((1, 3, 5, 15), "choice"),
    "start_minute": (0, 120, "int"),         # no new trades before this many minutes after the open
    "last_entry_minute": (60, 380, "int"),   # nor after this many
    "stop_ticks": (0, 80, "int"),
    "target_ticks": (0, 160, "int"),
}


def targets(bars: Any, params: dict[str, Any]) -> dict[str, np.ndarray]:
    symbol = str(params["symbol"])
    s = bars.series(symbol)
    up, down = liquidity_sweeps(s, bars)
    return {symbol: f.latest_signal(up, down, bars, params)}
