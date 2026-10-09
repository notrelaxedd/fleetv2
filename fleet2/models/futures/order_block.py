"""Order block: buy the first pullback into the last falling bar before an upward break of structure."""
from __future__ import annotations

from typing import Any

import numpy as np

from fleet2.models.futures import features as f
from fleet2.models.futures.recipe import order_block_retests

NAME = "Order block"
MARKET = "futures"
DESCRIPTION = "Buys a pullback into the last falling bar before an upward break, and sells short the mirror image."
HOW_IT_WORKS = (
    "A swing high is a bar higher than the bars on either side of it, and it only counts once those later "
    "bars have closed; a close above the latest swing high is an upward break. "
    "The order block is the last falling bar before that break: when the price later dips back into it and "
    "holds above its low, it buys, betting the break goes on; a downward break works the other way round. "
    "It holds until its stop, its target, a retest the other way or the close."
)
SYMBOLS = ("MES", "MNQ")
DEFAULT_PARAMS: dict[str, Any] = {"symbol": "MES", "bar_minutes": 5, "swing_bars": 3, "ob_bars": 24, "start_minute": 15,
                                  "last_entry_minute": 300, "stop_ticks": 24, "target_ticks": 48}
SEARCH_SPACE = {
    "symbol": (("MES", "MNQ"), "choice"),
    "bar_minutes": ((1, 3, 5, 15), "choice"),
    "swing_bars": (2, 10, "int"),            # bars on each side that make a swing
    "ob_bars": (3, 60, "int"),               # how many bars after a break a retest still counts
    "start_minute": (0, 120, "int"),         # no new trades before this many minutes after the open
    "last_entry_minute": (60, 380, "int"),   # nor after this many
    "stop_ticks": (0, 80, "int"),
    "target_ticks": (0, 160, "int"),
}


def targets(bars: Any, params: dict[str, Any]) -> dict[str, np.ndarray]:
    symbol = str(params["symbol"])
    s = bars.series(symbol)
    up, down = order_block_retests(s, bars, int(params["swing_bars"]), int(params["ob_bars"]))
    return {symbol: f.latest_signal(up, down, bars, params)}
