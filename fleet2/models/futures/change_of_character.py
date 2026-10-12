"""Change of character: trade a break of structure that goes against the day's last break."""
from __future__ import annotations

from typing import Any

import numpy as np

from fleet2.models.futures import features as f
from fleet2.models.futures.recipe import structure_breaks

NAME = "Change of character"
MARKET = "futures"
DESCRIPTION = "Buys when the price breaks a swing high after the day's last break was down, and the mirror image."
HOW_IT_WORKS = (
    "A swing high is a bar higher than the bars on either side of it, and it only counts once those later "
    "bars have closed; a close above the latest swing high is an upward break, below the latest swing low a "
    "downward one. "
    "When a break goes against the day's last break (a change of character), it trades in the new direction, "
    "betting the day has turned. "
    "It holds until its stop, its target, a change of character the other way or the close."
)
SYMBOLS = ("MES", "MNQ")
DEFAULT_PARAMS: dict[str, Any] = {"symbol": "MES", "bar_minutes": 5, "swing_bars": 3, "start_minute": 15,
                                  "last_entry_minute": 300, "stop_ticks": 24, "target_ticks": 48}
SEARCH_SPACE = {
    "symbol": (("MES", "MNQ"), "choice"),
    "bar_minutes": ((1, 3, 5, 15), "choice"),
    "swing_bars": (2, 10, "int"),            # bars on each side that make a swing
    "start_minute": (0, 120, "int"),         # no new trades before this many minutes after the open
    "last_entry_minute": (60, 380, "int"),   # nor after this many
    "stop_ticks": (0, 80, "int"),
    "target_ticks": (0, 160, "int"),
}


def targets(bars: Any, params: dict[str, Any]) -> dict[str, np.ndarray]:
    symbol = str(params["symbol"])
    s = bars.series(symbol)
    up, down, turned = structure_breaks(s, bars, int(params["swing_bars"]))
    up, down = up & turned, down & turned
    return {symbol: f.latest_signal(up, down, bars, params)}
