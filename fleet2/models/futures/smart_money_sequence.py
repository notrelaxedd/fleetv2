"""Smart money sequence: a sweep of yesterday's low, a change of character, then the order block pullback."""
from __future__ import annotations

from typing import Any

import numpy as np

from fleet2.models.futures import features as f
from fleet2.models.futures.recipe import smart_money_sequence

NAME = "Smart money sequence"
MARKET = "futures"
DESCRIPTION = "Waits for a sweep of yesterday's low, a turn upward and a pullback into the order block, then buys."
HOW_IT_WORKS = (
    "First the price must poke below yesterday's low and close back above it (a sweep); soon after, it must "
    "close above the latest swing high when the day's last break was downward (a change of character). "
    "It then buys when the price dips back into the last falling bar before that break and holds, and a sweep "
    "of yesterday's high followed by a turn downward works the other way round, with a short sale. "
    "It takes few trades and holds each until its stop, its target, a sequence the other way or the close."
)
SYMBOLS = ("MES", "MNQ")
DEFAULT_PARAMS: dict[str, Any] = {"symbol": "MES", "bar_minutes": 5, "swing_bars": 3, "ob_bars": 24, "sweep_bars": 36, "start_minute": 15,
                                  "last_entry_minute": 300, "stop_ticks": 24, "target_ticks": 48}
SEARCH_SPACE = {
    "symbol": (("MES", "MNQ"), "choice"),
    "bar_minutes": ((1, 3, 5, 15), "choice"),
    "swing_bars": (2, 10, "int"),            # bars on each side that make a swing
    "ob_bars": (3, 60, "int"),               # how many bars after the turn a pullback still counts
    "sweep_bars": (3, 120, "int"),           # how many bars after the sweep the turn may come
    "start_minute": (0, 120, "int"),         # no new trades before this many minutes after the open
    "last_entry_minute": (60, 380, "int"),   # nor after this many
    "stop_ticks": (0, 80, "int"),
    "target_ticks": (0, 160, "int"),
}


def targets(bars: Any, params: dict[str, Any]) -> dict[str, np.ndarray]:
    symbol = str(params["symbol"])
    s = bars.series(symbol)
    up, down = smart_money_sequence(s, bars, int(params["swing_bars"]), int(params["ob_bars"]), int(params["sweep_bars"]))
    return {symbol: f.latest_signal(up, down, bars, params)}
