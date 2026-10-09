"""Smart money: trade order block retests, changes of character or sweeps of yesterday's high or low."""
from __future__ import annotations

from typing import Any

import numpy as np

from fleet2.models.futures import features as f
from fleet2.models.futures.recipe import liquidity_sweeps, order_block_retests, structure_breaks

NAME = "Smart money"
MARKET = "futures"
DESCRIPTION = "Trades a pullback to an order block, a change of character, or a sweep of yesterday's high or low."
HOW_IT_WORKS = (
    "A swing high is a bar higher than the bars on either side of it, and it only counts once those later bars "
    "have closed; a close above the latest swing high is an upward break, and a break against the day's last "
    "break is a change of character. "
    "With the order block setup it waits for an upward break, then buys when the price dips back into the last "
    "falling bar before the break and holds; with the change of character setup it buys the turn itself; with "
    "the sweep setup it buys when the price dips below yesterday's low and closes back above it. "
    "Each works the other way round with a short sale, and a trade lasts until its stop, its target, a signal "
    "the other way or the close."
)
SYMBOLS = ("MES", "MNQ")
SETUPS = ("order_block", "choch", "sweep")
DEFAULT_PARAMS: dict[str, Any] = {"symbol": "MES", "bar_minutes": 5, "setup": "order_block", "swing_bars": 3,
                                  "ob_bars": 24, "start_minute": 15, "last_entry_minute": 300,
                                  "stop_ticks": 24, "target_ticks": 48}
SEARCH_SPACE = {
    "symbol": (("MES", "MNQ"), "choice"),
    "bar_minutes": ((1, 3, 5, 15), "choice"),
    "setup": (SETUPS, "choice"),
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
    setup = str(params["setup"])
    if setup == "order_block":
        up, down = order_block_retests(s, bars, int(params["swing_bars"]), int(params["ob_bars"]))
    elif setup == "choch":
        up, down, turned = structure_breaks(s, bars, int(params["swing_bars"]))
        up, down = up & turned, down & turned
    elif setup == "sweep":
        up, down = liquidity_sweeps(s, bars)
    else:
        raise ValueError(f"setup must be one of {', '.join(SETUPS)}, not {setup!r}")
    window = (bars.minute >= int(params["start_minute"])) & (bars.minute < int(params["last_entry_minute"]))
    up, down = up & window, down & window
    # The latest signal of the day decides the side (a signal the other way turns it round).
    return {symbol: f.latest_today(np.where(up, 1.0, -1.0), up | down, bars, before=0.0)}
