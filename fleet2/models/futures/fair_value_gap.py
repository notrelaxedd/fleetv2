"""Fair value gap: trade the first pullback into a gap left by a fast three-bar move."""
from __future__ import annotations

from typing import Any

import numpy as np

from fleet2.models.futures import features as f
from fleet2.models.futures.recipe import fair_value_gaps

NAME = "Fair value gap"
MARKET = "futures"
DESCRIPTION = "Buys a dip back into a gap left by a fast move up, and sells short a bounce into a gap left by a fast drop."
HOW_IT_WORKS = (
    "A fast move can leave a gap between three bars: the third bar's low is above the first bar's high, so no "
    "trading happened in between (a fair value gap). "
    "When the price later dips back into the day's latest such gap and holds above its bottom, it buys, betting "
    "the move resumes; a gap left by a fast drop works the other way round, with a short sale. "
    "It holds until its stop, its target, a retest of a gap the other way or the close, and ignores gaps that are "
    "tiny or that the price has already gone through."
)
SYMBOLS = ("MES", "MNQ")
DEFAULT_PARAMS: dict[str, Any] = {"symbol": "MES", "bar_minutes": 5, "fvg_size": 0.05, "fvg_bars": 24,
                                  "lookback_days": 10, "start_minute": 15, "last_entry_minute": 300,
                                  "stop_ticks": 24, "target_ticks": 48}
SEARCH_SPACE = {
    "symbol": (("MES", "MNQ"), "choice"),
    "bar_minutes": ((1, 3, 5, 15), "choice"),
    "fvg_size": (0.02, 0.3, "float"),        # smallest gap, as a share of a normal day's range
    "fvg_bars": (3, 60, "int"),              # how many bars after the gap a retest still counts
    "lookback_days": (5, 30, "int"),         # days that make "a normal day's range"
    "start_minute": (0, 120, "int"),         # no new trades before this many minutes after the open
    "last_entry_minute": (60, 380, "int"),   # nor after this many
    "stop_ticks": (0, 80, "int"),
    "target_ticks": (0, 160, "int"),
}


def targets(bars: Any, params: dict[str, Any]) -> dict[str, np.ndarray]:
    symbol = str(params["symbol"])
    s = bars.series(symbol)
    normal = f.average_range(s, bars, int(params["lookback_days"]))
    up, down = fair_value_gaps(s, bars, float(params["fvg_size"]) * normal, int(params["fvg_bars"]))
    window = (bars.minute >= int(params["start_minute"])) & (bars.minute < int(params["last_entry_minute"]))
    up, down = up & window, down & window
    # The latest retest of the day decides the side (a retest the other way turns it round).
    side = np.where(up, 1.0, -1.0)
    return {symbol: f.latest_today(side, up | down, bars, before=0.0)}
