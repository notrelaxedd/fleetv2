"""Opening range: trade a break above or below the first part of the day's range."""
from __future__ import annotations

from typing import Any

import numpy as np

from fleet2.models.futures import features as f
from fleet2.universe import CONTRACTS

NAME = "Opening range"
MARKET = "futures"
DESCRIPTION = "Waits for the price to break out of the range of the first part of the day, then trades in that direction."
HOW_IT_WORKS = (
    "It notes the highest and lowest price of the first 15 to 60 minutes after the open. "
    "If the price later closes clearly above that range it buys, and if it closes clearly below it sells short. "
    "It takes only the first breakout of the day and holds it until its stop, its target or the close."
)
SYMBOLS = ("MES", "MNQ")
DEFAULT_PARAMS: dict[str, Any] = {"symbol": "MES", "bar_minutes": 5, "range_minutes": 30, "buffer_ticks": 2,
                                  "last_entry_minute": 180, "stop_ticks": 24, "target_ticks": 48}
SEARCH_SPACE = {
    "symbol": (("MES", "MNQ"), "choice"),
    "bar_minutes": ((1, 3, 5, 15), "choice"),
    "range_minutes": (15, 60, "int"),
    "buffer_ticks": (0, 8, "int"),
    "last_entry_minute": (60, 300, "int"),
    "stop_ticks": (0, 80, "int"),
    "target_ticks": (0, 160, "int"),
}


def targets(bars: Any, params: dict[str, Any]) -> dict[str, np.ndarray]:
    symbol = str(params["symbol"])
    s = bars.series(symbol)
    size, span = bars.size, int(params["range_minutes"])
    buffer = int(params["buffer_ticks"]) * CONTRACTS[symbol]["tick"]
    in_range = bars.minute + size <= span  # bars wholly inside the opening range
    top = f.high_today_where(s.high, in_range, bars)
    bottom = f.low_today_where(s.low, in_range, bars)
    after = (bars.minute >= span) & (bars.minute < int(params["last_entry_minute"]))
    with np.errstate(invalid="ignore"):
        up = after & (s.close > top + buffer)
        down = after & (s.close < bottom - buffer)
    first = f.first_today(up | down, bars)
    side = np.where(up, 1.0, -1.0)
    return {symbol: f.latest_today(side, first, bars, before=0.0)}
