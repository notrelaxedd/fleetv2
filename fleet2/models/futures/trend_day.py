"""Trend day: the direction of the first hour often returns in the last hour."""
from __future__ import annotations

from typing import Any

import numpy as np

from fleet2.models.futures import features as f

NAME = "Trend day"
MARKET = "futures"
DESCRIPTION = "Trades the last part of the day in the direction the market moved in the first hour."
HOW_IT_WORKS = (
    "It measures how far the price moved from the open over the first hour or so. "
    "If that move was big enough, it buys in the afternoon after an up morning, or sells short after a down morning. "
    "It holds until its stop, its target or the close, and does nothing on days that started quietly."
)
SYMBOLS = ("MES", "MNQ")
DEFAULT_PARAMS: dict[str, Any] = {"symbol": "MES", "bar_minutes": 15, "first_minutes": 60, "min_move": 0.2,
                                  "lookback_days": 10, "enter_before_close": 90, "stop_ticks": 40, "target_ticks": 0}
SEARCH_SPACE = {
    "symbol": (("MES", "MNQ"), "choice"),
    "bar_minutes": ((1, 3, 5, 15), "choice"),
    "first_minutes": (30, 120, "int"),        # how long "the morning" is
    "min_move": (0.0, 0.5, "float"),          # the morning's move needed, as a share of a normal day's range
    "lookback_days": (5, 30, "int"),
    "enter_before_close": (30, 150, "int"),   # minutes before the close to get in
    "stop_ticks": (0, 80, "int"),
    "target_ticks": (0, 160, "int"),
}


def targets(bars: Any, params: dict[str, Any]) -> dict[str, np.ndarray]:
    symbol = str(params["symbol"])
    s = bars.series(symbol)
    morning = bars.minute + bars.size <= int(params["first_minutes"])
    move = f.latest_today(s.close, morning, bars) - f.day_open(s, bars)
    normal = f.average_range(s, bars, int(params["lookback_days"]))
    late = f.minutes_left(bars) <= int(params["enter_before_close"])
    late &= bars.minute >= int(params["first_minutes"])
    with np.errstate(invalid="ignore"):
        big = np.abs(move) > float(params["min_move"]) * normal
    side = np.where(late & big, np.sign(np.nan_to_num(move)), 0.0)
    return {symbol: side}
