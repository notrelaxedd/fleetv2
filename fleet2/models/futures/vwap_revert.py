"""VWAP revert: on a quiet day, bet that a price stretched far from the day's average comes back."""
from __future__ import annotations

from typing import Any

import numpy as np

from fleet2.models.futures import features as f

NAME = "VWAP revert"
MARKET = "futures"
DESCRIPTION = "On quiet days, bets that a price stretched far from the day's average price will drift back to it."
HOW_IT_WORKS = (
    "It keeps the day's volume-weighted average price, the average price paid so far today. "
    "When the market has been calm and the price runs well above that average it sells short, and well below it buys. "
    "It gets out once the price is back at the average, and stays away on busy days when moves tend to keep going."
)
SYMBOLS = ("MES", "MNQ")
DEFAULT_PARAMS: dict[str, Any] = {"symbol": "MES", "bar_minutes": 5, "band": 0.25, "quiet": 0.8, "lookback_days": 10,
                                  "start_minute": 45, "stop_ticks": 32, "target_ticks": 0}
SEARCH_SPACE = {
    "symbol": (("MES", "MNQ"), "choice"),
    "bar_minutes": ((1, 3, 5, 15), "choice"),
    "band": (0.1, 0.6, "float"),          # how far from the average, as a share of a normal day's range
    "quiet": (0.4, 1.2, "float"),         # today's range so far must be under this share of a normal day's
    "lookback_days": (5, 30, "int"),      # days that make "a normal day's range"
    "start_minute": (15, 120, "int"),     # no new trades in the first minutes of the day
    "stop_ticks": (0, 80, "int"),
    "target_ticks": (0, 160, "int"),
}


def targets(bars: Any, params: dict[str, Any]) -> dict[str, np.ndarray]:
    symbol = str(params["symbol"])
    s = bars.series(symbol)
    normal = f.average_range(s, bars, int(params["lookback_days"]))
    gap = s.close - f.vwap(s, bars)
    today = f.high_today(s.high, bars) - f.low_today(s.low, bars)
    with np.errstate(invalid="ignore"):
        ok = (today < float(params["quiet"]) * normal) & (bars.minute >= int(params["start_minute"]))
        stretch = float(params["band"]) * normal
        short = f.hold(ok & (gap > stretch), gap <= 0, -1.0, bars)
        long = f.hold(ok & (gap < -stretch), gap >= 0, 1.0, bars)
    return {symbol: short + long}
