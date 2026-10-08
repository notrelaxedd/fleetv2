"""Pullback: in a strong intraday trend, buy short dips (or sell short rallies)."""
from __future__ import annotations

from typing import Any

import numpy as np

from fleet2.models.futures import features as f

NAME = "Pullback"
MARKET = "futures"
DESCRIPTION = "On a day with a strong trend, buys the short dips of an up day and sells short the bounces of a down day."
HOW_IT_WORKS = (
    "It calls the day a trend when the price is well away from the open and on the same side of the day's average price. "
    "In an up trend it buys after the price slips back from the day's high, and in a down trend it sells short after a bounce off the low. "
    "It takes the profit when the price makes a new high (or low) for the day, and gets out if the trend breaks."
)
SYMBOLS = ("MES", "MNQ")
DEFAULT_PARAMS: dict[str, Any] = {"symbol": "MES", "bar_minutes": 5, "trend": 0.4, "dip": 0.15, "lookback_days": 10,
                                  "start_minute": 60, "stop_ticks": 32, "target_ticks": 0}
SEARCH_SPACE = {
    "symbol": (("MES", "MNQ"), "choice"),
    "bar_minutes": ((1, 3, 5, 15), "choice"),
    "trend": (0.2, 1.0, "float"),       # move from the open that makes a trend, as a share of a normal day's range
    "dip": (0.05, 0.4, "float"),        # how far back from the day's high (or low) counts as a dip
    "lookback_days": (5, 30, "int"),
    "start_minute": (30, 180, "int"),   # no new trades in the first minutes of the day
    "stop_ticks": (0, 80, "int"),
    "target_ticks": (0, 160, "int"),
}


def targets(bars: Any, params: dict[str, Any]) -> dict[str, np.ndarray]:
    symbol = str(params["symbol"])
    s = bars.series(symbol)
    normal = f.average_range(s, bars, int(params["lookback_days"]))
    average = f.vwap(s, bars)
    move = s.close - f.day_open(s, bars)
    top, bottom = f.high_today(s.high, bars), f.low_today(s.low, bars)
    prev_top = np.r_[np.nan, top[:-1]]
    prev_bottom = np.r_[np.nan, bottom[:-1]]
    early = bars.minute < int(params["start_minute"])
    with np.errstate(invalid="ignore"):
        trend = float(params["trend"]) * normal
        dip = float(params["dip"]) * normal
        up = (move > trend) & (s.close > average) & ~early
        down = (move < -trend) & (s.close < average) & ~early
        buy = up & (top - s.close > dip)
        sell = down & (s.close - bottom > dip)
        new_high = ~bars.first & (s.high >= prev_top)
        new_low = ~bars.first & (s.low <= prev_bottom)
        long = f.hold(buy, new_high | (s.close < average), 1.0, bars)
        short = f.hold(sell, new_low | (s.close > average), -1.0, bars)
    return {symbol: long + short}
