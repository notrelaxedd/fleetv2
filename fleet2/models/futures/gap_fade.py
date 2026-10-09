"""Gap fade: bet that part of a large overnight gap closes in the first hours."""
from __future__ import annotations

from typing import Any

import numpy as np

from fleet2.models.futures import features as f

NAME = "Gap fade"
MARKET = "futures"
DESCRIPTION = "When the day opens far from yesterday's close, bets that the price moves part of the way back."
HOW_IT_WORKS = (
    "It compares the first price of the day with the last price of the day before. "
    "After a large jump up it sells short at the open, and after a large drop it buys, expecting part of the gap to close. "
    "It gets out once enough of the gap has closed or at a set time of the morning, and skips days the contract changed."
)
SYMBOLS = ("MES", "MNQ")
DEFAULT_PARAMS: dict[str, Any] = {"symbol": "MES", "bar_minutes": 5, "min_gap": 0.25, "max_gap": 1.5, "fill": 0.5,
                                  "exit_minute": 120, "lookback_days": 10, "stop_ticks": 40, "target_ticks": 0}
SEARCH_SPACE = {
    "symbol": (("MES", "MNQ"), "choice"),
    "bar_minutes": ((1, 3, 5, 15), "choice"),
    "min_gap": (0.1, 1.0, "float"),     # smallest gap traded, as a share of a normal day's range
    "max_gap": (1.0, 3.0, "float"),     # largest (huge gaps are news, and tend to keep going)
    "fill": (0.3, 1.0, "float"),        # share of the gap that must close to take the profit
    "exit_minute": (30, 240, "int"),    # minutes after the open to give up
    "lookback_days": (5, 30, "int"),
    "stop_ticks": (0, 80, "int"),
    "target_ticks": (0, 160, "int"),
}


def targets(bars: Any, params: dict[str, Any]) -> dict[str, np.ndarray]:
    symbol = str(params["symbol"])
    s = bars.series(symbol)
    before = f.previous_close(s, bars)  # NaN on the first day and on roll days
    gap = f.day_open(s, bars) - before
    normal = f.average_range(s, bars, int(params["lookback_days"]))
    with np.errstate(invalid="ignore"):
        size = np.abs(gap) / normal
        trade = (size >= float(params["min_gap"])) & (size <= float(params["max_gap"]))
        level = before + (1.0 - float(params["fill"])) * gap  # where enough of the gap has closed
        closed = np.where(gap > 0, s.close <= level, s.close >= level)
    done = closed | (bars.minute + bars.size >= int(params["exit_minute"]))
    side = -np.sign(np.nan_to_num(gap))
    held = f.hold(bars.first & trade & ~done, done, 1.0, bars)  # in at the first bar, unless it is closed already
    return {symbol: np.where(trade, side * held, 0.0)}
