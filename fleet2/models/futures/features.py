"""Building blocks the futures models share, all computed for every bar at once with
numpy, and all using only the bar itself and earlier bars (the cut-off test in
tests/test_futures_cutoff.py checks every model built from them).

Days matter here: "so far today" values restart at each day's first bar, and anything
about earlier days (the average daily range, yesterday's close) only uses days that
have already finished.
"""
from __future__ import annotations

import numpy as np

from fleet2.sim.futures_data import Bars

_BIG = 1e7  # larger than any price, smaller than float rounding trouble over ~10,000 days


def first_index(bars: Bars) -> np.ndarray:
    """For every bar, the index of its day's first bar."""
    return np.maximum.accumulate(np.where(bars.first, np.arange(bars.n), 0))


def day_number(bars: Bars) -> np.ndarray:
    """0, 1, 2... for the days in `bars`, per bar."""
    return np.cumsum(bars.first) - 1


def sum_today(x: np.ndarray, bars: Bars) -> np.ndarray:
    """Running total of x since the day's first bar, this bar included."""
    cs = np.cumsum(x)
    start = first_index(bars)
    return cs - cs[start] + x[start]


def high_today(x: np.ndarray, bars: Bars) -> np.ndarray:
    """Running maximum of x since the day's first bar, this bar included."""
    shift = day_number(bars) * _BIG
    return np.maximum.accumulate(x + shift) - shift


def low_today(x: np.ndarray, bars: Bars) -> np.ndarray:
    return -high_today(-x, bars)


def high_today_where(x: np.ndarray, mask: np.ndarray, bars: Bars) -> np.ndarray:
    """Running maximum of x over the day's bars where mask is true (NaN before the first)."""
    out = high_today(np.where(mask, x, -_BIG / 2), bars)
    return np.where(out > -_BIG / 4, out, np.nan)


def low_today_where(x: np.ndarray, mask: np.ndarray, bars: Bars) -> np.ndarray:
    return -high_today_where(-x, mask, bars)


def latest_today(values: np.ndarray, mask: np.ndarray, bars: Bars, before: float = np.nan) -> np.ndarray:
    """Per bar, values[j] at the latest bar j <= this one of the same day where mask is
    true; `before` until there is one."""
    idx = np.maximum.accumulate(np.where(mask, np.arange(bars.n), -1))
    seen = idx >= first_index(bars)
    return np.where(seen, values[np.maximum(idx, 0)], before)


def hold(enter: np.ndarray, leave: np.ndarray, side: float, bars: Bars) -> np.ndarray:
    """A position on one side: `side` from a bar where enter is true until (not
    including) a later bar of the same day where leave is true; enter wins a tie."""
    event = np.where(enter, side, np.where(leave, 0.0, np.nan))
    return latest_today(np.nan_to_num(event), ~np.isnan(event), bars, before=0.0)


def first_today(enter: np.ndarray, bars: Bars) -> np.ndarray:
    """True only at the first bar of each day where enter is true."""
    return enter & (sum_today(enter.astype(np.int64), bars) == 1)


def vwap(series, bars: Bars) -> np.ndarray:
    """The day's volume-weighted average price so far (typical price (H+L+C)/3; plain
    average of it while the day has no volume yet)."""
    typical = (series.high + series.low + series.close) / 3.0
    vol = sum_today(series.volume, bars)
    weighted = sum_today(typical * series.volume, bars)
    plain = sum_today(typical, bars) / (np.arange(bars.n) - first_index(bars) + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(vol > 0, weighted / np.where(vol > 0, vol, 1.0), plain)


def average_range(series, bars: Bars, days: int) -> np.ndarray:
    """Per bar, the average high-to-low range of the `days` days before its own (NaN
    until there are that many)."""
    starts = np.flatnonzero(bars.first)
    rng = np.maximum.reduceat(series.high, starts) - np.minimum.reduceat(series.low, starts)
    cs = np.r_[0.0, np.cumsum(rng)]
    j = np.arange(starts.shape[0])
    avg = np.where(j >= days, (cs[j] - cs[np.maximum(j - days, 0)]) / max(days, 1), np.nan)
    return avg[day_number(bars)]


def previous_close(series, bars: Bars) -> np.ndarray:
    """Per bar, the last close of the day before, NaN on the first day and on a day
    whose contract differs from the day before's (a roll: the two prices are of
    different contracts and cannot be compared)."""
    starts = np.flatnonzero(bars.first)
    lasts = np.r_[starts[1:] - 1, bars.n - 1]
    prev_close = np.r_[np.nan, series.close[lasts[:-1]]]
    prev_iid = np.r_[-1, series.instrument[lasts[:-1]]]
    same = prev_iid == series.instrument[starts]
    per_day = np.where(same, prev_close, np.nan)
    return per_day[day_number(bars)]


def previous_high_low(series, bars: Bars) -> tuple[np.ndarray, np.ndarray]:
    """Per bar, the high and the low of the day before (NaN as for previous_close)."""
    starts = np.flatnonzero(bars.first)
    lasts = np.r_[starts[1:] - 1, bars.n - 1]
    highs = np.maximum.reduceat(series.high, starts)
    lows = np.minimum.reduceat(series.low, starts)
    prev_iid = np.r_[-1, series.instrument[lasts[:-1]]]
    same = prev_iid == series.instrument[starts]
    day = day_number(bars)
    return (np.where(same, np.r_[np.nan, highs[:-1]], np.nan)[day],
            np.where(same, np.r_[np.nan, lows[:-1]], np.nan)[day])


def day_open(series, bars: Bars) -> np.ndarray:
    """The open of the day's first bar."""
    return series.open[first_index(bars)]


def minutes_left(bars: Bars) -> np.ndarray:
    """Minutes from the end of each bar to the close of its session."""
    return bars.session - (bars.minute + bars.size)


def latest_signal(up: np.ndarray, down: np.ndarray, bars: Bars, params: dict) -> np.ndarray:
    """+1 from an "up" bar, -1 from a "down" bar, until a signal the other way or the
    close; only signals from start_minute to before last_entry_minute count."""
    window = (bars.minute >= int(params["start_minute"])) & (bars.minute < int(params["last_entry_minute"]))
    up, down = up & window, down & window
    return latest_today(np.where(up, 1.0, -1.0), up | down, bars, before=0.0)
