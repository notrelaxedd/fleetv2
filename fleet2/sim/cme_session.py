"""CME trading hours for the micro index futures (MES, MNQ), in America/Chicago time.

The fleet only uses the regular session, the hours the New York stock market is open:
8:30 to 15:00 Chicago time on a normal day, 8:30 to 12:00 on a half day. CME's equity
index futures follow the stock market's holidays and early closes for that session, so
the calendar below is the New York Stock Exchange's:

- Closed: New Year's Day, Martin Luther King Day, Presidents Day, Good Friday, Memorial
  Day, Juneteenth (from 2022), Independence Day, Labor Day, Thanksgiving, Christmas,
  plus one-off closures (national days of mourning). A holiday on a Saturday is taken on
  the Friday before, one on a Sunday on the Monday after (New Year's Day on a Saturday
  is not moved).
- Half days (close at 12:00 Chicago time): July 3 when it is a Monday to Thursday, the
  day after Thanksgiving, and December 24 when it is a Monday to Thursday.

Every bar belongs to a trading day. CME's trading day starts at 17:00 Chicago time the
evening before, so a bar at or after 17:00 belongs to the next trading day. Regular
session bars always belong to their own calendar date.

Used by the coordinator (which bars to keep) and by the workers (the minute grid the
backtester trades on), so both always agree on the hours. Standard library and numpy only.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np

CHICAGO = ZoneInfo("America/Chicago")
OPEN = time(8, 30)
CLOSE = time(15, 0)
HALF_DAY_CLOSE = time(12, 0)
TRADING_DAY_ROLLS_AT = time(17, 0)  # a bar from 17:00 on belongs to the next trading day

# Closures that no rule predicts (national days of mourning).
ONE_OFF_CLOSURES = frozenset({date(2018, 12, 5), date(2025, 1, 9)})


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th given weekday (0 = Monday) of a month; n = -1 for the last one."""
    if n > 0:
        first = date(year, month, 1)
        return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Easter Sunday (the anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _observed(d: date) -> date:
    """Saturday holidays move to Friday, Sunday holidays to Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=None)
def holidays(year: int) -> frozenset[date]:
    """Weekdays of `year` with no regular session."""
    out = {
        _nth_weekday(year, 1, 0, 3),   # Martin Luther King Day
        _nth_weekday(year, 2, 0, 3),   # Presidents Day
        _easter(year) - timedelta(days=2),  # Good Friday
        _nth_weekday(year, 5, 0, -1),  # Memorial Day
        _observed(date(year, 7, 4)),   # Independence Day
        _nth_weekday(year, 9, 0, 1),   # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),  # Christmas
    }
    new_year = date(year, 1, 1)
    if new_year.weekday() < 5:
        out.add(new_year)
    elif new_year.weekday() == 6:
        out.add(new_year + timedelta(days=1))
    if year >= 2022:
        out.add(_observed(date(year, 6, 19)))  # Juneteenth
    out |= {d for d in ONE_OFF_CLOSURES if d.year == year}
    return frozenset(d for d in out if d.year == year)


@lru_cache(maxsize=None)
def half_days(year: int) -> frozenset[date]:
    """Days of `year` whose regular session ends at 12:00 Chicago time."""
    out = {_nth_weekday(year, 11, 3, 4) + timedelta(days=1)}  # the day after Thanksgiving
    for d in (date(year, 7, 3), date(year, 12, 24)):
        if d.weekday() < 4:  # Monday to Thursday
            out.add(d)
    return frozenset(d for d in out if d not in holidays(year))


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in holidays(d.year)


def is_half_day(d: date) -> bool:
    return d in half_days(d.year)


def session(d: date) -> tuple[int, int] | None:
    """(open, close) of the regular session on `d` as epoch seconds, None when closed."""
    if not is_trading_day(d):
        return None
    close = HALF_DAY_CLOSE if is_half_day(d) else CLOSE
    open_dt = datetime.combine(d, OPEN, CHICAGO)
    close_dt = datetime.combine(d, close, CHICAGO)
    return int(open_dt.timestamp()), int(close_dt.timestamp())


def session_minutes(d: date) -> int:
    """Length of the regular session in minutes (390, 210 on a half day, 0 when closed)."""
    s = session(d)
    return 0 if s is None else (s[1] - s[0]) // 60


def trading_days(first: date, last: date) -> list[date]:
    """Every trading day from `first` to `last`, both included."""
    out, d = [], first
    while d <= last:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def trading_day(epoch: int) -> date:
    """The trading day a bar starting at `epoch` belongs to."""
    local = datetime.fromtimestamp(int(epoch), timezone.utc).astimezone(CHICAGO)
    return local.date() + timedelta(days=1) if local.time() >= TRADING_DAY_ROLLS_AT else local.date()


def chicago_dates(epochs: np.ndarray) -> np.ndarray:
    """Calendar date in Chicago of every epoch, as int YYYYMMDD (vectorised: one zone
    lookup per distinct UTC day instead of one per bar)."""
    epochs = np.asarray(epochs, dtype=np.int64)
    if epochs.size == 0:
        return np.zeros(0, dtype=np.int64)
    local = epochs + utc_offsets(epochs)
    days = local // 86400
    uniq, inverse = np.unique(days, return_inverse=True)
    as_int = np.array([int((date(1970, 1, 1) + timedelta(days=int(u))).strftime("%Y%m%d")) for u in uniq],
                      dtype=np.int64)
    return as_int[inverse]


def utc_offsets(epochs: np.ndarray) -> np.ndarray:
    """Chicago's offset from UTC in seconds (-21600 or -18000) for every epoch."""
    epochs = np.asarray(epochs, dtype=np.int64)
    if epochs.size == 0:
        return np.zeros(0, dtype=np.int64)
    first = datetime.fromtimestamp(int(epochs.min()), timezone.utc).year - 1
    last = datetime.fromtimestamp(int(epochs.max()), timezone.utc).year + 1
    edges, offsets = _transitions(first, last)
    return offsets[np.searchsorted(edges, epochs, side="right")]


@lru_cache(maxsize=None)
def _transitions(first_year: int, last_year: int) -> tuple[np.ndarray, np.ndarray]:
    """Clock-change instants (epoch seconds) between the years, and the offset in force
    before the first one and after each one."""
    edges: list[int] = []
    start = datetime(first_year, 1, 1, tzinfo=timezone.utc)
    hour = timedelta(hours=1)
    prev = start.astimezone(CHICAGO).utcoffset()
    offsets = [int(prev.total_seconds())]
    t = start
    end = datetime(last_year + 1, 1, 1, tzinfo=timezone.utc)
    while t < end:  # step a day at a time, then find the exact hour of each change
        nxt = t + timedelta(days=1)
        off = nxt.astimezone(CHICAGO).utcoffset()
        if off != prev:
            h = t
            while (h + hour).astimezone(CHICAGO).utcoffset() == prev:
                h += hour
            edges.append(int((h + hour).timestamp()))
            offsets.append(int(off.total_seconds()))
            prev = off
        t = nxt
    return np.asarray(edges, dtype=np.int64), np.asarray(offsets, dtype=np.int64)


def as_date(yyyymmdd: int) -> date:
    v = int(yyyymmdd)
    return date(v // 10000, v // 100 % 100, v % 100)


def as_int(d: date) -> int:
    return d.year * 10000 + d.month * 100 + d.day


def in_session(epochs: np.ndarray) -> np.ndarray:
    """True for every bar start inside its day's regular session."""
    epochs = np.asarray(epochs, dtype=np.int64)
    out = np.zeros(epochs.shape, dtype=bool)
    if epochs.size == 0:
        return out
    days = chicago_dates(epochs)
    for d in np.unique(days):
        s = session(as_date(int(d)))
        if s is None:
            continue
        mask = days == d
        out[mask] = (epochs[mask] >= s[0]) & (epochs[mask] < s[1])
    return out
