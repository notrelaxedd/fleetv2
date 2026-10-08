"""Futures prices on a worker: the coordinator's .npz file turned into a regular grid of
regular-session minutes, cached in memory by ETag, and resampled into the 1-, 3-, 5- or
15-minute bars a model decides on.

The grid has one slot for every minute of every trading day in the data (390 on a
normal day, 210 on a half day), so a day is a fixed block and resampling is a reshape.
A minute with no trade (common on the SPY/QQQ stand-in) is filled with the last price
and no volume. Only earlier prices are ever used to fill a gap, so filling can never
leak a later price into an earlier minute. A day on which a symbol has no bars at all
is marked as not tradable for that symbol.

Timing rule (as for stocks and crypto): a decision bar covers its minutes and closes
at the end of its last minute. The decision made on it fills at the open of the next
1-minute bar, never at a price the model has already seen.
"""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fleet2.common import http
from fleet2.sim import cme_session
from fleet2.universe import CONTRACTS

BAR_SIZES = (1, 3, 5, 15)


@dataclass(frozen=True)
class FuturesData:
    """Regular-session minutes of one or more futures symbols.

    Per minute (shape (minutes,)): times (epoch seconds of the minute's start), day
    (index into days) and minute (minutes since that day's open). Per symbol and minute
    (shape (symbols, minutes)): open, high, low, close, volume, instrument (contract id)
    and real (False where the minute was filled). Per day: days (YYYYMMDD), day_start
    and day_end (first and one-past-last minute index), session (minutes in the session)
    and, per symbol, tradable (the symbol had bars that day).
    """

    feed: str
    symbols: tuple[str, ...]
    times: np.ndarray
    day: np.ndarray
    minute: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    instrument: np.ndarray
    real: np.ndarray
    days: np.ndarray
    day_start: np.ndarray
    day_end: np.ndarray
    session: np.ndarray
    tradable: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def n_minutes(self) -> int:
        return int(self.times.shape[0])

    @property
    def n_days(self) -> int:
        return int(self.days.shape[0])

    def row(self, symbol: str) -> int:
        return self.symbols.index(symbol)

    def until_minute(self, stop: int) -> "FuturesData":
        """Minutes [0, stop) only: prices cut off at minute `stop` (the cut-off tests).
        A day cut part way through keeps its first minutes."""
        stop = int(max(0, min(stop, self.n_minutes)))
        n_days = int(np.count_nonzero(self.day_start < stop))
        sl = slice(0, stop)
        return FuturesData(self.feed, self.symbols, self.times[sl], self.day[sl], self.minute[sl], self.open[:, sl],
                           self.high[:, sl], self.low[:, sl], self.close[:, sl], self.volume[:, sl],
                           self.instrument[:, sl], self.real[:, sl], self.days[:n_days], self.day_start[:n_days],
                           np.minimum(self.day_end[:n_days], stop), self.session[:n_days], self.tradable[:, :n_days],
                           self.meta)

    def until_day(self, n_days: int) -> "FuturesData":
        """The first `n_days` trading days only."""
        n_days = int(max(0, min(n_days, self.n_days)))
        return self.until_minute(int(self.day_end[n_days - 1]) if n_days else 0)

    def day_index(self, yyyymmdd: int, side: str = "left") -> int:
        """Index of the first day on or after (side="left") or after (side="right") a date."""
        return int(np.searchsorted(self.days, int(yyyymmdd), side=side))

    def point_value(self, symbol: str) -> float:
        return float(CONTRACTS[symbol]["point_value"])

    def tick(self, symbol: str) -> float:
        return float(CONTRACTS[symbol]["tick"])


def build(feed: str, series: dict[str, dict[str, np.ndarray]], meta: dict[str, Any] | None = None) -> FuturesData:
    """The minute grid from raw per-symbol bars {"t", "o", "h", "l", "c", "v", "iid"}."""
    symbols = tuple(s for s in series if np.asarray(series[s]["t"]).size)
    if not symbols:
        raise ValueError("no futures prices; run a Futures prices job first")
    all_t = np.unique(np.concatenate([np.asarray(series[s]["t"], dtype=np.int64) for s in symbols]))
    day_ints = np.unique(cme_session.chicago_dates(all_t))
    days, starts, lengths = [], [], []
    for d in day_ints:
        sess = cme_session.session(cme_session.as_date(int(d)))
        if sess is None:
            continue
        days.append(int(d))
        starts.append(sess[0])
        lengths.append((sess[1] - sess[0]) // 60)
    lengths_a = np.asarray(lengths, dtype=np.int64)
    day_end = np.cumsum(lengths_a)
    day_start = day_end - lengths_a
    n = int(day_end[-1]) if len(days) else 0
    times = np.empty(n, dtype=np.int64)
    minute = np.empty(n, dtype=np.int32)
    day = np.repeat(np.arange(len(days), dtype=np.int32), lengths_a)
    for k, (s0, ln) in enumerate(zip(starts, lengths)):
        times[day_start[k]:day_end[k]] = s0 + 60 * np.arange(ln, dtype=np.int64)
        minute[day_start[k]:day_end[k]] = np.arange(ln, dtype=np.int32)
    shape = (len(symbols), n)
    cols = {k: np.full(shape, np.nan) for k in ("o", "h", "l", "c", "v")}
    iid = np.zeros(shape, dtype=np.int64)
    real = np.zeros(shape, dtype=bool)
    tradable = np.zeros((len(symbols), len(days)), dtype=bool)
    for k, sym in enumerate(symbols):
        t = np.asarray(series[sym]["t"], dtype=np.int64)
        pos = np.searchsorted(times, t)
        ok = (pos < n) & (times[np.minimum(pos, n - 1)] == t) if n else np.zeros(t.shape, dtype=bool)
        pos = pos[ok]
        for key, arr in cols.items():
            arr[k, pos] = np.asarray(series[sym][key], dtype=float)[ok]
        real[k, pos] = True
        raw_iid = np.asarray(series[sym]["iid"], dtype=np.int64)[ok]
        bar_day = day[pos]
        tradable[k, np.unique(bar_day)] = True
        # One contract per day: the one of the day's first real bar.
        first_of_day = np.r_[True, bar_day[1:] != bar_day[:-1]] if bar_day.size else bar_day.astype(bool)
        per_day = np.zeros(len(days), dtype=np.int64)
        per_day[bar_day[first_of_day]] = raw_iid[first_of_day]
        iid[k] = per_day[day]
    _fill(cols, real)
    return FuturesData(feed, symbols, times, day, minute, cols["o"], cols["h"], cols["l"], cols["c"], cols["v"],
                       iid, real, np.asarray(days, dtype=np.int64), day_start, day_end, lengths_a, tradable,
                       meta or {})


def _fill(cols: dict[str, np.ndarray], real: np.ndarray) -> None:
    """Minutes with no bar take the last earlier close (open = high = low = close, no
    volume). Minutes before a symbol's first bar stay NaN."""
    close = cols["c"]
    n = close.shape[1]
    idx = np.where(real, np.arange(n)[None, :], -1)
    np.maximum.accumulate(idx, axis=1, out=idx)
    rows = np.arange(close.shape[0])[:, None]
    carried = np.where(idx >= 0, close[rows, np.maximum(idx, 0)], np.nan)
    missing = ~real
    for key in ("o", "h", "l", "c"):
        cols[key][missing] = carried[missing]
    cols["v"][missing] = 0.0


def from_npz(raw: bytes) -> FuturesData:
    """FuturesData from the coordinator's file (coordinator.futures_data.payload)."""
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        series = {s: {k: z[f"{s}_{k}"] for k in ("t", "o", "h", "l", "c", "v", "iid")} for s in meta["symbols"]}
    return build(str(meta.get("feed") or ""), series, meta)


# ------------------------------------------------------------------ download with an ETag cache


class PriceCache:
    """The futures prices a job has downloaded, one per period, kept in memory and
    fetched again only when the coordinator's ETag changes (a 304 answer otherwise)."""

    def __init__(self, context: dict[str, Any], opener: Any = None) -> None:
        self.host = str(context["host_url"])
        self.token = str(context["worker_token"])
        self._open = opener or urllib.request.build_opener(urllib.request.ProxyHandler({})).open
        self._kept: dict[str, tuple[str | None, FuturesData]] = {}
        self.downloads = 0
        self.requested: list[str] = []

    def get(self, through: str, job_id: str | None = None) -> FuturesData:
        url = f"{self.host}/api/v1/data/futures-bars?through={through}"
        if job_id:
            url += f"&job_id={job_id}"
        self.requested.append(through)
        return self._fetch(through, url)

    def live(self, source: str) -> FuturesData:
        """The latest weeks of 1-minute prices from a live source (live trading)."""
        self.requested.append(f"live:{source}")
        return self._fetch(f"live:{source}", f"{self.host}/api/v1/data/futures-live?source={source}")

    def _fetch(self, key: str, url: str) -> FuturesData:
        etag, data = self._kept.get(key, (None, None))
        headers = {"Authorization": "Bearer " + self.token}
        if etag and data is not None:
            headers["If-None-Match"] = etag
        req = urllib.request.Request(url, headers=headers)
        try:
            with self._open(req, timeout=300) as resp:
                raw = resp.read()
                new_etag = resp.headers.get("ETag")
        except urllib.error.HTTPError as exc:
            if exc.code == 304 and data is not None:
                return data
            raise http.HttpError(exc.code, exc.read().decode("utf-8", "replace")[:300], url) from None
        except (urllib.error.URLError, OSError) as exc:
            raise http.HttpConnectionError(str(exc)) from None
        data = from_npz(raw)
        self.downloads += 1
        self._kept[key] = (new_etag, data)
        return data


# ------------------------------------------------------------------ decision bars


@dataclass(frozen=True)
class Bars:
    """Decision bars of `size` minutes built from the minute grid, day by day from each
    day's open (every bar size divides a 390 or 210 minute session evenly).

    Per bar: start and end (first and one-past-last minute index of the grid; a
    decision on bar k fills at minute end[k]), day, minute (minutes since the open at
    the bar's start), session (minutes in that day's session), first (first bar of its
    day), complete (the bar has all its minutes). Per symbol and bar: open, high, low,
    close, volume, instrument, tradable."""

    size: int
    symbols: tuple[str, ...]
    times: np.ndarray
    start: np.ndarray
    end: np.ndarray
    day: np.ndarray
    minute: np.ndarray
    session: np.ndarray
    first: np.ndarray
    complete: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    instrument: np.ndarray
    tradable: np.ndarray

    @property
    def n(self) -> int:
        return int(self.start.shape[0])

    def row(self, symbol: str) -> int:
        return self.symbols.index(symbol)

    def series(self, symbol: str) -> "Series":
        k = self.row(symbol)
        return Series(self, self.open[k], self.high[k], self.low[k], self.close[k], self.volume[k],
                      self.instrument[k], self.tradable[k])


@dataclass(frozen=True)
class Series:
    """One symbol's decision bars, plus the shared timing arrays (bars.day, bars.minute...)."""

    bars: Bars
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    instrument: np.ndarray
    tradable: np.ndarray


def resample(data: FuturesData, size: int) -> Bars:
    """Decision bars of `size` minutes (1, 3, 5 or 15)."""
    size = int(size)
    if size not in BAR_SIZES:
        raise ValueError(f"bar size must be one of {BAR_SIZES} minutes, not {size}")
    if data.n_minutes == 0:
        raise ValueError("no prices to build bars from")
    starts = np.concatenate([np.arange(s, e, size, dtype=np.int64) for s, e in zip(data.day_start, data.day_end)])
    day = data.day[starts].astype(np.int64)
    ends = np.minimum(starts + size, data.day_end[day])
    if size == 1:
        o, h, lo, c, v = data.open, data.high, data.low, data.close, data.volume
    else:
        o = data.open[:, starts]
        h = np.maximum.reduceat(data.high, starts, axis=1)
        lo = np.minimum.reduceat(data.low, starts, axis=1)
        c = data.close[:, ends - 1]
        v = np.add.reduceat(data.volume, starts, axis=1)
    first = np.r_[True, day[1:] != day[:-1]]
    return Bars(size, data.symbols, data.times[starts], starts, ends, day, data.minute[starts].astype(np.int64),
                data.session[day], first, (ends - starts) == size, o, h, lo, c, v, data.instrument[:, starts],
                data.tradable[:, day])
