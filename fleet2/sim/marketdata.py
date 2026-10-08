"""Price data on a worker: fetched from the coordinator into memory, aligned, and shown to
models only through History, the lookahead guard.

Wire format (GET /api/v1/data/bars?market=stocks|crypto, gzip-compressed JSON, worker
bearer token): {"market", "timeframe", "feed", "updated_at",
"symbols": {"SPY": {"t": [epoch seconds of bar start, ascending], "o": [...], "h": [...],
"l": [...], "c": [...], "v": [...]}, ...}}. Nothing is written to disk.

Timing rule used everywhere (backtest and paper trading): bar i covers [t[i], t[i+1]).
A decision made "at bar i" happens when bar i opens, so it may only use bars 0..i-1,
which have closed; its orders fill at bar i's open. History(data, i) enforces this: it
holds the cut-off i and hands out copies of data before it, never a view into the full
arrays, so no model can reach a later bar even by accident.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from fleet2.common import http


class LookaheadError(IndexError):
    """A model asked for a bar at or after the decision time."""


@dataclass(frozen=True)
class MarketData:
    """Aligned bars of one market: arrays of shape (symbols, bars), NaN where a symbol has
    no bar at that time (not listed yet, or no trade in that bar)."""

    market: str
    timeframe: str
    feed: str
    symbols: tuple[str, ...]
    times: np.ndarray  # int64 epoch seconds, ascending, shape (bars,)
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    @property
    def n_bars(self) -> int:
        return int(self.times.shape[0])

    def row(self, symbol: str) -> int:
        return self.symbols.index(symbol)

    def last_close(self) -> np.ndarray:
        """Close carried forward over missing bars (for valuing a held position);
        NaN before a symbol's first bar."""
        out = np.empty_like(self.close)
        steps = np.arange(self.close.shape[1])
        for k in range(self.close.shape[0]):
            row = self.close[k]
            idx = np.where(~np.isnan(row), steps, -1)
            np.maximum.accumulate(idx, out=idx)
            out[k] = np.where(idx >= 0, row[np.maximum(idx, 0)], np.nan)
        return out

    def slice(self, start: int, stop: int) -> "MarketData":
        """Bars [start, stop) only (tests use this to cut the future off)."""
        sl = slice(start, stop)
        return MarketData(self.market, self.timeframe, self.feed, self.symbols, self.times[sl],
                          self.open[:, sl], self.high[:, sl], self.low[:, sl], self.close[:, sl], self.volume[:, sl])


def from_payload(payload: dict[str, Any], symbols: tuple[str, ...] | None = None) -> MarketData:
    """Align the coordinator's per-symbol columns on the union of their timestamps."""
    series = payload.get("symbols") or {}
    names = tuple(s for s in (symbols or tuple(series)) if s in series and series[s].get("t"))
    if not names:
        raise ValueError(f"no price data for market {payload.get('market')!r}; run a data refresh first")
    times = np.unique(np.concatenate([np.asarray(series[s]["t"], dtype=np.int64) for s in names]))
    shape = (len(names), times.shape[0])
    cols = {k: np.full(shape, np.nan) for k in ("o", "h", "l", "c", "v")}
    for k, name in enumerate(names):
        t = np.asarray(series[name]["t"], dtype=np.int64)
        pos = np.searchsorted(times, t)
        for key, arr in cols.items():
            arr[k, pos] = np.asarray(series[name][key], dtype=float)
    return MarketData(str(payload.get("market")), str(payload.get("timeframe")), str(payload.get("feed", "")),
                      names, times, cols["o"], cols["h"], cols["l"], cols["c"], cols["v"])


def fetch(host_url: str, token: str, market: str, timeout: float = 120.0) -> dict[str, Any]:
    """Download one market's bars from the coordinator into memory."""
    raw = http.get_bytes(f"{host_url}/api/v1/data/bars?market={market}", token=token, timeout=timeout)
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def load(context: dict[str, Any], market: str) -> MarketData:
    """MarketData for a job, from the job context the agent passes (address and token)."""
    return from_payload(fetch(str(context["host_url"]), str(context["worker_token"]), market))


class History:
    """What a model may see when it decides at bar `now_index`: bars before it, only.

    Every accessor returns a fresh copy, so a model can neither look ahead nor change
    the data. `bars` limits the copy to the most recent bars (cheap for long histories).
    """

    def __init__(self, data: MarketData, now_index: int) -> None:
        if not 0 <= now_index <= data.n_bars:
            raise LookaheadError(f"decision index {now_index} outside 0..{data.n_bars}")
        self._data = data
        self._n = now_index

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._data.symbols

    @property
    def market(self) -> str:
        return self._data.market

    def __len__(self) -> int:
        """How many closed bars are visible."""
        return self._n

    @property
    def now(self) -> int | None:
        """Decision time (epoch seconds of the bar about to open), None past the last bar."""
        return int(self._data.times[self._n]) if self._n < self._data.n_bars else None

    def _window(self, arr: np.ndarray, symbol: str, bars: int | None) -> np.ndarray:
        k = self._data.row(symbol)
        start = 0 if bars is None else max(0, self._n - int(bars))
        return np.array(arr[k, start:self._n], dtype=float, copy=True)

    def close(self, symbol: str, bars: int | None = None) -> np.ndarray:
        return self._window(self._data.close, symbol, bars)

    def open(self, symbol: str, bars: int | None = None) -> np.ndarray:
        return self._window(self._data.open, symbol, bars)

    def high(self, symbol: str, bars: int | None = None) -> np.ndarray:
        return self._window(self._data.high, symbol, bars)

    def low(self, symbol: str, bars: int | None = None) -> np.ndarray:
        return self._window(self._data.low, symbol, bars)

    def volume(self, symbol: str, bars: int | None = None) -> np.ndarray:
        return self._window(self._data.volume, symbol, bars)

    def closes(self, bars: int | None = None) -> np.ndarray:
        """Every symbol's closes, shape (symbols, bars), in self.symbols order."""
        start = 0 if bars is None else max(0, self._n - int(bars))
        return np.array(self._data.close[:, start:self._n], dtype=float, copy=True)

    def at(self, symbol: str, index: int, field: str = "close") -> float:
        """One value by absolute bar index; LookaheadError for a bar not closed yet."""
        if index < 0:
            index += self._n
        if not 0 <= index < self._n:
            raise LookaheadError(f"bar {index} is not visible at decision bar {self._n}")
        return float(getattr(self._data, field)[self._data.row(symbol), index])
