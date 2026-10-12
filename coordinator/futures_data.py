"""Futures prices on the coordinator: MES and MNQ 1-minute bars of the regular session.

Where they come from (one source at a time, chosen at start from .env):
- Databento (DATABENTO_API_KEY set): CME Globex, dataset GLBX.MDP3, schema ohlcv-1m, the
  front contract of each day by volume ("MES.v.0"). Pay-as-you-go: before every
  download the coordinator asks Databento what it will cost and refuses when the cost
  of bringing every futures symbol up to date is above max_download_usd in
  config/topstep.toml. The instrument id (the actual contract) is stored on every bar.
- Proxy (no Databento key): SPY and QQQ 1-minute bars from Alpaca's free IEX feed,
  scaled to roughly index points and rounded to whole ticks. Labelled "proxy"
  everywhere; good for building and testing, never for a "ready for a Combine" verdict.
- Synthetic (FLEET_FAKE_BROKER=1 demos and tests): made-up prices, labelled so.

Only regular-session bars are kept (fleet2.sim.cme_session). One symbol's bars always
come from one source: when the source changes (a Databento key is added), the next
refresh replaces that symbol's stored bars instead of mixing the two.

Workers get the bars as one compressed numpy file (.npz) with an ETag, cut off at the
end of the period they may see: training, held-out or lockbox. The three periods are
trading-day ranges fixed once (futures_periods) and never moved. Bars after the lockbox
(newer prices) are not served at all; they are for shadow trading later.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import logging
import re
import threading
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import zlib
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import psycopg
from psycopg.rows import tuple_row

from coordinator.config import Config
from coordinator.data import ALPACA_LIMITER, MISSING_KEYS_MESSAGE, RateLimiter
from coordinator.errors import Conflict
from coordinator.settings import get_setting, put_setting
from fleet2 import universe
from fleet2.sim import cme_session

log = logging.getLogger(__name__)

TIMEFRAME = "1Min"
WINDOW = timedelta(days=31)  # one refresh step downloads at most this much of one symbol
DATABENTO_URL = "https://hist.databento.com/v0"
NO_SOURCE_MESSAGE = ("Add DATABENTO_API_KEY (paid futures prices) or your Alpaca paper keys (the free SPY/QQQ "
                     "stand-in) to .env on box1")
MIN_DAYS_TO_FIX_PERIODS = 250  # about a year: fixing the periods on less would waste the history
FEED_TEXT = {
    "databento": "Databento (real MES and MNQ prices)",
    "proxy": "proxy: SPY and QQQ standing in for MES and MNQ",
    "synthetic": "synthetic: made-up prices for the demo",
}


# ------------------------------------------------------------------ settings


def max_download_usd(path: Path) -> float:
    """[data] max_download_usd from config/topstep.toml; 0 (refuse every paid download)
    when the file or the value is missing, so a typo can never spend money."""
    if not path.is_file():
        return 0.0
    with path.open("rb") as fh:
        raw = (tomllib.load(fh).get("data") or {}).get("max_download_usd")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw < 0:
        return 0.0
    return float(raw)


# ------------------------------------------------------------------ sources


class FuturesSource(Protocol):
    """fetch returns bars {"t": epoch seconds of the bar start, "o", "h", "l", "c", "v",
    "iid": instrument id} for start <= t < end, ascending. cost is in US dollars."""

    feed: str

    def fetch(self, symbol: str, start: datetime, end: datetime) -> list[dict]: ...

    def cost(self, symbol: str, start: datetime, end: datetime) -> float: ...


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else _utc(value).isoformat().replace("+00:00", "Z")


class DatabentoError(RuntimeError):
    pass


class _PastAvailableEnd(DatabentoError):
    """A 422 from Databento naming the latest end it will serve: "data_end_after_available_end"
    (its history runs a few minutes behind the market) or "dataset_unavailable_range" (the
    last day or so of CME prices needs a live-data license, which usage-based accounts
    do not have)."""

    def __init__(self, end: datetime) -> None:
        super().__init__(f"Databento has data up to {_iso(end)}")
        self.end = end


class _Busy(DatabentoError):
    """A busy Databento (a 5xx answer) or a lost connection: worth another try."""


_TIME = r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
_AVAILABLE_UP_TO = re.compile(r"available up to '?" + _TIME)
_END_BEFORE = re.compile(r"end time before '?" + _TIME)
_LATEST_END_CASES = ("data_end_after_available_end", "dataset_unavailable_range")


def _read_time(text: str) -> datetime | None:
    text = re.sub(r"(\.\d{6})\d+", r"\1", text.strip().replace("Z", "+00:00"))
    try:
        return _utc(datetime.fromisoformat(text))
    except ValueError:
        return None


def _available_end(detail: str) -> datetime | None:
    """The latest end Databento will serve, from its 422 message, or None:
    "... has data available up to '2026-10-09 02:00:00+00:00' ..." (that very time), or
    "... Try again with an end time before 2026-10-08T18:37:54.511021000Z." (the whole
    minute before it, so the end is strictly earlier)."""
    match = _AVAILABLE_UP_TO.search(detail)
    if match:
        return _read_time(match.group(1))
    match = _END_BEFORE.search(detail)
    if match:
        before = _read_time(match.group(1))
        if before is not None:
            return (before - timedelta(microseconds=1)).replace(second=0, microsecond=0)
    return None


class DatabentoSource:
    """Databento's historical HTTP API (the same calls the official SDK makes):
    POST {DATABENTO_URL}/metadata.get_cost and /timeseries.get_range, form fields, the
    key as the basic-auth user name. Bars come back as plain CSV (encoding=csv,
    compression=none, pretty_px=true), so no Databento library is needed.

    Symbols are Databento's continuous contracts by volume: "MES.v.0" is, each day, the
    MES contract that traded the most the day before. The CSV carries the real
    contract's instrument_id on every row."""

    feed = "databento"

    ATTEMPTS = 3  # a busy answer (500, 502, 503, 504) or a lost connection is tried again...
    PAUSES = (5.0, 20.0)  # ...after these many seconds

    def __init__(self, key: str, opener: Callable[..., Any] | None = None, timeout: float = 120.0,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self._auth = "Basic " + base64.b64encode(f"{key}:".encode()).decode()
        self._open = opener or urllib.request.urlopen
        self._timeout = timeout
        self._sleep = sleep

    def _post(self, path: str, fields: dict[str, str]) -> bytes:
        for attempt in range(self.ATTEMPTS):
            try:
                return self._post_once(path, fields)
            except _Busy as exc:
                if attempt == self.ATTEMPTS - 1:
                    raise DatabentoError(f"{exc} ({self.ATTEMPTS} tries). Press Load futures prices again later: "
                                         "what is already downloaded is kept.") from None
                log.warning("%s; trying again", exc)
                self._sleep(self.PAUSES[min(attempt, len(self.PAUSES) - 1)])
        raise AssertionError("unreachable")

    def _post_once(self, path: str, fields: dict[str, str]) -> bytes:
        body = urllib.parse.urlencode(fields).encode()
        req = urllib.request.Request(f"{DATABENTO_URL}/{path}", data=body, method="POST",
                                     headers={"Authorization": self._auth, "Accept": "application/json",
                                              "Content-Type": "application/x-www-form-urlencoded",
                                              "User-Agent": "fleet-v2"})
        try:
            with self._open(req, timeout=self._timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            if exc.code in (401, 403):
                raise DatabentoError("Databento refused DATABENTO_API_KEY in .env on box1: check the key") from None
            latest = exc.code == 422 and any(case in detail for case in _LATEST_END_CASES)
            end = _available_end(detail) if latest else None
            if end is not None:
                raise _PastAvailableEnd(end) from None
            if exc.code in (500, 502, 503, 504):
                raise _Busy(f"Databento was busy and answered {exc.code}") from None
            raise DatabentoError(f"Databento answered {exc.code}: {detail[:300]}") from None
        except (urllib.error.URLError, OSError) as exc:
            raise _Busy(f"Cannot reach Databento: {exc}") from None

    def _fields(self, symbol: str, start: datetime, end: datetime) -> dict[str, str]:
        return {
            "dataset": universe.FUTURES["dataset"],
            "schema": universe.FUTURES["schema"],
            "symbols": universe.CONTRACTS[symbol]["databento"],
            "stype_in": "continuous",
            "start": _iso(start) or "",
            "end": _iso(end) or "",
        }

    def _ranged(self, path: str, symbol: str, start: datetime, end: datetime,
                extra: dict[str, str] | None = None) -> bytes | None:
        """POST a request for start..end. When `end` is past what Databento serves (its
        history runs a few minutes behind the market, and the last day or so needs a
        live-data license), ask again up to the end it names.
        None when it has nothing after `start` yet."""
        for _ in range(3):  # Databento may name an earlier end twice (behind, then unlicensed)
            if end <= start:
                return None
            try:
                return self._post(path, {**self._fields(symbol, start, end), **(extra or {})})
            except _PastAvailableEnd as exc:
                if exc.end >= end:
                    break
                end = exc.end
        raise DatabentoError(f"Databento kept refusing the end time {_iso(end)} for {symbol}")

    def cost(self, symbol: str, start: datetime, end: datetime) -> float:
        raw = self._ranged("metadata.get_cost", symbol, start, end)
        if raw is None:
            return 0.0
        try:
            return float(json.loads(raw))
        except (TypeError, ValueError):
            raise DatabentoError(f"Databento sent an unreadable cost: {raw[:100]!r}") from None

    def fetch(self, symbol: str, start: datetime, end: datetime) -> list[dict]:
        extra = {"stype_out": "instrument_id", "encoding": "csv", "compression": "none", "pretty_px": "true",
                 "pretty_ts": "false", "map_symbols": "false"}
        raw = self._ranged("timeseries.get_range", symbol, start, end, extra)
        return [] if raw is None else parse_databento_csv(raw.decode("utf-8", "replace"))


def _price(text: str) -> float:
    """A price from Databento's CSV: a decimal with pretty_px, else an integer in
    billionths of a point (both are handled, so a change of default cannot scale prices)."""
    value = float(text)
    if "." not in text and "e" not in text.lower() and abs(value) >= 1e7:
        value /= 1e9
    return value


def _epoch(text: str) -> int:
    """ts_event: nanoseconds since 1970 (pretty_ts=false), or an ISO time."""
    text = text.strip()
    if text.isdigit():
        return int(text) // 1_000_000_000
    return int(datetime.fromisoformat(text[:19]).replace(tzinfo=timezone.utc).timestamp())


def parse_databento_csv(text: str) -> list[dict]:
    """ohlcv-1m rows: ts_event (bar start), instrument_id, open, high, low, close, volume."""
    reader = csv.DictReader(io.StringIO(text))
    out = []
    for row in reader:
        out.append({"t": _epoch(row["ts_event"]), "o": _price(row["open"]), "h": _price(row["high"]),
                    "l": _price(row["low"]), "c": _price(row["close"]), "v": float(row["volume"]),
                    "iid": int(row["instrument_id"])})
    out.sort(key=lambda b: b["t"])
    return out


class ProxySource:
    """The free stand-in: SPY (for MES) and QQQ (for MNQ) 1-minute bars from Alpaca's IEX
    feed, unadjusted, through the shared Alpaca rate limiter. Prices are scaled to
    roughly index points and rounded to the futures' tick, so stops and costs in ticks
    mean about the same as on the real contract. IEX sees a small share of all trades,
    so quiet minutes have no bar; the worker fills them with the last price."""

    feed = "proxy"

    def __init__(self, config: Config, limiter: RateLimiter | None = None) -> None:
        self._key = config.alpaca_paper_key_id
        self._secret = config.alpaca_paper_secret
        self.limiter = limiter or ALPACA_LIMITER
        self._client: Any = None
        self._lock = threading.Lock()

    def cost(self, symbol: str, start: datetime, end: datetime) -> float:
        return 0.0

    def _stocks(self) -> Any:
        if not (self._key and self._secret):
            raise RuntimeError(NO_SOURCE_MESSAGE)
        with self._lock:
            if self._client is None:
                from alpaca.data.historical import StockHistoricalDataClient

                self._client = StockHistoricalDataClient(self._key, self._secret)
            return self._client

    def fetch(self, symbol: str, start: datetime, end: datetime) -> list[dict]:
        from alpaca.common.exceptions import APIError
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        spec = universe.CONTRACTS[symbol]
        stand_in = spec["proxy"]
        request = StockBarsRequest(symbol_or_symbols=stand_in, timeframe=TimeFrame.Minute, start=_utc(start),
                                   end=_utc(end) - timedelta(seconds=1), feed=DataFeed.IEX, adjustment=Adjustment.RAW)
        client = self._stocks()
        self.limiter.acquire()
        try:
            bars = list(client.get_stock_bars(request).data.get(stand_in, []))
        except APIError as exc:
            if getattr(exc, "status_code", None) in (401, 403):
                raise RuntimeError(MISSING_KEYS_MESSAGE) from None
            raise RuntimeError(f"Alpaca error for {stand_in}: {exc}") from None
        return [_scaled(int(_utc(b.timestamp).timestamp()), b.open, b.high, b.low, b.close, b.volume, spec)
                for b in bars]


def _scaled(t: int, o: float, h: float, l: float, c: float, v: float, spec: dict[str, Any]) -> dict:
    scale, tick = spec["proxy_scale"], spec["tick"]

    def px(x: float) -> float:
        return round(float(x) * scale / tick) * tick

    return {"t": t, "o": px(o), "h": px(h), "l": px(l), "c": px(c), "v": float(v), "iid": 0}


class FakeFuturesSource:
    """Deterministic made-up 1-minute bars for tests and FLEET_FAKE_BROKER=1 demos: a
    random walk on the tick grid for every regular-session minute since the futures
    launched, with a new contract (instrument id) every quarter and a small jump at each
    roll, so the roll handling is exercised. Any window is a slice of one series."""

    feed = "synthetic"

    def __init__(self, seed: int = 0, first: date | None = None, last: date | None = None) -> None:
        self.seed = int(seed)
        self.first = first or date.fromisoformat(universe.FUTURES["start"])
        self.last = last or date(2027, 12, 31)
        self._cache: dict[str, dict[str, np.ndarray]] = {}
        self._lock = threading.Lock()
        self.calls: list[tuple[str, datetime, datetime]] = []

    def cost(self, symbol: str, start: datetime, end: datetime) -> float:
        return 0.0

    def series(self, symbol: str) -> dict[str, np.ndarray]:
        with self._lock:
            if symbol in self._cache:
                return self._cache[symbol]
        spec = universe.CONTRACTS[symbol]
        days = cme_session.trading_days(self.first, self.last)
        starts, lengths, quarter = [], [], []
        for d in days:
            s = cme_session.session(d)
            starts.append(s[0])
            lengths.append((s[1] - s[0]) // 60)
            quarter.append(d.year * 4 + (d.month - 1) // 3)
        lengths_a = np.asarray(lengths)
        t = np.concatenate([np.arange(n, dtype=np.int64) * 60 + s for s, n in zip(starts, lengths)])
        day_of = np.repeat(np.arange(len(days)), lengths_a)
        rng = np.random.default_rng([self.seed, zlib.crc32(symbol.encode())])
        tick = spec["tick"]
        level = 2800.0 if symbol == "MES" else 7600.0
        vol = 0.0006  # per minute
        steps = rng.standard_normal(t.shape[0]) * vol + 0.000002
        overnight = np.zeros(t.shape[0])
        first_minute = np.r_[0, np.cumsum(lengths_a)[:-1]]
        overnight[first_minute] = rng.standard_normal(len(days)) * 0.006
        close = level * np.exp(np.cumsum(steps + overnight))
        open_ = np.empty_like(close)
        open_[0] = level
        open_[1:] = close[:-1]
        open_[first_minute[1:]] = close[first_minute[1:]] * np.exp(-steps[first_minute[1:]])
        wiggle = np.abs(rng.standard_normal((2, t.shape[0]))) * vol * 0.6 * close
        high = np.maximum(open_, close) + wiggle[0]
        low = np.minimum(open_, close) - wiggle[1]
        q = np.asarray(quarter)[day_of]
        iid = (100_000 + q - q.min()).astype(np.int64) + (1000 if symbol == "MNQ" else 0)
        snap = lambda x: np.round(x / tick) * tick  # noqa: E731
        series = {"t": t, "o": snap(open_), "h": snap(high), "l": snap(low), "c": snap(close),
                  "v": np.round(np.exp(6.0 + 0.5 * rng.standard_normal(t.shape[0]))), "iid": iid}
        series["h"] = np.maximum.reduce([series["h"], series["o"], series["c"]])
        series["l"] = np.minimum.reduce([series["l"], series["o"], series["c"]])
        with self._lock:
            self._cache[symbol] = series
        return series

    def fetch(self, symbol: str, start: datetime, end: datetime) -> list[dict]:
        self.calls.append((symbol, start, end))
        s = self.series(symbol)
        lo = int(np.searchsorted(s["t"], int(_utc(start).timestamp()), side="left"))
        hi = int(np.searchsorted(s["t"], int(_utc(end).timestamp()), side="left"))
        return [{"t": int(s["t"][i]), "o": float(s["o"][i]), "h": float(s["h"][i]), "l": float(s["l"][i]),
                 "c": float(s["c"][i]), "v": float(s["v"][i]), "iid": int(s["iid"][i])} for i in range(lo, hi)]


def make_futures_source(config: Config) -> FuturesSource:
    """Synthetic for demos, Databento when its key is set, else the free proxy."""
    if config.fake_broker:
        return FakeFuturesSource(seed=7)
    if config.databento_api_key:
        return DatabentoSource(config.databento_api_key)
    return ProxySource(config)


# ------------------------------------------------------------------ refresh one symbol


def _market_start() -> datetime:
    d = date.fromisoformat(universe.FUTURES["start"])
    return datetime.combine(d, cme_session.OPEN, cme_session.CHICAGO).astimezone(timezone.utc)


def _stored(conn: psycopg.Connection, symbol: str) -> dict[str, Any]:
    return conn.execute(
        "SELECT count(*) AS n, max(ts) AS last_ts, min(feed) AS feed_a, max(feed) AS feed_b FROM bars "
        "WHERE symbol = %s AND timeframe = %s", (symbol, TIMEFRAME)).fetchone()


def _next_start(stored: dict[str, Any], feed: str) -> datetime:
    """Where the symbol's next download starts: after its last bar, or from the
    beginning when it has none or they came from another source (to be replaced)."""
    same = stored["n"] and stored["feed_a"] == feed and stored["feed_b"] == feed
    return _utc(stored["last_ts"]) + timedelta(minutes=1) if same else _market_start()


def refresh_step(conn: psycopg.Connection, source: FuturesSource, symbol: str, cap_usd: float,
                 now: datetime | None = None, after: datetime | None = None) -> dict[str, Any]:
    """One unit of refresh work: one window of at most 31 days of one symbol. Never raises.

    Before a symbol's first paid download (no `after`) it asks the source what bringing
    EVERY futures symbol up to date would cost and refuses when that is above `cap_usd`,
    so the whole refresh run can never spend more than the cap. The later steps of the
    run (with `after`) only cover part of what was priced, so they are not priced again.
    Returns {"symbol", "market", "added", "bars", "last_ts", "done", "error", "from",
    "cursor", "feed", "cost_usd"}."""
    now = _utc(now) if now is not None else datetime.now(timezone.utc)
    out: dict[str, Any] = {"symbol": symbol, "market": "futures", "added": 0, "bars": 0, "last_ts": None,
                           "done": False, "error": None, "from": None, "cursor": None,
                           "feed": source.feed, "cost_usd": 0.0}
    try:
        if symbol not in universe.CONTRACTS:
            raise ValueError(f"{symbol} is not a futures symbol")
        stored = _stored(conn, symbol)
        replacing = bool(stored["n"]) and not (stored["feed_a"] == source.feed == stored["feed_b"])
        out["bars"] = 0 if replacing else int(stored["n"])
        out["last_ts"] = None if replacing else _iso(stored["last_ts"])
        window_start = _next_start(stored, source.feed)
        if after is not None and not replacing:
            window_start = max(window_start, _utc(after))
        window_end = min(window_start + WINDOW, now)
        out["from"], out["cursor"] = _iso(window_start), _iso(window_end)
        out["done"] = window_end >= now
        if window_start >= now:
            return out

        if source.feed == "databento" and after is None:
            # Priced once, at a symbol's first step: the later steps of the same run only
            # cover part of what was priced here.
            total = 0.0
            for other in universe.FUTURES["symbols"]:
                begin = window_start if other == symbol else _next_start(_stored(conn, other), source.feed)
                if begin < now:
                    total += float(source.cost(other, begin, now))
            out["cost_usd"] = round(total, 2)
            if total > cap_usd:
                raise RuntimeError(
                    f"Databento says bringing the futures prices up to date costs ${total:,.2f}, above your cap of "
                    f"${cap_usd:,.2f} (max_download_usd in config/topstep.toml). Nothing was downloaded.")

        fetched = source.fetch(symbol, window_start, window_end)
        start_s, end_s, now_s = (int(window_start.timestamp()), int(window_end.timestamp()), int(now.timestamp()))
        keep = [b for b in fetched if start_s <= int(b["t"]) < end_s and int(b["t"]) + 60 <= now_s]
        if keep:
            ok = cme_session.in_session(np.asarray([int(b["t"]) for b in keep], dtype=np.int64))
            keep = [b for b, good in zip(keep, ok) if good]

        with conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"bars:{symbol}:{TIMEFRAME}",))
            if replacing:
                conn.execute("DELETE FROM bars WHERE symbol = %s AND timeframe = %s", (symbol, TIMEFRAME))
            if keep:
                conn.cursor().executemany(
                    "INSERT INTO bars (symbol, timeframe, ts, open, high, low, close, volume, feed, instrument_id) "
                    "VALUES (%s, %s, to_timestamp(%s), %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (symbol, timeframe, ts) DO UPDATE SET open = EXCLUDED.open, high = EXCLUDED.high,"
                    " low = EXCLUDED.low, close = EXCLUDED.close, volume = EXCLUDED.volume, feed = EXCLUDED.feed,"
                    " instrument_id = EXCLUDED.instrument_id",
                    [(symbol, TIMEFRAME, int(b["t"]), b["o"], b["h"], b["l"], b["c"], b["v"], source.feed,
                      int(b.get("iid") or 0)) for b in keep])
            row = conn.execute(
                "SELECT min(ts) AS first_ts, max(ts) AS last_ts, count(*) AS n FROM bars "
                "WHERE symbol = %s AND timeframe = %s", (symbol, TIMEFRAME)).fetchone()
            conn.execute(
                "INSERT INTO bar_status (symbol, timeframe, first_ts, last_ts, bars, feed, refreshed_at, error) "
                "VALUES (%s, %s, %s, %s, %s, %s, clock_timestamp(), NULL) "
                "ON CONFLICT (symbol, timeframe) DO UPDATE SET first_ts = EXCLUDED.first_ts, last_ts = EXCLUDED.last_ts,"
                " bars = EXCLUDED.bars, feed = EXCLUDED.feed, refreshed_at = EXCLUDED.refreshed_at, error = NULL",
                (symbol, TIMEFRAME, row["first_ts"], row["last_ts"], row["n"], source.feed))
        out["added"] = max(0, int(row["n"]) - out["bars"])
        out["bars"], out["last_ts"] = int(row["n"]), _iso(row["last_ts"])
        return out
    except Exception as exc:  # noqa: BLE001 - a failing symbol is reported, not raised
        message = str(exc) or type(exc).__name__
        log.warning("futures refresh %s failed: %s", symbol, message)
        out["error"], out["done"] = message, False
        try:
            conn.execute(
                "INSERT INTO bar_status (symbol, timeframe, feed, error) VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (symbol, timeframe) DO UPDATE SET error = EXCLUDED.error",
                (symbol, TIMEFRAME, source.feed, message[:1000]))
        except Exception:  # noqa: BLE001
            log.exception("could not store the futures refresh error for %s", symbol)
        return out


# ------------------------------------------------------------------ the three periods


def _stored_days(conn: psycopg.Connection) -> list[date]:
    rows = conn.execute(
        "SELECT DISTINCT (ts AT TIME ZONE 'America/Chicago')::date AS d FROM bars WHERE timeframe = %s "
        "AND symbol = ANY(%s) ORDER BY d", (TIMEFRAME, list(universe.FUTURES["symbols"]))).fetchall()
    return [r["d"] for r in rows]


def split_days(days: list[date]) -> dict[str, str]:
    """Last trading day of training, held-out and lockbox: the first 60%, the next 25%
    and the last 15% of the days (universe.FUTURES_PERIODS)."""
    n = len(days)
    cut_train = max(1, round(n * universe.FUTURES_PERIODS[0][1]))
    cut_held = max(cut_train + 1, round(n * (universe.FUTURES_PERIODS[0][1] + universe.FUTURES_PERIODS[1][1])))
    return {"train_end": days[cut_train - 1].isoformat(), "held_out_end": days[cut_held - 1].isoformat(),
            "lockbox_end": days[-1].isoformat()}


def futures_periods(conn: psycopg.Connection) -> dict[str, Any]:
    """The three periods, fixed the first time they are needed and never moved, so model
    search can never train on days that later rank it. Raises Conflict until there are
    enough prices to split."""
    fixed = get_setting(conn, "futures_periods", None)
    if fixed:
        return fixed
    days = _stored_days(conn)
    if len(days) < MIN_DAYS_TO_FIX_PERIODS:
        raise Conflict(f"Load futures prices first: {len(days)} trading days stored, at least "
                       f"{MIN_DAYS_TO_FIX_PERIODS} needed (Assign a job > Futures prices)")
    feed = conn.execute("SELECT max(feed) AS f FROM bar_status WHERE timeframe = %s", (TIMEFRAME,)).fetchone()["f"]
    periods = {**split_days(days), "first_day": days[0].isoformat(), "fixed_with": feed}
    put_setting(conn, "futures_periods", periods, "system", "futures_periods_set")
    return periods


PERIOD_END = {"train": "train_end", "held_out": "held_out_end", "lockbox": "lockbox_end"}


def period_cutoff(periods: dict[str, Any], through: str) -> datetime:
    """The first moment NOT served for `through`: midnight Chicago after its last day."""
    last = date.fromisoformat(periods[PERIOD_END[through]])
    return datetime.combine(last + timedelta(days=1), datetime.min.time(), cme_session.CHICAGO).astimezone(timezone.utc)


# ------------------------------------------------------------------ serving workers

_payload_cache: dict[str, tuple[Any, bytes, str]] = {}
_payload_lock = threading.Lock()


def futures_feed(conn: psycopg.Connection) -> str | None:
    """The source of the stored futures prices: databento, proxy, synthetic, or None."""
    rows = conn.execute("SELECT DISTINCT feed FROM bar_status WHERE timeframe = %s AND bars > 0 AND symbol = ANY(%s)",
                        (TIMEFRAME, list(universe.FUTURES["symbols"]))).fetchall()
    feeds = {r["feed"] for r in rows}
    if not feeds:
        return None
    return feeds.pop() if len(feeds) == 1 else "mixed"


def payload(conn: psycopg.Connection, through: str) -> tuple[bytes, str]:
    """(.npz bytes, ETag) of every futures bar up to the end of period `through`.

    The file holds, per symbol, the arrays SYMBOL_t (epoch seconds of the bar start),
    SYMBOL_o/h/l/c/v and SYMBOL_iid (instrument id), plus "meta": a JSON string with
    the source (feed), the periods and the period served. Rebuilt only when a symbol's
    bar_status changes. LookupError when nothing is stored yet."""
    if through not in PERIOD_END:
        raise ValueError(f"unknown period {through!r}")
    periods = futures_periods(conn)
    status = conn.execute(
        "SELECT symbol, refreshed_at, bars, feed FROM bar_status WHERE timeframe = %s AND bars > 0 "
        "AND symbol = ANY(%s) ORDER BY symbol", (TIMEFRAME, list(universe.FUTURES["symbols"]))).fetchall()
    if not status:
        raise LookupError("No futures prices yet: run a Futures prices job")
    key = (tuple((r["symbol"], r["refreshed_at"], r["bars"], r["feed"]) for r in status), json.dumps(periods, sort_keys=True))
    with _payload_lock:
        cached = _payload_cache.get(through)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
    cutoff = period_cutoff(periods, through)
    arrays: dict[str, np.ndarray] = {}
    for r in status:
        symbol = r["symbol"]
        n = conn.execute("SELECT count(*) AS n FROM bars WHERE symbol = %s AND timeframe = %s AND ts < %s",
                         (symbol, TIMEFRAME, cutoff)).fetchone()["n"]
        cols = {k: np.empty(n, dtype=np.int64 if k in ("t", "iid") else np.float64) for k in ("t", "o", "h", "l", "c", "v", "iid")}
        filled = 0
        with conn.cursor(row_factory=tuple_row) as cur:
            cur.execute("SELECT extract(epoch FROM ts)::bigint, open, high, low, close, volume, coalesce(instrument_id, 0) "
                        "FROM bars WHERE symbol = %s AND timeframe = %s AND ts < %s ORDER BY ts",
                        (symbol, TIMEFRAME, cutoff))
            while True:
                chunk = cur.fetchmany(50_000)
                if not chunk:
                    break
                block = np.asarray(chunk, dtype=np.float64)
                stop = filled + block.shape[0]
                for j, k in enumerate(("t", "o", "h", "l", "c", "v", "iid")):
                    cols[k][filled:stop] = block[:, j]
                filled = stop
        for k, arr in cols.items():
            arrays[f"{symbol}_{k}"] = arr[:filled]
    meta = {"market": "futures", "timeframe": TIMEFRAME, "feed": futures_feed(conn), "through": through,
            "periods": periods, "symbols": [r["symbol"] for r in status],
            "updated_at": _iso(max(r["refreshed_at"] for r in status))}
    buf = io.BytesIO()
    np.savez_compressed(buf, meta=np.asarray(json.dumps(meta)), **arrays)
    body = buf.getvalue()
    etag = '"' + hashlib.sha256(body).hexdigest()[:24] + '"'
    with _payload_lock:
        _payload_cache[through] = (key, body, etag)
    return body, etag


def lockbox_allowed(conn: psycopg.Connection, worker_id: str, job_id: str | None) -> bool:
    """Lockbox prices only go to a worker that holds a running Final check job."""
    if not job_id:
        return False
    try:
        row = conn.execute("SELECT 1 FROM jobs WHERE id = %s::uuid AND kind = 'final_check' AND status = 'leased' "
                           "AND lease_worker_id = %s", (job_id, worker_id)).fetchone()
    except psycopg.errors.InvalidTextRepresentation:
        return False
    return row is not None


def status_line(conn: psycopg.Connection) -> dict[str, Any]:
    """What the dashboard says about futures prices: source, days, date range."""
    feed = futures_feed(conn)
    if feed is None:
        return {"feed": None, "text": "No futures prices yet: run a Futures prices job", "proxy": False}
    row = conn.execute("SELECT min(first_ts) AS a, max(last_ts) AS b, sum(bars) AS n FROM bar_status "
                       "WHERE timeframe = %s AND bars > 0", (TIMEFRAME,)).fetchone()
    first = row["a"].astimezone(cme_session.CHICAGO).date() if row["a"] else None
    last = row["b"].astimezone(cme_session.CHICAGO).date() if row["b"] else None
    text = f"Futures prices: {FEED_TEXT.get(feed, feed)}"
    if first and last:
        text += f", {first:%b} {first.day}, {first.year} to {last:%b} {last.day}, {last.year}"
    return {"feed": feed, "text": text, "proxy": feed != "databento", "bars": int(row["n"] or 0)}
