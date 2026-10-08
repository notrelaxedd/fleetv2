"""Price bars on the coordinator: where they come from, where they are kept, what workers get.

The Alpaca keys are used here and nowhere else. The coordinator downloads each symbol's
bars once, keeps them in the `bars` table, and serves them to workers as one gzip
document per market (bars_payload). A worker's "Data refresh" job only drives
refresh_step, one symbol at a time, so its progress shows on the Fleet screen.

Stocks are daily and adjusted for splits and dividends (Adjustment.ALL): an adjustment
changes every earlier price, so a stock symbol is re-downloaded in full and REPLACES its
stored bars. Crypto is hourly and unadjusted, so it only appends, in windows of at most
120 days. A bar that has not closed yet (start + timeframe after now) is never stored.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import threading
import time
import zlib
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Protocol

import numpy as np
import psycopg
from psycopg.rows import tuple_row

from coordinator.config import Config
from fleet2 import universe

log = logging.getLogger(__name__)

TIMEFRAME_DELTA = {"1Day": timedelta(days=1), "1Hour": timedelta(hours=1)}
CRYPTO_WINDOW = timedelta(days=120)  # one crypto step fetches at most this much history

# Alpaca's free market data allows 200 requests a minute; stay well under it.
ALPACA_CALLS_PER_MINUTE = 150

MISSING_KEYS_MESSAGE = "Add your Alpaca paper keys to .env on box1 (stock prices need them)"


# --------------------------------------------------------------------- rate limit


class RateLimiter:
    """Minimum spacing between calls, safe across threads.

    Spacing (60 / per_minute seconds) rather than a token bucket: a bucket with burst
    capacity C lets C + rate*T calls through in a window T, which can pass the cap in
    the first minute. With spacing, any 60 s window holds at most per_minute calls.
    `clock` and `sleep` are injectable so tests never really wait.
    """

    def __init__(
        self,
        per_minute: float = ALPACA_CALLS_PER_MINUTE,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if per_minute <= 0:
            raise ValueError("per_minute must be positive")
        self.interval = 60.0 / per_minute
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next = float("-inf")

    def acquire(self) -> float:
        """Block until a call may go out; returns how long this caller waited (seconds)."""
        with self._lock:
            now = self._clock()
            slot = max(now, self._next)  # each caller reserves the next free slot
            self._next = slot + self.interval
        wait = slot - now
        if wait > 0:
            self._sleep(wait)
        return max(0.0, wait)


# The one limiter every Alpaca data call of this process goes through.
ALPACA_LIMITER = RateLimiter(ALPACA_CALLS_PER_MINUTE)


# ------------------------------------------------------------------- bar sources


class BarSource(Protocol):
    """Where bars come from. fetch returns closed or open bars with
    {"t": datetime UTC (bar start), "o", "h", "l", "c", "v": float}, ascending, for
    start <= t <= end. `feed` is the stock feed ("iex"/"sip"); feed_for(symbol) is the
    feed of one symbol ("crypto" for crypto symbols)."""

    feed: str

    def fetch(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> list[dict]: ...

    def feed_for(self, symbol: str) -> str: ...


def is_crypto(symbol: str) -> bool:
    return "/" in symbol


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


class AlpacaBarSource:
    """Real bars through the official alpaca-py SDK. Every call goes through the limiter.

    Stocks: StockHistoricalDataClient(key, secret).get_stock_bars(StockBarsRequest(
    symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=, end=, feed=DataFeed.IEX
    (DataFeed.SIP only when ALPACA_DATA_FEED=sip), adjustment=Adjustment.ALL)).
    Crypto: CryptoHistoricalDataClient(...).get_crypto_bars(CryptoBarsRequest(
    symbol_or_symbols=symbol, timeframe=TimeFrame.Hour, start=, end=)).
    The SDK pages through results itself (10,000 bars a page, more than one stock
    history or one 120-day crypto window), so a step is one request in practice.
    """

    def __init__(self, config: Config, limiter: RateLimiter | None = None) -> None:
        self._key = config.alpaca_paper_key_id
        self._secret = config.alpaca_paper_secret
        # Free data is IEX. SIP is a paid feed: only the exact setting "sip" selects it.
        self.feed = "sip" if config.alpaca_data_feed == "sip" else "iex"
        self.limiter = limiter or ALPACA_LIMITER
        self._stock_client: Any = None
        self._crypto_client: Any = None
        self._lock = threading.Lock()

    def feed_for(self, symbol: str) -> str:
        return "crypto" if is_crypto(symbol) else self.feed

    def _stocks(self) -> Any:
        if not (self._key and self._secret):
            raise RuntimeError(MISSING_KEYS_MESSAGE)
        with self._lock:
            if self._stock_client is None:
                from alpaca.data.historical import StockHistoricalDataClient

                self._stock_client = StockHistoricalDataClient(self._key, self._secret)
            return self._stock_client

    def _crypto(self) -> Any:
        with self._lock:
            if self._crypto_client is None:
                from alpaca.data.historical import CryptoHistoricalDataClient

                # Crypto data needs no keys; keys (when present) only raise the rate limit.
                self._crypto_client = CryptoHistoricalDataClient(self._key or None, self._secret or None)
            return self._crypto_client

    def fetch(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> list[dict]:
        from alpaca.common.exceptions import APIError

        start, end = _utc(start), _utc(end)
        try:
            if is_crypto(symbol):
                bars = self._fetch_crypto(symbol, timeframe, start, end)
            else:
                bars = self._fetch_stock(symbol, timeframe, start, end)
        except APIError as exc:
            status = getattr(exc, "status_code", None)
            if status in (401, 403):
                raise RuntimeError(
                    "Alpaca refused the keys in .env on box1: check ALPACA_PAPER_KEY_ID and ALPACA_PAPER_SECRET_KEY"
                ) from None
            if status == 429:
                raise RuntimeError("Alpaca says too many requests: try the refresh again in a minute") from None
            raise RuntimeError(f"Alpaca error for {symbol}: {exc}") from None
        return [
            {"t": _utc(b.timestamp), "o": float(b.open), "h": float(b.high), "l": float(b.low),
             "c": float(b.close), "v": float(b.volume)}
            for b in bars
        ]

    def _fetch_stock(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> list[Any]:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        if timeframe != "1Day":
            raise ValueError(f"unsupported stock timeframe {timeframe!r}")
        client = self._stocks()
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.SIP if self.feed == "sip" else DataFeed.IEX,
            adjustment=Adjustment.ALL,
        )
        self.limiter.acquire()
        return list(client.get_stock_bars(request).data.get(symbol, []))

    def _fetch_crypto(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> list[Any]:
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame

        if timeframe != "1Hour":
            raise ValueError(f"unsupported crypto timeframe {timeframe!r}")
        client = self._crypto()
        request = CryptoBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Hour, start=start, end=end,
        )
        self.limiter.acquire()
        return list(client.get_crypto_bars(request).data.get(symbol, []))


class FakeBarSource:
    """Deterministic synthetic bars for tests and FLEET_FAKE_BROKER=1 demos.

    A seeded random walk with a little drift per symbol, laid out on a fixed time grid,
    so any window of the same symbol is a slice of one series (a stock re-download gives
    identical bars, and crypto windows join without a seam). Daily bars (stocks) are on
    weekdays only, at 05:00 UTC like Alpaca's; hourly bars (crypto) every hour.
    """

    _DAILY_ORIGIN = datetime(2010, 1, 1, 5, tzinfo=timezone.utc)
    _HOURLY_ORIGIN = datetime(2019, 1, 1, tzinfo=timezone.utc)
    _DAILY_STEPS = 365 * 40
    _HOURLY_STEPS = 24 * 365 * 20

    feed = "iex"

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self._cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        self._lock = threading.Lock()

    def feed_for(self, symbol: str) -> str:
        return "crypto" if is_crypto(symbol) else self.feed

    def _series(self, symbol: str, timeframe: str) -> dict[str, np.ndarray]:
        key = (symbol, timeframe)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        daily = timeframe == "1Day"
        if not daily and timeframe != "1Hour":
            raise ValueError(f"unsupported timeframe {timeframe!r}")
        rng = np.random.default_rng([self.seed, zlib.crc32(symbol.encode()), 1 if daily else 2])
        steps = self._DAILY_STEPS if daily else self._HOURLY_STEPS
        origin = self._DAILY_ORIGIN if daily else self._HOURLY_ORIGIN
        step_s = 86400 if daily else 3600
        t0 = int(origin.timestamp())
        t = t0 + step_s * np.arange(steps, dtype=np.int64)
        if daily:
            weekday = ((t // 86400) + 3) % 7  # 1970-01-01 was a Thursday; 0=Mon .. 6=Sun
            t = t[weekday < 5]
        n = t.shape[0]
        vol = 0.015 if daily else 0.004
        drift = 0.0004 if daily else 0.00002
        start_price = 20.0 + 380.0 * rng.random()
        log_ret = drift + vol * rng.standard_normal(n)
        close = start_price * np.exp(np.cumsum(log_ret))
        open_ = np.empty(n)
        open_[0] = start_price
        open_[1:] = close[:-1] * np.exp(0.15 * vol * rng.standard_normal(n - 1))
        top = np.maximum(open_, close) * (1.0 + np.abs(rng.standard_normal(n)) * vol * 0.5)
        bottom = np.minimum(open_, close) * (1.0 - np.abs(rng.standard_normal(n)) * vol * 0.5)
        volume = np.round(np.exp(11.0 + 0.6 * rng.standard_normal(n)))
        series = {"t": t, "o": open_, "h": top, "l": bottom, "c": close, "v": volume}
        with self._lock:
            self._cache[key] = series
        return series

    def fetch(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> list[dict]:
        s = self._series(symbol, timeframe)
        lo = int(np.searchsorted(s["t"], int(_utc(start).timestamp()), side="left"))
        hi = int(np.searchsorted(s["t"], int(_utc(end).timestamp()), side="right"))
        return [
            {"t": datetime.fromtimestamp(int(s["t"][i]), timezone.utc), "o": float(s["o"][i]),
             "h": float(s["h"][i]), "l": float(s["l"][i]), "c": float(s["c"][i]), "v": float(s["v"][i])}
            for i in range(lo, hi)
        ]


def make_bar_source(config: Config) -> BarSource:
    """FakeBarSource for FLEET_FAKE_BROKER demos and tests, else the real Alpaca source."""
    if config.fake_broker:
        return FakeBarSource(seed=7)
    return AlpacaBarSource(config)


# ----------------------------------------------------------------- refresh a symbol


def _market_spec(market: str) -> dict[str, Any]:
    try:
        return universe.MARKETS[market]
    except KeyError:
        raise ValueError(f"unknown market {market!r}") from None


def _market_start(spec: dict[str, Any]) -> datetime:
    d = date.fromisoformat(spec["start"])
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return _utc(value).isoformat().replace("+00:00", "Z")


def _store_error(conn: psycopg.Connection, symbol: str, timeframe: str, feed: str | None, message: str) -> None:
    """Record the failure on bar_status without touching what is already cached."""
    conn.execute(
        "INSERT INTO bar_status (symbol, timeframe, feed, error) VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (symbol, timeframe) DO UPDATE SET error = EXCLUDED.error",
        (symbol, timeframe, feed, message[:1000]),
    )


def _bar_rows(symbol: str, timeframe: str, feed: str, bars: list[dict]) -> list[tuple]:
    return [(symbol, timeframe, b["t"], b["o"], b["h"], b["l"], b["c"], b["v"], feed) for b in bars]


_INSERT_BAR = (
    "INSERT INTO bars (symbol, timeframe, ts, open, high, low, close, volume, feed) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
)
_UPSERT_BAR = _INSERT_BAR + (
    " ON CONFLICT (symbol, timeframe, ts) DO UPDATE SET open = EXCLUDED.open, high = EXCLUDED.high,"
    " low = EXCLUDED.low, close = EXCLUDED.close, volume = EXCLUDED.volume, feed = EXCLUDED.feed"
)


def _write_status(conn: psycopg.Connection, symbol: str, timeframe: str, feed: str) -> tuple[int, datetime | None]:
    """Recompute bar_status from the bars table (inside the caller's transaction)."""
    row = conn.execute(
        "SELECT min(ts) AS first_ts, max(ts) AS last_ts, count(*) AS n FROM bars WHERE symbol = %s AND timeframe = %s",
        (symbol, timeframe),
    ).fetchone()
    conn.execute(
        "INSERT INTO bar_status (symbol, timeframe, first_ts, last_ts, bars, feed, refreshed_at, error) "
        "VALUES (%s, %s, %s, %s, %s, %s, clock_timestamp(), NULL) "
        "ON CONFLICT (symbol, timeframe) DO UPDATE SET first_ts = EXCLUDED.first_ts, last_ts = EXCLUDED.last_ts,"
        " bars = EXCLUDED.bars, feed = EXCLUDED.feed, refreshed_at = EXCLUDED.refreshed_at, error = NULL",
        (symbol, timeframe, row["first_ts"], row["last_ts"], row["n"], feed),
    )
    return int(row["n"]), row["last_ts"]


def _lock_symbol(conn: psycopg.Connection, symbol: str, timeframe: str) -> None:
    """Serialise writers of one symbol (two workers asking for the same one)."""
    conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"bars:{symbol}:{timeframe}",))


def refresh_step(
    conn: psycopg.Connection,
    source: BarSource,
    market: str,
    symbol: str,
    now: datetime | None = None,
    after: datetime | None = None,
) -> dict[str, Any]:
    """One unit of refresh work for one symbol; never raises for a source or write error.

    Stocks: the whole history is downloaded and replaces the stored bars in one
    transaction; done is true. Crypto: one window of at most 120 days after the last
    stored bar (or after `after`, the cursor of the previous step, which lets a run walk
    over windows with no bars at all); done is true once the window reaches `now`.
    Returns {"symbol", "market", "added", "bars", "last_ts", "done", "error"} plus
    "from" and "cursor" (ISO times: where this step started and where the next one
    should) for progress.
    """
    spec = _market_spec(market)
    timeframe = spec["timeframe"]
    now = _utc(now) if now is not None else datetime.now(timezone.utc)
    delta = TIMEFRAME_DELTA[timeframe]
    out: dict[str, Any] = {"symbol": symbol, "market": market, "added": 0, "bars": 0, "last_ts": None,
                           "done": False, "error": None, "from": None, "cursor": None}
    feed: str | None = None
    try:
        feed = source.feed_for(symbol)
        prior = conn.execute(
            "SELECT count(*) AS n, max(ts) AS last_ts FROM bars WHERE symbol = %s AND timeframe = %s",
            (symbol, timeframe),
        ).fetchone()
        out["bars"], out["last_ts"] = int(prior["n"]), _iso(prior["last_ts"])
        market_start = _market_start(spec)

        if market == "stocks":
            window_start, window_end = market_start, now
        else:
            window_start = market_start if prior["last_ts"] is None else _utc(prior["last_ts"]) + delta
            if after is not None:
                window_start = max(window_start, _utc(after))
            window_end = min(window_start + CRYPTO_WINDOW, now)
        out["from"] = _iso(window_start)
        out["cursor"] = _iso(window_end)
        out["done"] = window_end >= now

        if window_start >= now:  # nothing new can have closed yet
            return out
        fetched = source.fetch(symbol, timeframe, window_start, window_end)
        bars = sorted(
            (b for b in fetched if _utc(b["t"]) >= window_start and _utc(b["t"]) + delta <= now),
            key=lambda b: b["t"],
        )
        for b in bars:
            b["t"] = _utc(b["t"])
        if market == "stocks" and not bars:
            raise RuntimeError(f"No daily bars came back for {symbol}")

        with conn.transaction():
            _lock_symbol(conn, symbol, timeframe)
            if market == "stocks":
                conn.execute("DELETE FROM bars WHERE symbol = %s AND timeframe = %s", (symbol, timeframe))
                sql = _INSERT_BAR
            else:
                sql = _UPSERT_BAR
            if bars:
                conn.cursor().executemany(sql, _bar_rows(symbol, timeframe, feed, bars))
            total, last_ts = _write_status(conn, symbol, timeframe, feed)
        out["added"] = max(0, total - out["bars"])
        out["bars"], out["last_ts"] = total, _iso(last_ts)
        return out
    except Exception as exc:  # noqa: BLE001 - a failing symbol is reported, not raised
        message = str(exc) or type(exc).__name__
        log.warning("data refresh %s failed: %s", symbol, message)
        out["error"] = message
        out["done"] = False
        try:
            _store_error(conn, symbol, timeframe, feed, message)
        except Exception:  # noqa: BLE001
            log.exception("could not store the refresh error for %s", symbol)
        return out


# ----------------------------------------------------------------- serving workers

_payload_cache: dict[str, tuple[Any, bytes, str]] = {}
_payload_lock = threading.Lock()


def _status_key(conn: psycopg.Connection, market: str) -> tuple[Any, ...]:
    """What the cached payload depends on: each symbol's refresh time and bar count."""
    spec = _market_spec(market)
    rows = conn.execute(
        "SELECT symbol, refreshed_at, bars FROM bar_status WHERE timeframe = %s AND symbol = ANY(%s) "
        "AND bars > 0 ORDER BY symbol",
        (spec["timeframe"], list(spec["symbols"])),
    ).fetchall()
    return tuple((r["symbol"], r["refreshed_at"], r["bars"]) for r in rows)


def _round(values: list[float], digits: int) -> list[float]:
    return [round(v, digits) for v in values]


def bars_payload(conn: psycopg.Connection, market: str) -> tuple[bytes, str]:
    """(gzip JSON bytes, ETag) of a market's bars in the wire format of
    fleet2/sim/marketdata.py. Raises LookupError when nothing is cached yet.

    The built bytes are cached per market and rebuilt only when a symbol's
    bar_status (refreshed_at, bars) changes, so repeated requests read one small table.
    """
    spec = _market_spec(market)
    key = _status_key(conn, market)
    if not key:
        raise LookupError("No price data yet: run a Data refresh job")
    with _payload_lock:
        cached = _payload_cache.get(market)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]

    timeframe = spec["timeframe"]
    symbols = [k[0] for k in key]
    series: dict[str, dict[str, list]] = {}
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(
            "SELECT symbol, extract(epoch FROM ts)::bigint, open, high, low, close, volume FROM bars "
            "WHERE timeframe = %s AND symbol = ANY(%s) ORDER BY symbol, ts",
            (timeframe, symbols),
        )
        for symbol, t, o, h, l, c, v in cur:
            s = series.setdefault(symbol, {"t": [], "o": [], "h": [], "l": [], "c": [], "v": []})
            s["t"].append(t)
            s["o"].append(o)
            s["h"].append(h)
            s["l"].append(l)
            s["c"].append(c)
            s["v"].append(v)
    meta = conn.execute(
        "SELECT max(refreshed_at) AS updated_at, mode() WITHIN GROUP (ORDER BY feed) AS feed FROM bar_status "
        "WHERE timeframe = %s AND symbol = ANY(%s) AND bars > 0",
        (timeframe, symbols),
    ).fetchone()
    doc = {
        "market": market,
        "timeframe": timeframe,
        "feed": meta["feed"] or ("crypto" if market == "crypto" else "iex"),
        "updated_at": _iso(meta["updated_at"]),
        "symbols": {
            sym: {"t": s["t"], "o": _round(s["o"], 6), "h": _round(s["h"], 6), "l": _round(s["l"], 6),
                  "c": _round(s["c"], 6), "v": _round(s["v"], 2)}
            for sym, s in series.items()
        },
    }
    body = gzip.compress(json.dumps(doc, separators=(",", ":")).encode(), compresslevel=6, mtime=0)
    etag = '"' + hashlib.sha256(body).hexdigest()[:24] + '"'
    with _payload_lock:
        _payload_cache[market] = (key, body, etag)
    return body, etag


def data_status(conn: psycopg.Connection) -> list[dict[str, Any]]:
    """One row per universe symbol: bars, first/last date, refreshed_at, error."""
    stored = {
        (r["symbol"], r["timeframe"]): r
        for r in conn.execute("SELECT * FROM bar_status").fetchall()
    }
    out: list[dict[str, Any]] = []
    for market, spec in universe.MARKETS.items():
        for symbol in spec["symbols"]:
            r = stored.get((symbol, spec["timeframe"]))
            out.append({
                "symbol": symbol,
                "market": market,
                "bars": int(r["bars"]) if r else 0,
                "first": _iso(r["first_ts"]) if r else None,
                "last": _iso(r["last_ts"]) if r else None,
                "refreshed_at": _iso(r["refreshed_at"]) if r else None,
                "error": r["error"] if r else None,
            })
    return out
