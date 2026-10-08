"""Live 1-minute prices for futures models that are trading, on the coordinator.

Each place a model trades has its own prices, so the model decides on what is traded:
- "alpaca": SPY and QQQ from Alpaca's free IEX feed, scaled to index points (the
  proxy source of coordinator.futures_data), for Alpaca paper trading;
- "topstepx": the real MES and MNQ contracts from TopstepX, for Topstep;
- "synthetic": made-up prices in demo mode (FLEET_FAKE_BROKER=1).

The latest LIVE_DAYS calendar days are kept in the live_bars table (models look back
up to about 30 trading days), refreshed every REFRESH_S seconds while a book of that
place is trading and the session is open. Only closed regular-session minutes are kept.
Workers get them in the same .npz format as the research prices (futures_data), with
an ETag (GET /api/v1/data/futures-live?source=...).
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import psycopg
from psycopg.rows import tuple_row

from fleet2 import universe
from fleet2.sim import cme_session

log = logging.getLogger(__name__)

LIVE_DAYS = 60
KEEP_DAYS = 70
WINDOW = timedelta(days=10)
REFRESH_S = 20.0
SOURCES = ("alpaca", "topstepx", "synthetic")
FEED = {"alpaca": "proxy", "topstepx": "topstepx", "synthetic": "synthetic"}


def source_for(venue: str, fake: bool) -> str:
    if fake:
        return "synthetic"
    return "alpaca" if venue == "alpaca_paper" else "topstepx"


def refresh(conn: psycopg.Connection, name: str, source: Any, symbol: str, now: datetime | None = None) -> int:
    """Bring one symbol's live prices up to date; returns bars added. Raises on a source error."""
    now = now or datetime.now(timezone.utc)
    last = conn.execute("SELECT max(ts) AS t FROM live_bars WHERE source = %s AND symbol = %s",
                        (name, symbol)).fetchone()["t"]
    start = max(last + timedelta(minutes=1), now - timedelta(days=LIVE_DAYS)) if last else now - timedelta(days=LIVE_DAYS)
    added = 0
    while start < now:
        end = min(start + WINDOW, now)
        bars = [b for b in source.fetch(symbol, start, end) if int(b["t"]) + 60 <= now.timestamp()]
        if bars:
            ok = cme_session.in_session(np.asarray([int(b["t"]) for b in bars], dtype=np.int64))
            rows = [(name, symbol, int(b["t"]), b["o"], b["h"], b["l"], b["c"], b["v"], int(b.get("iid") or 0))
                    for b, good in zip(bars, ok) if good]
            if rows:
                conn.cursor().executemany(
                    "INSERT INTO live_bars (source, symbol, ts, open, high, low, close, volume, instrument_id) "
                    "VALUES (%s, %s, to_timestamp(%s), %s, %s, %s, %s, %s, %s) ON CONFLICT (source, symbol, ts) DO UPDATE "
                    "SET open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, close = EXCLUDED.close, "
                    "volume = EXCLUDED.volume, instrument_id = EXCLUDED.instrument_id", rows)
                added += len(rows)
        start = end
    conn.execute("DELETE FROM live_bars WHERE source = %s AND ts < %s", (name, now - timedelta(days=KEEP_DAYS)))
    return added


def session_window(now: datetime, margin_minutes: int = 5) -> bool:
    """True from a few minutes before today's open to a few minutes after its close."""
    day = now.astimezone(cme_session.CHICAGO).date()
    s = cme_session.session(day)
    t = now.timestamp()
    return s is not None and s[0] - margin_minutes * 60 <= t <= s[1] + margin_minutes * 60


def latest(conn: psycopg.Connection, name: str, symbol: str) -> dict[str, Any] | None:
    """The newest live bar of a symbol (None when there is none)."""
    return conn.execute("SELECT ts, open, high, low, close, instrument_id FROM live_bars WHERE source = %s AND "
                        "symbol = %s ORDER BY ts DESC LIMIT 1", (name, symbol)).fetchone()


def share_price(conn: psycopg.Connection, stand_in: str) -> float | None:
    """The latest SPY or QQQ price implied by the live Alpaca (or synthetic) prices."""
    for symbol, spec in universe.CONTRACTS.items():
        if spec["proxy"] == stand_in:
            for name in ("alpaca", "synthetic"):
                row = latest(conn, name, symbol)
                if row is not None:
                    return float(row["close"]) / spec["proxy_scale"]
    return None


_cache: dict[str, tuple[Any, bytes, str]] = {}
_lock = threading.Lock()


def payload(conn: psycopg.Connection, name: str) -> tuple[bytes, str]:
    """(.npz bytes, ETag) of the live prices of one source. LookupError when none yet."""
    if name not in SOURCES:
        raise ValueError(f"unknown live source {name!r}")
    key = conn.execute("SELECT count(*) AS n, max(ts) AS t FROM live_bars WHERE source = %s", (name,)).fetchone()
    if not key["n"]:
        raise LookupError("No live futures prices yet")
    stamp = (key["n"], key["t"])
    with _lock:
        cached = _cache.get(name)
        if cached is not None and cached[0] == stamp:
            return cached[1], cached[2]
    arrays: dict[str, np.ndarray] = {}
    symbols = []
    for symbol in universe.FUTURES["symbols"]:
        with conn.cursor(row_factory=tuple_row) as cur:
            rows = cur.execute("SELECT extract(epoch FROM ts)::bigint, open, high, low, close, volume, instrument_id "
                               "FROM live_bars WHERE source = %s AND symbol = %s ORDER BY ts", (name, symbol)).fetchall()
        if not rows:
            continue
        block = np.asarray(rows, dtype=np.float64)
        symbols.append(symbol)
        for j, k in enumerate(("t", "o", "h", "l", "c", "v", "iid")):
            arrays[f"{symbol}_{k}"] = block[:, j].astype(np.int64) if k in ("t", "iid") else block[:, j]
    meta = {"market": "futures", "timeframe": "1Min", "feed": FEED[name], "through": "live", "symbols": symbols,
            "updated_at": key["t"].isoformat()}
    buf = io.BytesIO()
    np.savez_compressed(buf, meta=np.asarray(json.dumps(meta)), **arrays)
    body = buf.getvalue()
    etag = '"' + hashlib.sha256(body).hexdigest()[:24] + '"'
    with _lock:
        _cache[name] = (stamp, body, etag)
    return body, etag
