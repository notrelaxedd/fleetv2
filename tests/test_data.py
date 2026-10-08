"""Price data: bar sources, the rate limiter, refresh steps, the worker download route and
the Data refresh job. No test touches the network: the real Alpaca source is exercised
through a stand-in SDK client that records the request it was given."""
from __future__ import annotations

import gzip
import json
import threading
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

from coordinator import data
from coordinator.config import Config
from fleet2 import universe
from fleet2.common import http
from fleet2.sim import marketdata
from fleet2.sim.control import JobStopped
from fleet2.worker import data_job
from tests.conftest import enroll

UTC = timezone.utc
NOW = datetime(2026, 10, 8, 12, 30, tzinfo=UTC)  # a Thursday, half past noon UTC


def small_universe(start: str = "2026-01-01") -> dict[str, Any]:
    return {
        "stocks": {"symbols": ("SPY", "AAPL", "MSFT"), "timeframe": "1Day", "start": start,
                   "benchmark": "SPY", "bars_per_year": 252},
        "crypto": {"symbols": ("BTC/USD", "ETH/USD"), "timeframe": "1Hour", "start": start,
                   "benchmark": "BTC/USD", "bars_per_year": 24 * 365},
    }


@pytest.fixture(autouse=True)
def fresh_cache():
    data._payload_cache.clear()
    yield
    data._payload_cache.clear()


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(universe, "MARKETS", small_universe())
    return universe.MARKETS


@pytest.fixture
def source() -> data.FakeBarSource:
    return data.FakeBarSource(seed=3)


def status_row(conn, symbol):
    return conn.execute("SELECT * FROM bar_status WHERE symbol = %s", (symbol,)).fetchone()


def count_bars(conn, symbol) -> int:
    return conn.execute("SELECT count(*) AS n FROM bars WHERE symbol = %s", (symbol,)).fetchone()["n"]


def run_crypto(conn, source, symbol="BTC/USD", now=NOW, limit=20) -> list[dict]:
    steps, after = [], None
    while len(steps) < limit:
        r = data.refresh_step(conn, source, "crypto", symbol, now=now, after=after)
        steps.append(r)
        if r["done"] or r["error"]:
            break
        after = datetime.fromisoformat(r["cursor"].replace("Z", "+00:00"))
    return steps


class FailingSource:
    feed = "iex"

    def feed_for(self, symbol: str) -> str:
        return "crypto" if "/" in symbol else "iex"

    def fetch(self, symbol, timeframe, start, end):
        raise RuntimeError("Alpaca is having a bad day")


# ------------------------------------------------------------------ fake source


def test_fake_source_is_deterministic_and_consistent():
    a, b = data.FakeBarSource(seed=1), data.FakeBarSource(seed=1)
    start, end = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 6, 1, tzinfo=UTC)
    bars = a.fetch("AAPL", "1Day", start, end)
    assert bars == b.fetch("AAPL", "1Day", start, end)
    assert bars != data.FakeBarSource(seed=2).fetch("AAPL", "1Day", start, end)
    assert len(bars) > 90
    assert all(x["t"].weekday() < 5 for x in bars)
    assert [x["t"] for x in bars] == sorted({x["t"] for x in bars})
    for x in bars:
        assert x["l"] <= min(x["o"], x["c"]) and max(x["o"], x["c"]) <= x["h"]
        assert min(x["o"], x["h"], x["l"], x["c"]) > 0 and x["v"] > 0

    hourly = a.fetch("BTC/USD", "1Hour", start, start + timedelta(days=3))
    assert len(hourly) == 72 + 1  # both ends inclusive
    assert {hourly[i + 1]["t"] - hourly[i]["t"] for i in range(72)} == {timedelta(hours=1)}
    for x in hourly:
        assert x["l"] <= min(x["o"], x["c"]) and max(x["o"], x["c"]) <= x["h"] and x["l"] > 0
    # Windows are slices of one series: no seam where an incremental refresh joins them.
    first = a.fetch("BTC/USD", "1Hour", start, start + timedelta(hours=10))
    second = a.fetch("BTC/USD", "1Hour", start + timedelta(hours=11), start + timedelta(hours=20))
    assert first + second == a.fetch("BTC/USD", "1Hour", start, start + timedelta(hours=20))
    assert a.feed_for("AAPL") == "iex" and a.feed_for("BTC/USD") == "crypto"


def test_make_bar_source_follows_fake_broker():
    assert isinstance(data.make_bar_source(Config(fake_broker=True)), data.FakeBarSource)
    real = data.make_bar_source(Config())
    assert isinstance(real, data.AlpacaBarSource)
    assert real.feed == "iex" and real.feed_for("SPY") == "iex" and real.feed_for("BTC/USD") == "crypto"
    # SIP is paid: only the exact setting selects it, anything else stays on free IEX.
    assert data.AlpacaBarSource(Config(alpaca_data_feed="sip")).feed == "sip"
    assert data.AlpacaBarSource(Config(alpaca_data_feed="whatever")).feed == "iex"


# --------------------------------------------------------------------- limiter


def test_limiter_spaces_calls_with_an_injected_clock():
    clock = {"t": 100.0}
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock["t"] += seconds

    limiter = data.RateLimiter(150, clock=lambda: clock["t"], sleep=sleep)
    assert limiter.interval == pytest.approx(0.4)
    assert limiter.acquire() == 0
    assert limiter.acquire() == pytest.approx(0.4)
    assert limiter.acquire() == pytest.approx(0.4)
    assert slept == pytest.approx([0.4, 0.4])
    clock["t"] += 5  # idle for a while: no waiting, and no stored-up burst either
    assert limiter.acquire() == 0
    assert limiter.acquire() == pytest.approx(0.4)
    assert data.ALPACA_CALLS_PER_MINUTE == 150 and data.ALPACA_LIMITER.interval == pytest.approx(0.4)


def test_limiter_never_passes_the_cap_in_a_minute():
    clock = {"t": 0.0}
    times: list[float] = []

    def sleep(seconds: float) -> None:
        clock["t"] += seconds

    limiter = data.RateLimiter(150, clock=lambda: clock["t"], sleep=sleep)
    for _ in range(400):
        limiter.acquire()
        times.append(clock["t"])
    for t in times:
        assert sum(1 for u in times if t <= u < t + 60 - 1e-6) <= 150


def test_limiter_is_thread_safe():
    clock = {"t": 0.0}
    waits: list[float] = []
    lock = threading.Lock()
    limiter = data.RateLimiter(60, clock=lambda: clock["t"], sleep=lambda s: None)  # 1 s spacing

    def worker() -> None:
        w = limiter.acquire()
        with lock:
            waits.append(w)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(waits) == pytest.approx([float(i) for i in range(10)])  # ten distinct slots


# ----------------------------------------------------------------- alpaca source


class FakeSdkClient:
    """Stands in for the SDK client; returns a real BarSet built from raw API rows."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows, self.requests = rows, []

    def _answer(self, request):
        self.requests.append(request)
        return __import__("alpaca.data.models.bars", fromlist=["BarSet"]).BarSet({request.symbol_or_symbols: self.rows})

    get_stock_bars = _answer
    get_crypto_bars = _answer


RAW = [{"t": "2026-01-02T05:00:00Z", "o": 10.0, "h": 12.0, "l": 9.5, "c": 11.0, "v": 1000, "n": 5, "vw": 10.5},
       {"t": "2026-01-05T05:00:00Z", "o": 11.0, "h": 13.0, "l": 10.5, "c": 12.5, "v": 2000, "n": 7, "vw": 11.5}]


def test_alpaca_stock_request_uses_the_sdk_as_documented(monkeypatch):
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.timeframe import TimeFrame

    calls: list[str] = []
    limiter = data.RateLimiter(150, clock=lambda: 0.0, sleep=lambda s: calls.append("slept"))
    src = data.AlpacaBarSource(Config(alpaca_paper_key_id="k", alpaca_paper_secret="s"), limiter=limiter)
    client = FakeSdkClient(RAW)
    monkeypatch.setattr(src, "_stocks", lambda: client)
    start, end = datetime(2016, 1, 1, tzinfo=UTC), NOW
    bars = src.fetch("SPY", "1Day", start, end)

    (req,) = client.requests
    assert req.symbol_or_symbols == "SPY" and str(req.timeframe) == str(TimeFrame.Day) == "1Day"
    assert req.feed == DataFeed.IEX and req.adjustment == Adjustment.ALL
    naive = lambda d: d.replace(tzinfo=None)  # the SDK stores UTC times as naive
    assert req.start == naive(start) and req.end == naive(end)
    assert bars[0] == {"t": datetime(2026, 1, 2, 5, tzinfo=UTC), "o": 10.0, "h": 12.0, "l": 9.5, "c": 11.0, "v": 1000.0}
    assert len(bars) == 2
    src.fetch("SPY", "1Day", start, end)
    assert calls == ["slept"]  # the second call went through the same limiter and had to wait


def test_alpaca_sip_only_when_configured(monkeypatch):
    from alpaca.data.enums import DataFeed

    src = data.AlpacaBarSource(Config(alpaca_paper_key_id="k", alpaca_paper_secret="s", alpaca_data_feed="sip"))
    client = FakeSdkClient(RAW)
    monkeypatch.setattr(src, "_stocks", lambda: client)
    monkeypatch.setattr(src.limiter, "acquire", lambda: 0.0)
    src.fetch("SPY", "1Day", NOW - timedelta(days=5), NOW)
    assert client.requests[0].feed == DataFeed.SIP


def test_alpaca_crypto_request_needs_no_keys(monkeypatch):
    from alpaca.data.timeframe import TimeFrame

    src = data.AlpacaBarSource(Config())  # no keys at all
    client = FakeSdkClient(RAW)
    monkeypatch.setattr(src, "_crypto", lambda: client)
    monkeypatch.setattr(src.limiter, "acquire", lambda: 0.0)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    assert len(src.fetch("BTC/USD", "1Hour", start, start + timedelta(days=120))) == 2
    (req,) = client.requests
    assert req.symbol_or_symbols == "BTC/USD" and str(req.timeframe) == str(TimeFrame.Hour) == "1Hour"
    assert req.start == start.replace(tzinfo=None) and req.end == (start + timedelta(days=120)).replace(tzinfo=None)


def test_alpaca_stocks_without_keys_give_a_clear_error(conn, small):
    src = data.AlpacaBarSource(Config())
    with pytest.raises(RuntimeError, match="Add your Alpaca paper keys to .env on box1"):
        src.fetch("SPY", "1Day", datetime(2026, 1, 1, tzinfo=UTC), NOW)
    r = data.refresh_step(conn, src, "stocks", "SPY", now=NOW)
    assert r["error"] == "Add your Alpaca paper keys to .env on box1 (stock prices need them)"
    assert status_row(conn, "SPY")["error"] == r["error"]


# ------------------------------------------------------------------ refresh steps


def test_stock_step_stores_bars_and_replaces_on_the_second_call(conn, small, source):
    r = data.refresh_step(conn, source, "stocks", "SPY", now=NOW)
    assert r["error"] is None and r["done"] is True
    assert r["symbol"] == "SPY" and r["market"] == "stocks"
    n = count_bars(conn, "SPY")
    assert n > 150 and r["bars"] == n and r["added"] == n
    # 2026-10-08's bar ends after NOW, so the last stored bar is the 7th.
    assert r["last_ts"] == "2026-10-07T05:00:00Z"
    st = status_row(conn, "SPY")
    assert st["bars"] == n and st["feed"] == "iex" and st["error"] is None and st["refreshed_at"] is not None
    assert st["first_ts"] == datetime(2026, 1, 1, 5, tzinfo=UTC)
    assert st["last_ts"] == datetime(2026, 10, 7, 5, tzinfo=UTC)

    again = data.refresh_step(conn, source, "stocks", "SPY", now=NOW)
    assert again["done"] and again["error"] is None and again["added"] == 0 and again["bars"] == n
    assert count_bars(conn, "SPY") == n
    assert status_row(conn, "SPY")["bars"] == n


def test_stock_refresh_replaces_old_adjusted_prices(conn, small, source):
    data.refresh_step(conn, source, "stocks", "AAPL", now=NOW)
    before = conn.execute("SELECT ts, close FROM bars WHERE symbol = 'AAPL' ORDER BY ts").fetchall()

    class Split:  # the same history after a 2:1 split: every earlier price halves
        feed = "iex"
        feed_for = staticmethod(lambda s: "iex")

        def fetch(self, symbol, timeframe, start, end):
            return [{**b, "o": b["o"] / 2, "h": b["h"] / 2, "l": b["l"] / 2, "c": b["c"] / 2}
                    for b in source.fetch(symbol, timeframe, start, end)]

    r = data.refresh_step(conn, Split(), "stocks", "AAPL", now=NOW)
    after = conn.execute("SELECT ts, close FROM bars WHERE symbol = 'AAPL' ORDER BY ts").fetchall()
    assert r["added"] == 0 and len(after) == len(before)
    assert all(a["close"] == pytest.approx(b["close"] / 2) for a, b in zip(after, before))


def test_stock_step_one_call_even_for_a_long_history(conn, monkeypatch, source):
    monkeypatch.setattr(universe, "MARKETS", small_universe("2016-01-01"))
    calls = []
    real = source.fetch
    source.fetch = lambda *a: calls.append(a) or real(*a)
    r = data.refresh_step(conn, source, "stocks", "SPY", now=NOW)
    assert len(calls) == 1 and r["done"] and r["bars"] > 2500


def test_crypto_is_incremental_across_calls_and_reaches_done(conn, small, source):
    steps = run_crypto(conn, source)
    assert len(steps) == 3  # 280 days in windows of at most 120 days
    assert [s["done"] for s in steps] == [False, False, True]
    assert all(s["error"] is None for s in steps)
    assert all(s["added"] > 0 for s in steps)
    n = count_bars(conn, "BTC/USD")
    assert n == sum(s["added"] for s in steps) == steps[-1]["bars"]
    # Every hour from 2026-01-01 00:00 to the last closed bar (11:00 on the 8th), with no gaps or duplicates.
    hours = (datetime(2026, 10, 8, 11, tzinfo=UTC) - datetime(2026, 1, 1, tzinfo=UTC)) // timedelta(hours=1) + 1
    assert n == hours
    assert steps[-1]["last_ts"] == "2026-10-08T11:00:00Z"
    st = status_row(conn, "BTC/USD")
    assert st["bars"] == n and st["feed"] == "crypto" and st["first_ts"] == datetime(2026, 1, 1, tzinfo=UTC)

    # Up to date: nothing to add. Three hours later: exactly three new bars.
    r = data.refresh_step(conn, source, "crypto", "BTC/USD", now=NOW)
    assert r["done"] and r["added"] == 0 and r["bars"] == n
    r = data.refresh_step(conn, source, "crypto", "BTC/USD", now=NOW + timedelta(hours=3))
    assert r["done"] and r["added"] == 3 and r["last_ts"] == "2026-10-08T14:00:00Z"
    assert count_bars(conn, "BTC/USD") == n + 3


def test_crypto_without_the_cursor_still_resumes_from_the_last_stored_bar(conn, small, source):
    first = data.refresh_step(conn, source, "crypto", "ETH/USD", now=NOW)
    second = data.refresh_step(conn, source, "crypto", "ETH/USD", now=NOW)
    assert not first["done"] and second["from"] > first["from"]
    assert second["from"] == (datetime.fromisoformat(first["last_ts"].replace("Z", "+00:00")) + timedelta(hours=1)
                              ).isoformat().replace("+00:00", "Z")


def test_crypto_windows_with_no_bars_are_walked_with_the_cursor(conn, small):
    class LateListing:  # the coin only starts trading in June
        feed = "iex"
        feed_for = staticmethod(lambda s: "crypto")
        inner = data.FakeBarSource(1)

        def fetch(self, symbol, timeframe, start, end):
            return self.inner.fetch(symbol, timeframe, max(start, datetime(2026, 6, 1, tzinfo=UTC)), end) \
                if end > datetime(2026, 6, 1, tzinfo=UTC) else []

    steps = run_crypto(conn, LateListing(), "ETH/USD")
    assert steps[0]["added"] == 0 and not steps[0]["done"]  # an empty window does not stall the run
    assert steps[-1]["done"] and steps[-1]["bars"] > 0


def test_unfinished_bars_are_never_stored(conn, small, source):
    now = datetime(2026, 10, 8, 12, 59, 59, tzinfo=UTC)
    run_crypto(conn, source, now=now)
    assert conn.execute("SELECT max(ts) AS t FROM bars WHERE symbol = 'BTC/USD'").fetchone()["t"] == \
        datetime(2026, 10, 8, 11, tzinfo=UTC)  # the 12:00 bar closes at 13:00
    data.refresh_step(conn, source, "crypto", "BTC/USD", now=now + timedelta(seconds=1))
    assert conn.execute("SELECT max(ts) AS t FROM bars WHERE symbol = 'BTC/USD'").fetchone()["t"] == \
        datetime(2026, 10, 8, 12, tzinfo=UTC)

    data.refresh_step(conn, source, "stocks", "SPY", now=NOW)
    last = conn.execute("SELECT max(ts) AS t FROM bars WHERE symbol = 'SPY'").fetchone()["t"]
    assert last + timedelta(days=1) <= NOW  # today's daily bar is not there before the day is over
    data.refresh_step(conn, source, "stocks", "SPY", now=datetime(2026, 10, 9, 5, 0, 1, tzinfo=UTC))
    assert conn.execute("SELECT max(ts) AS t FROM bars WHERE symbol = 'SPY'").fetchone()["t"] == \
        datetime(2026, 10, 8, 5, tzinfo=UTC)


def test_source_errors_are_stored_and_returned_not_raised(conn, small, source):
    r = data.refresh_step(conn, FailingSource(), "stocks", "SPY", now=NOW)
    assert r["error"] == "Alpaca is having a bad day" and r["done"] is False and r["added"] == 0
    st = status_row(conn, "SPY")
    assert st["error"] == "Alpaca is having a bad day" and st["bars"] == 0 and st["refreshed_at"] is None

    r = data.refresh_step(conn, FailingSource(), "crypto", "BTC/USD", now=NOW)
    assert r["error"] and status_row(conn, "BTC/USD")["error"]

    # A later failure keeps the cached bars; the next success clears the error.
    data.refresh_step(conn, source, "stocks", "SPY", now=NOW)
    n = count_bars(conn, "SPY")
    assert status_row(conn, "SPY")["error"] is None
    data.refresh_step(conn, FailingSource(), "stocks", "SPY", now=NOW)
    assert count_bars(conn, "SPY") == n and status_row(conn, "SPY")["error"] == "Alpaca is having a bad day"
    assert status_row(conn, "SPY")["bars"] == n
    data.refresh_step(conn, source, "stocks", "SPY", now=NOW)
    assert status_row(conn, "SPY")["error"] is None


def test_an_empty_stock_answer_keeps_the_old_bars(conn, small, source):
    data.refresh_step(conn, source, "stocks", "SPY", now=NOW)
    n = count_bars(conn, "SPY")

    class Empty(FailingSource):
        def fetch(self, *a):
            return []

    r = data.refresh_step(conn, Empty(), "stocks", "SPY", now=NOW)
    assert "SPY" in r["error"] and count_bars(conn, "SPY") == n


def test_data_status_lists_every_symbol(conn, small, source):
    data.refresh_step(conn, source, "stocks", "SPY", now=NOW)
    data.refresh_step(conn, FailingSource(), "stocks", "AAPL", now=NOW)
    rows = {r["symbol"]: r for r in data.data_status(conn)}
    assert set(rows) == {"SPY", "AAPL", "MSFT", "BTC/USD", "ETH/USD"}
    assert rows["SPY"]["bars"] > 0 and rows["SPY"]["last"] == "2026-10-07T05:00:00Z" and rows["SPY"]["error"] is None
    assert rows["AAPL"]["bars"] == 0 and rows["AAPL"]["error"]
    assert rows["MSFT"] == {"symbol": "MSFT", "market": "stocks", "bars": 0, "first": None, "last": None,
                            "refreshed_at": None, "error": None}


# --------------------------------------------------------------------- the routes


def auth(worker: dict[str, Any]) -> dict[str, str]:
    return {"Authorization": "Bearer " + worker["worker_token"]}


@pytest.fixture
def worker(client, conn):
    return enroll(client, conn, "w1")


@pytest.fixture
def api(client, small, source):
    client.app.state.bar_source = source
    return client


def step(client, worker, market, symbol, **extra):
    return client.post("/api/v1/data/refresh-step", json={"market": market, "symbol": symbol, **extra},
                       headers=auth(worker))


def test_routes_need_a_worker_token(api, worker):
    assert api.post("/api/v1/data/refresh-step", json={"market": "stocks", "symbol": "SPY"}).status_code == 401
    assert api.get("/api/v1/data/bars?market=stocks").status_code == 401
    bad = {"Authorization": "Bearer nope"}
    assert api.get("/api/v1/data/bars?market=stocks", headers=bad).status_code == 401
    assert api.post("/api/v1/data/refresh-step", json={"market": "stocks", "symbol": "SPY"}, headers=bad).status_code == 401


def test_refresh_step_route_validates_and_refreshes(api, worker, conn):
    r = step(api, worker, "stocks", "SPY")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["symbol"] == "SPY" and body["done"] is True and body["error"] is None and body["added"] > 0
    assert count_bars(conn, "SPY") == body["bars"]

    bad = step(api, worker, "stocks", "ZZZZ")
    assert bad.status_code == 400 and "ZZZZ" in bad.json()["detail"]
    assert step(api, worker, "stocks", "BTC/USD").status_code == 400  # a crypto symbol is not a stock
    assert step(api, worker, "forex", "SPY").status_code == 400

    c = step(api, worker, "crypto", "BTC/USD")
    assert c.status_code == 200 and c.json()["done"] is False
    c2 = step(api, worker, "crypto", "BTC/USD", after=c.json()["cursor"])
    assert c2.json()["from"] == c.json()["cursor"] or c2.json()["from"] > c.json()["from"]


def test_refresh_step_route_reports_source_errors_as_a_reply(api, worker, conn):
    api.app.state.bar_source = FailingSource()
    r = step(api, worker, "stocks", "SPY")
    assert r.status_code == 200 and r.json()["error"] == "Alpaca is having a bad day"


def test_bars_404_before_any_data(api, worker):
    r = api.get("/api/v1/data/bars?market=stocks", headers=auth(worker))
    assert r.status_code == 404
    assert r.json()["detail"] == "No price data yet: run a Data refresh job"
    assert api.get("/api/v1/data/bars?market=nope", headers=auth(worker)).status_code == 400


def test_bars_payload_is_gzip_json_the_worker_can_read(api, worker, conn):
    for symbol in ("SPY", "AAPL"):
        assert step(api, worker, "stocks", symbol).json()["error"] is None
    r = api.get("/api/v1/data/bars?market=stocks", headers=auth(worker))
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/gzip" and r.headers["etag"].startswith('"')
    doc = json.loads(gzip.decompress(r.content))
    assert doc["market"] == "stocks" and doc["timeframe"] == "1Day" and doc["feed"] == "iex"
    assert doc["updated_at"].endswith("Z") and set(doc["symbols"]) == {"SPY", "AAPL"}  # MSFT has none yet
    spy = doc["symbols"]["SPY"]
    assert set(spy) == {"t", "o", "h", "l", "c", "v"} and all(isinstance(t, int) for t in spy["t"][:3])
    assert spy["t"] == sorted(spy["t"]) and len(spy["t"]) == len(spy["c"]) == count_bars(conn, "SPY")

    md = marketdata.from_payload(doc)
    assert set(md.symbols) == {"SPY", "AAPL"}
    assert md.close.shape == (2, md.n_bars) and md.times.dtype == np.int64
    assert not np.isnan(md.close).any()  # both symbols share the same fake calendar: fully aligned
    stored = conn.execute("SELECT close FROM bars WHERE symbol = 'SPY' ORDER BY ts").fetchall()
    assert md.close[md.row("SPY")] == pytest.approx([round(s["close"], 6) for s in stored])

    assert marketdata.from_payload(json.loads(gzip.decompress(r.content)), ("AAPL",)).symbols == ("AAPL",)


def test_crypto_payload_has_hourly_bars(api, worker):
    for _ in range(3):
        r = step(api, worker, "crypto", "BTC/USD").json()
        if r["done"]:
            break
    doc = json.loads(gzip.decompress(api.get("/api/v1/data/bars?market=crypto", headers=auth(worker)).content))
    assert doc["market"] == "crypto" and doc["timeframe"] == "1Hour" and doc["feed"] == "crypto"
    t = doc["symbols"]["BTC/USD"]["t"]
    assert {b - a for a, b in zip(t, t[1:])} == {3600}


def test_etag_and_304(api, worker, conn):
    step(api, worker, "stocks", "SPY")
    first = api.get("/api/v1/data/bars?market=stocks", headers=auth(worker))
    etag = first.headers["etag"]
    again = api.get("/api/v1/data/bars?market=stocks", headers={**auth(worker), "If-None-Match": etag})
    assert again.status_code == 304 and again.content == b"" and again.headers["etag"] == etag
    weak = api.get("/api/v1/data/bars?market=stocks", headers={**auth(worker), "If-None-Match": f'"x", W/{etag}'})
    assert weak.status_code == 304
    other = api.get("/api/v1/data/bars?market=stocks", headers={**auth(worker), "If-None-Match": '"stale"'})
    assert other.status_code == 200 and other.content == first.content

    step(api, worker, "stocks", "AAPL")  # new data: the old ETag no longer matches
    changed = api.get("/api/v1/data/bars?market=stocks", headers={**auth(worker), "If-None-Match": etag})
    assert changed.status_code == 200 and changed.headers["etag"] != etag


def test_payload_is_cached_until_bar_status_changes(conn, small, source, monkeypatch):
    data.refresh_step(conn, source, "stocks", "SPY", now=NOW)
    body, etag = data.bars_payload(conn, "stocks")
    def boom(*a, **k):
        raise AssertionError("payload rebuilt although nothing changed")

    monkeypatch.setattr(data.gzip, "compress", boom)
    assert data.bars_payload(conn, "stocks") == (body, etag)
    monkeypatch.undo()
    data.refresh_step(conn, source, "stocks", "SPY", now=NOW + timedelta(days=3))
    assert data.bars_payload(conn, "stocks")[1] != etag
    with pytest.raises(LookupError, match="No price data yet"):
        data.bars_payload(conn, "crypto")


def test_owner_status_route(api, worker):
    step(api, worker, "stocks", "SPY")
    rows = api.get("/api/data/status").json()
    assert {r["symbol"] for r in rows} == {"SPY", "AAPL", "MSFT", "BTC/USD", "ETH/USD"}
    spy = next(r for r in rows if r["symbol"] == "SPY")
    assert spy["bars"] > 0 and spy["last"] and spy["error"] is None


# ------------------------------------------------------------- the worker job


@pytest.fixture
def job_env(api, worker, monkeypatch):
    """run_data_refresh against the real routes: post_json is routed to the TestClient."""
    calls: list[dict] = []

    def post_json(url, body, token=None, timeout=4.0):
        calls.append(body)
        path = url.split("testserver", 1)[1]
        resp = api.post(path, json=body, headers={"Authorization": "Bearer " + token})
        if resp.status_code >= 400:
            raise http.HttpError(resp.status_code, resp.json().get("detail", ""), url)
        return resp.json()

    monkeypatch.setattr(http, "post_json", post_json)
    monkeypatch.setattr(data_job, "_sleep", lambda s: None)
    params = {"_context": {"host_url": "http://testserver", "worker_token": worker["worker_token"], "worker_id": worker["worker_id"]}}
    return params, calls


class Recorder:
    def __init__(self, stop_after: int | None = None) -> None:
        self.events: list[tuple[dict, float | None, str | None]] = []
        self.stop_after = stop_after

    def emit(self, checkpoint, progress, detail=None):
        self.events.append((checkpoint, progress, detail))

    def should_stop(self):
        return self.stop_after is not None and len(self.events) >= self.stop_after


def recent_universe(days: int = 200) -> dict[str, Any]:
    return small_universe((date.today() - timedelta(days=days)).isoformat())


def test_data_refresh_job_end_to_end(job_env, monkeypatch, conn):
    monkeypatch.setattr(universe, "MARKETS", recent_universe())
    params, calls = job_env
    rec = Recorder()
    result = data_job.run_data_refresh(params, None, rec.emit, rec.should_stop)

    assert result["errors"] == {}
    assert result["bars_added"] == sum(count_bars(conn, s) for s in ("SPY", "AAPL", "MSFT", "BTC/USD", "ETH/USD"))
    assert result["summary"] == f"Downloaded {result['bars_added']:,} new bars for 5 symbols"
    # Stock symbols took one call each; crypto took a call per 120-day window.
    assert [c["symbol"] for c in calls[:3]] == ["SPY", "AAPL", "MSFT"]
    assert any(c.get("after") for c in calls if c["symbol"] == "BTC/USD")

    progress = [p for _, p, _ in rec.events]
    assert progress == sorted(progress) and 0 < progress[0] < 1 and progress[-1] == pytest.approx(1.0)
    assert rec.events[0][2] == "Downloading SPY (1 of 5)"
    assert rec.events[0][0]["symbol_index"] == 0
    assert rec.events[-1][2] == "Downloading ETH/USD (5 of 5)" and rec.events[-1][0]["symbol_index"] == 4
    assert {d for _, _, d in rec.events} == {f"Downloading {s} ({i + 1} of 5)" for i, s in
                                             enumerate(["SPY", "AAPL", "MSFT", "BTC/USD", "ETH/USD"])}
    assert all(0 <= p <= 1 for p in progress)

    # A second run only adds what closed since (nothing, within the same hour).
    rec2 = Recorder()
    again = data_job.run_data_refresh(params, None, rec2.emit, rec2.should_stop)
    assert again["bars_added"] <= 1 and again["errors"] == {}


def test_data_refresh_job_markets_param_and_resume(job_env, monkeypatch):
    monkeypatch.setattr(universe, "MARKETS", recent_universe())
    params, calls = job_env
    rec = Recorder()
    result = data_job.run_data_refresh({**params, "markets": ["stocks"]}, None, rec.emit, rec.should_stop)
    assert {c["market"] for c in calls} == {"stocks"} and result["summary"].endswith("for 3 symbols")

    calls.clear()
    rec = Recorder()
    data_job.run_data_refresh({**params, "markets": ["stocks"]}, {"symbol_index": 2, "bars_added": 7, "errors": {}},
                              rec.emit, rec.should_stop)
    assert [c["symbol"] for c in calls] == ["MSFT"]  # resumes at the symbol that was in progress
    assert rec.events[0][2] == "Downloading MSFT (3 of 3)"
    with pytest.raises(ValueError, match="unknown market"):
        data_job.run_data_refresh({**params, "markets": ["forex"]}, None, rec.emit, rec.should_stop)


def test_data_refresh_job_stops_between_calls(job_env, monkeypatch):
    monkeypatch.setattr(universe, "MARKETS", recent_universe())
    params, calls = job_env
    rec = Recorder(stop_after=2)
    with pytest.raises(JobStopped):
        data_job.run_data_refresh(params, None, rec.emit, rec.should_stop)
    assert len(calls) == 2 and len(rec.events) == 2
    assert rec.events[-1][0]["symbol_index"] == 1  # the checkpoint a restart resumes from


def test_data_refresh_job_records_failed_symbols_and_carries_on(job_env, monkeypatch, api):
    monkeypatch.setattr(universe, "MARKETS", recent_universe())
    params, calls = job_env

    class Flaky(data.FakeBarSource):
        def fetch(self, symbol, *a):
            if symbol in ("AAPL", "ETH/USD"):
                raise RuntimeError(f"no luck with {symbol}")
            return super().fetch(symbol, *a)

    api.app.state.bar_source = Flaky(1)
    rec = Recorder()
    result = data_job.run_data_refresh(params, None, rec.emit, rec.should_stop)
    assert result["errors"] == {"AAPL": "no luck with AAPL", "ETH/USD": "no luck with ETH/USD"}
    assert result["summary"] == f"Downloaded {result['bars_added']:,} new bars for 5 symbols · 2 failed: AAPL, ETH/USD"
    assert result["bars_added"] > 0
    assert [e[2] for e in rec.events if "AAPL" in e[2]]  # the failed symbol still reported progress
    assert rec.events[-1][1] == pytest.approx(1.0)


def test_data_refresh_job_fails_with_the_first_error_when_everything_failed(job_env, monkeypatch, api):
    monkeypatch.setattr(universe, "MARKETS", recent_universe())
    params, _ = job_env
    api.app.state.bar_source = FailingSource()
    rec = Recorder()
    with pytest.raises(RuntimeError, match="^Alpaca is having a bad day$"):
        data_job.run_data_refresh(params, None, rec.emit, rec.should_stop)


def test_data_refresh_job_retries_5xx_and_lost_connections(monkeypatch):
    monkeypatch.setattr(universe, "MARKETS", {"stocks": {**small_universe()["stocks"], "symbols": ("SPY",)}})
    sleeps: list[float] = []
    monkeypatch.setattr(data_job, "_sleep", sleeps.append)
    attempts = {"n": 0}
    outcomes = [http.HttpError(503, "busy", "u"), http.HttpConnectionError("reset"),
                {"symbol": "SPY", "added": 5, "done": True, "error": None}]

    def post_json(url, body, token=None, timeout=4.0):
        out = outcomes[attempts["n"]]
        attempts["n"] += 1
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(http, "post_json", post_json)
    params = {"markets": ["stocks"], "_context": {"host_url": "http://h", "worker_token": "t"}}
    rec = Recorder()
    result = data_job.run_data_refresh(params, None, rec.emit, rec.should_stop)
    assert attempts["n"] == 3 and len(sleeps) == 2 and result["bars_added"] == 5 and result["errors"] == {}

    # A 4xx is not retried: it is that symbol's error.
    attempts["n"] = 0
    outcomes[:] = [http.HttpError(401, "invalid worker token", "u")]
    sleeps.clear()
    with pytest.raises(RuntimeError, match="invalid worker token"):
        data_job.run_data_refresh(params, None, rec.emit, rec.should_stop)
    assert attempts["n"] == 1 and sleeps == []

    # Three failed attempts in a row end that symbol's tries.
    attempts["n"] = 0
    outcomes[:] = [http.HttpError(502, "bad gateway", "u")] * 3
    with pytest.raises(RuntimeError, match="bad gateway"):
        data_job.run_data_refresh(params, None, rec.emit, rec.should_stop)
    assert attempts["n"] == 3


def test_data_refresh_is_registered_and_job_writes_nothing_to_disk():
    from fleet2.worker import jobs

    assert jobs.JOBS["data_refresh"] is jobs.run_data_refresh
    source = open(data_job.__file__, encoding="utf-8").read()
    assert "open(" not in source and "print(" not in source and "logging" not in source
