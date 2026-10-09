"""Futures prices: the CME session calendar, the three sources (Databento with its cost
check, the SPY/QQQ proxy, synthetic), refresh steps, the fixed periods, the .npz payload
with its ETag and period cut-off, and the worker's minute grid and decision bars.
No test touches the network."""
from __future__ import annotations

import io
import json
import urllib.error
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from coordinator import futures_data as fd
from coordinator.config import Config
from fleet2.sim import cme_session as cme
from fleet2.sim import futures_data as wfd
from fleet2.worker import data_job
from tests.conftest import enroll
from tests.test_data import Recorder

UTC = timezone.utc
NOW = datetime(2019, 6, 21, 23, 0, tzinfo=UTC)  # a Friday evening, about 34 trading days after the start


@pytest.fixture(autouse=True)
def fresh_cache():
    fd._payload_cache.clear()
    yield
    fd._payload_cache.clear()


def ct(d: date, hh: int, mm: int) -> int:
    return int(datetime(d.year, d.month, d.day, hh, mm, tzinfo=cme.CHICAGO).timestamp())


# ------------------------------------------------------------------ sessions


def test_holidays_and_half_days_match_the_exchange_calendar():
    assert date(2024, 3, 29) in cme.holidays(2024)          # Good Friday
    assert date(2022, 6, 20) in cme.holidays(2022)          # Juneteenth on a Sunday, taken on Monday
    assert date(2021, 12, 24) in cme.holidays(2021)         # Christmas on a Saturday, taken on Friday
    assert date(2022, 1, 1) not in cme.holidays(2022) and date(2021, 12, 31) not in cme.holidays(2021)
    assert date(2025, 1, 9) in cme.holidays(2025)           # national day of mourning
    assert date(2021, 6, 18) not in cme.holidays(2021)      # Juneteenth only from 2022
    assert cme.half_days(2024) == {date(2024, 7, 3), date(2024, 11, 29), date(2024, 12, 24)}
    assert cme.half_days(2020) == {date(2020, 11, 27), date(2020, 12, 24)}  # July 3 2020 was the holiday
    assert not cme.is_trading_day(date(2024, 3, 30)) and cme.is_trading_day(date(2024, 4, 1))


def test_session_hours_in_chicago_time_across_clock_changes():
    for d in (date(2024, 1, 10), date(2024, 7, 10)):  # winter and summer time
        open_, close = cme.session(d)
        assert open_ == ct(d, 8, 30) and close == ct(d, 15, 0) and cme.session_minutes(d) == 390
    half = date(2024, 11, 29)
    assert cme.session(half) == (ct(half, 8, 30), ct(half, 12, 0)) and cme.session_minutes(half) == 210
    assert cme.session(date(2024, 12, 25)) is None and cme.session_minutes(date(2024, 12, 25)) == 0


def test_trading_day_of_a_bar():
    d = date(2024, 3, 12)
    assert cme.trading_day(ct(d, 8, 30)) == d
    assert cme.trading_day(ct(d, 16, 59)) == d
    assert cme.trading_day(ct(d, 17, 0)) == d + timedelta(days=1)  # the evening session starts the next day
    epochs = np.asarray([ct(date(2024, 3, 8), 14, 59), ct(date(2024, 3, 11), 8, 30), ct(date(2024, 11, 4), 0, 30)])
    assert cme.chicago_dates(epochs).tolist() == [20240308, 20240311, 20241104]
    assert cme.in_session(np.asarray([ct(d, 8, 29), ct(d, 8, 30), ct(d, 14, 59), ct(d, 15, 0)])).tolist() == \
        [False, True, True, False]


# ------------------------------------------------------------------ Databento


class FakeDatabento:
    """Stands in for urllib's urlopen: records every request, answers cost and CSV."""

    def __init__(self, cost: float = 1.0, rows: list[str] | None = None, status: int = 200,
                 available_end: str | None = None) -> None:
        self.cost, self.rows, self.status = cost, rows or [], status
        self.available_end = available_end  # Databento's history ends here (e.g. "2026-10-09T02:00:00Z")
        self.busy: list[int] = []  # answer these statuses first, one per request (e.g. [504, 504])
        self.requests: list[tuple[str, dict[str, list[str]], dict[str, str]]] = []

    def __call__(self, req, timeout=None):
        from urllib.parse import parse_qs

        fields = parse_qs(req.data.decode())
        self.requests.append((req.full_url, fields, dict(req.header_items())))
        if self.status != 200:
            raise urllib.error.HTTPError(req.full_url, self.status, "no", {}, io.BytesIO(b'{"detail":"bad key"}'))
        if self.busy:
            page = b"<html><body><h1>504 Gateway Time-out</h1> The server didn't respond in time. </body></html>"
            raise urllib.error.HTTPError(req.full_url, self.busy.pop(0), "busy", {}, io.BytesIO(page))
        if self.available_end and fields["end"][0] > self.available_end:
            # The answer Databento gave on box1, word for word apart from the times.
            shown = self.available_end.replace("T", " ").replace("Z", "+00:00")
            asked = fields["end"][0].replace("T", " ").replace("Z", ".046092+00:00")
            body = json.dumps({"detail": {
                "case": "data_end_after_available_end",
                "message": f"The dataset GLBX.MDP3 has data available up to '{shown}'. The `end` in the query "
                           f"('{asked}') is after the available range. Try requesting with an earlier `end`.",
                "status_code": 422, "docs": "https://databento.com/docs/api-reference-historical/basics/datasets",
                "payload": None}}).encode()
            raise urllib.error.HTTPError(req.full_url, 422, "Unprocessable", {}, io.BytesIO(body))
        if req.full_url.endswith("metadata.get_cost"):
            body = json.dumps(self.cost).encode()
        else:
            body = ("ts_event,rtype,publisher_id,instrument_id,open,high,low,close,volume\n" + "\n".join(self.rows)).encode()
        return _Resp(body)


class _Resp:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return None


def databento(cost=1.0, rows=None, status=200, available_end=None) -> tuple[fd.DatabentoSource, FakeDatabento]:
    fake = FakeDatabento(cost, rows, status, available_end)
    fake.pauses = []
    return fd.DatabentoSource("db-test-key", opener=fake, sleep=fake.pauses.append), fake


def csv_row(epoch: int, iid: int, o: float, h: float, l: float, c: float, v: int = 10) -> str:
    return f"{epoch * 1_000_000_000},33,1,{iid},{o:.9f},{h:.9f},{l:.9f},{c:.9f},{v}"


def test_databento_requests_carry_the_key_dataset_schema_and_front_contract():
    d = date(2024, 3, 12)
    src, fake = databento(cost=0.42, rows=[csv_row(ct(d, 8, 30), 42140, 5100.25, 5101.0, 5099.5, 5100.75)])
    start, end = datetime(2024, 3, 1, tzinfo=UTC), datetime(2024, 4, 1, tzinfo=UTC)
    assert src.cost("MES", start, end) == pytest.approx(0.42)
    bars = src.fetch("MES", start, end)
    assert bars == [{"t": ct(d, 8, 30), "o": 5100.25, "h": 5101.0, "l": 5099.5, "c": 5100.75, "v": 10.0, "iid": 42140}]
    (cost_url, cost_fields, headers), (range_url, range_fields, _) = fake.requests
    assert cost_url == "https://hist.databento.com/v0/metadata.get_cost"
    assert range_url == "https://hist.databento.com/v0/timeseries.get_range"
    assert headers["Authorization"] == "Basic ZGItdGVzdC1rZXk6"  # base64 of "db-test-key:"
    for fields in (cost_fields, range_fields):
        assert fields["dataset"] == ["GLBX.MDP3"] and fields["schema"] == ["ohlcv-1m"]
        assert fields["symbols"] == ["MES.v.0"] and fields["stype_in"] == ["continuous"]
        assert fields["start"] == ["2024-03-01T00:00:00Z"] and fields["end"] == ["2024-04-01T00:00:00Z"]
    assert range_fields["encoding"] == ["csv"] and range_fields["compression"] == ["none"]


def test_databento_csv_in_either_price_format():
    t = ct(date(2024, 3, 12), 9, 0)
    raw = ("ts_event,rtype,publisher_id,instrument_id,open,high,low,close,volume\n"
           f"{t * 10**9},33,1,7,5100250000000,5101000000000,5099500000000,5100750000000,3\n")
    assert fd.parse_databento_csv(raw)[0]["o"] == 5100.25
    pretty = ("ts_event,rtype,publisher_id,instrument_id,open,high,low,close,volume\n"
              "2024-03-12T14:00:00.000000000Z,33,1,7,5100.25,5101.0,5099.5,5100.75,3\n")
    assert fd.parse_databento_csv(pretty)[0]["t"] == t


def test_databento_refuses_a_bad_key_in_plain_words():
    src, _ = databento(status=401)
    with pytest.raises(fd.DatabentoError, match="DATABENTO_API_KEY"):
        src.cost("MES", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC))


def test_databento_history_ends_a_few_minutes_behind_so_the_end_is_moved_back():
    d = date(2026, 10, 8)
    src, fake = databento(cost=0.3, rows=[csv_row(ct(d, 8, 30), 7, 6700.0, 6701.0, 6699.0, 6700.5)],
                          available_end="2026-10-09T02:00:00Z")
    start, now = datetime(2026, 10, 1, tzinfo=UTC), datetime(2026, 10, 9, 2, 4, 38, tzinfo=UTC)
    assert src.cost("MES", start, now) == pytest.approx(0.3)
    assert len(src.fetch("MES", start, now)) == 1
    ends = [f["end"][0] for _, f, _ in fake.requests]
    assert ends == ["2026-10-09T02:04:38Z", "2026-10-09T02:00:00Z"] * 2
    # Nothing after the start yet: no request at all for the empty part, no error.
    late = datetime(2026, 10, 9, 2, 1, tzinfo=UTC)
    assert src.cost("MES", late, now) == 0.0 and src.fetch("MES", late, now) == []


def test_a_refresh_right_after_the_market_works_although_databento_is_behind(conn):
    d = date(2026, 10, 8)
    src, _ = databento(cost=0.5, rows=[csv_row(ct(d, 8, 30), 7, 6700.0, 6701.0, 6699.0, 6700.5)],
                       available_end="2026-10-09T02:00:00Z")
    now = datetime(2026, 10, 9, 2, 4, 38, tzinfo=UTC)
    conn.execute("INSERT INTO bars (symbol, timeframe, ts, open, high, low, close, volume, feed, instrument_id) "
                 "VALUES ('MES', '1Min', '2026-10-07 19:59:00+00', 1, 1, 1, 1, 1, 'databento', 7)")
    r = fd.refresh_step(conn, src, "MES", cap_usd=10.0, now=now)
    assert r["error"] is None and r["added"] == 1 and r["done"]


def test_a_busy_databento_is_tried_again_before_giving_up():
    start, end = datetime(2024, 3, 1, tzinfo=UTC), datetime(2024, 4, 1, tzinfo=UTC)
    src, fake = databento(cost=0.42)
    fake.busy = [504, 502]
    assert src.cost("MES", start, end) == pytest.approx(0.42)
    assert len(fake.requests) == 3 and fake.pauses == [5.0, 20.0]

    src, fake = databento()
    fake.busy = [504, 504, 504]
    with pytest.raises(fd.DatabentoError, match="busy and answered 504 .3 tries.*Press Load futures prices again"):
        src.fetch("MES", start, end)
    assert len(fake.requests) == 3


def test_only_a_symbols_first_step_asks_what_the_download_costs(conn):
    d = date(2019, 5, 6)
    src, fake = databento(cost=0.5, rows=[csv_row(ct(d, 8, 30), 11, 2900, 2901, 2899, 2900.5)])
    first = fd.refresh_step(conn, src, "MES", cap_usd=10.0, now=NOW)
    assert first["error"] is None and first["cost_usd"] == 1.0
    assert [u.rsplit("/", 1)[1] for u, _, _ in fake.requests] == ["metadata.get_cost"] * 2 + ["timeseries.get_range"]
    fake.requests.clear()
    after = datetime.fromisoformat(first["cursor"].replace("Z", "+00:00"))
    later = fd.refresh_step(conn, src, "MES", cap_usd=10.0, now=NOW, after=after)
    assert later["error"] is None and later["cost_usd"] == 0.0
    assert [u.rsplit("/", 1)[1] for u, _, _ in fake.requests] == ["timeseries.get_range"]


def test_the_available_end_is_read_in_every_time_format():
    assert fd._available_end("has data available up to '2026-10-09 02:00:00+00:00'.") == datetime(2026, 10, 9, 2, tzinfo=UTC)
    assert fd._available_end("has data available up to '2026-10-09T02:00:00.123456789Z'") == datetime(2026, 10, 9, 2, 0, 0, 123456, tzinfo=UTC)
    assert fd._available_end("something else") is None


def test_a_download_above_the_cap_is_refused_before_anything_is_fetched(conn):
    src, fake = databento(cost=6.0)  # each symbol: $6, so $12 to bring both up to date
    r = fd.refresh_step(conn, src, "MES", cap_usd=10.0, now=NOW)
    assert r["error"].startswith("Databento says bringing the futures prices up to date costs $12.00")
    assert "max_download_usd in config/topstep.toml" in r["error"] and r["cost_usd"] == 12.0
    assert [u for u, _, _ in fake.requests] == ["https://hist.databento.com/v0/metadata.get_cost"] * 2
    assert {f["symbols"][0] for _, f, _ in fake.requests} == {"MES.v.0", "MNQ.v.0"}
    assert conn.execute("SELECT count(*) AS n FROM bars").fetchone()["n"] == 0
    assert conn.execute("SELECT error FROM bar_status WHERE symbol = 'MES'").fetchone()["error"].startswith("Databento says")


def test_a_databento_window_stores_session_bars_with_their_contract(conn):
    d = date(2019, 5, 6)
    rows = [csv_row(ct(d, 8, 29), 1, 2900, 2901, 2899, 2900),          # before the open: dropped
            csv_row(ct(d, 8, 30), 11, 2900, 2901, 2899, 2900.5),
            csv_row(ct(d, 14, 59), 11, 2905, 2906, 2904, 2905.25),
            csv_row(ct(d, 15, 0), 11, 2905, 2906, 2904, 2905.25)]       # after the close: dropped
    src, fake = databento(cost=0.5, rows=rows)
    r = fd.refresh_step(conn, src, "MES", cap_usd=10.0, now=NOW)
    assert r["error"] is None and r["added"] == 2 and r["feed"] == "databento" and r["cost_usd"] == 1.0
    stored = conn.execute("SELECT ts, close, instrument_id, feed FROM bars WHERE symbol = 'MES' ORDER BY ts").fetchall()
    assert [s["instrument_id"] for s in stored] == [11, 11] and {s["feed"] for s in stored} == {"databento"}
    assert stored[0]["ts"] == datetime(2019, 5, 6, 13, 30, tzinfo=UTC)
    status = conn.execute("SELECT * FROM bar_status WHERE symbol = 'MES' AND timeframe = '1Min'").fetchone()
    assert status["bars"] == 2 and status["feed"] == "databento"


def test_no_key_means_the_proxy_and_a_demo_means_synthetic():
    assert isinstance(fd.make_futures_source(Config()), fd.ProxySource)
    assert isinstance(fd.make_futures_source(Config(databento_api_key="k")), fd.DatabentoSource)
    assert isinstance(fd.make_futures_source(Config(fake_broker=True, databento_api_key="k")), fd.FakeFuturesSource)
    assert "DATABENTO_API_KEY" not in repr(Config(databento_api_key="secret")) and "secret" not in repr(
        Config(databento_api_key="secret"))
    assert Config.from_env({"DATABENTO_API_KEY": " abc "}).databento_api_key == "abc"


def test_the_proxy_scales_spy_and_qqq_to_index_points_on_the_tick_grid():
    seen = {}

    class Client:
        def get_stock_bars(self, request):
            seen["request"] = request
            bar = SimpleNamespace(timestamp=datetime(2024, 3, 12, 13, 30, tzinfo=UTC), open=510.03, high=510.11,
                                  low=509.98, close=510.06, volume=1200)
            return SimpleNamespace(data={request.symbol_or_symbols: [bar]})

    src = fd.ProxySource(Config(alpaca_paper_key_id="k", alpaca_paper_secret="s"))
    src._client = Client()
    src.limiter = SimpleNamespace(acquire=lambda: 0.0)
    (bar,) = src.fetch("MES", datetime(2024, 3, 12, tzinfo=UTC), datetime(2024, 3, 13, tzinfo=UTC))
    assert seen["request"].symbol_or_symbols == "SPY" and str(seen["request"].timeframe) == "1Min"
    assert bar == {"t": int(datetime(2024, 3, 12, 13, 30, tzinfo=UTC).timestamp()), "o": 5100.25, "h": 5101.0,
                   "l": 5099.75, "c": 5100.5, "v": 1200.0, "iid": 0}
    src._client = Client()
    assert src.fetch("MNQ", datetime(2024, 3, 12, tzinfo=UTC), datetime(2024, 3, 13, tzinfo=UTC))[0]["o"] == \
        round(510.03 * 41 / 0.25) * 0.25
    with pytest.raises(RuntimeError, match="DATABENTO_API_KEY"):
        fd.ProxySource(Config()).fetch("MES", datetime(2024, 3, 12, tzinfo=UTC), datetime(2024, 3, 13, tzinfo=UTC))


# ------------------------------------------------------------------ refresh with synthetic prices


def fill(conn, symbols=("MES", "MNQ"), now=NOW, source=None) -> fd.FakeFuturesSource:
    source = source or fd.FakeFuturesSource(seed=3)
    for symbol in symbols:
        after = None
        for _ in range(20):
            r = fd.refresh_step(conn, source, symbol, 0.0, now=now, after=after)
            assert r["error"] is None, r
            if r["done"]:
                break
            after = datetime.fromisoformat(r["cursor"].replace("Z", "+00:00"))
    return source


def test_synthetic_refresh_walks_month_windows_and_keeps_only_the_session(conn):
    source = fill(conn, ("MES",))
    windows = [(s, e) for sym, s, e in source.calls]
    assert all(e - s <= timedelta(days=31) for s, e in windows) and len(windows) == 2
    days = conn.execute("SELECT (ts AT TIME ZONE 'America/Chicago')::date AS d, count(*) AS n, "
                        "min((ts AT TIME ZONE 'America/Chicago')::time) AS a, max((ts AT TIME ZONE 'America/Chicago')::time) AS b "
                        "FROM bars WHERE symbol = 'MES' GROUP BY 1 ORDER BY 1").fetchall()
    assert days[0]["d"] == date(2019, 5, 6) and days[-1]["d"] == date(2019, 6, 21)
    assert {r["n"] for r in days} == {390} and date(2019, 5, 27) not in {r["d"] for r in days}  # Memorial Day
    assert {str(r["a"]) for r in days} == {"08:30:00"} and {str(r["b"]) for r in days} == {"14:59:00"}
    again = fd.refresh_step(conn, source, "MES", 0.0, now=NOW)
    assert again["added"] == 0 and again["done"]


def test_a_new_source_replaces_a_symbols_bars_instead_of_mixing(conn):
    fill(conn, ("MES",), now=datetime(2019, 5, 10, 23, 0, tzinfo=UTC))
    assert conn.execute("SELECT DISTINCT feed FROM bars WHERE symbol = 'MES'").fetchone()["feed"] == "synthetic"
    d = date(2019, 5, 6)
    src, _ = databento(cost=0.1, rows=[csv_row(ct(d, 8, 30), 5, 2900, 2901, 2899, 2900)])
    r = fd.refresh_step(conn, src, "MES", 10.0, now=datetime(2019, 5, 10, 23, 0, tzinfo=UTC))
    assert r["error"] is None and r["bars"] == 1 and r["from"] == "2019-05-06T13:30:00Z"
    assert [x["feed"] for x in conn.execute("SELECT DISTINCT feed FROM bars WHERE symbol = 'MES'")] == ["databento"]


# ------------------------------------------------------------------ periods and the payload


@pytest.fixture
def few_days(monkeypatch):
    monkeypatch.setattr(fd, "MIN_DAYS_TO_FIX_PERIODS", 20)


def test_periods_are_split_60_25_15_and_never_move(conn, few_days):
    with pytest.raises(Exception, match="Load futures prices first"):
        fd.futures_periods(conn)
    fill(conn)
    periods = fd.futures_periods(conn)
    days = cme.trading_days(date(2019, 5, 6), date(2019, 6, 21))
    assert len(days) == 34
    assert periods["train_end"] == days[19].isoformat()  # 20 of 34 days (60%)
    assert periods["held_out_end"] == days[28].isoformat()  # 29 of 34 (85%)
    assert periods["lockbox_end"] == days[-1].isoformat() and periods["fixed_with"] == "synthetic"
    fill(conn, now=datetime(2019, 8, 30, 23, 0, tzinfo=UTC))
    assert fd.futures_periods(conn) == periods  # more prices never move the periods


def futures_api(client, conn, source=None):
    client.app.state.futures_source = source or fd.FakeFuturesSource(seed=3)
    return enroll(client, conn, "w1")


def auth(worker: dict[str, Any]) -> dict[str, str]:
    return {"Authorization": "Bearer " + worker["worker_token"]}


def test_payload_serves_each_period_up_to_its_end_only(client, conn, few_days):
    worker = futures_api(client, conn)
    fill(conn)
    periods = fd.futures_periods(conn)
    for through, end_key in (("train", "train_end"), ("held_out", "held_out_end")):
        r = client.get(f"/api/v1/data/futures-bars?through={through}", headers=auth(worker))
        assert r.status_code == 200 and r.headers["content-type"] == "application/octet-stream"
        data = wfd.from_npz(r.content)
        assert data.meta["through"] == through and data.feed == "synthetic"
        assert cme.as_date(int(data.days[-1])).isoformat() == periods[end_key]
        assert data.symbols == ("MES", "MNQ") and data.instrument.dtype == np.int64
    again = client.get("/api/v1/data/futures-bars?through=train",
                       headers={**auth(worker), "If-None-Match": r.headers["etag"]})
    assert again.status_code == 200  # a different period has a different file
    train = client.get("/api/v1/data/futures-bars?through=train", headers=auth(worker))
    same = client.get("/api/v1/data/futures-bars?through=train",
                      headers={**auth(worker), "If-None-Match": train.headers["etag"]})
    assert same.status_code == 304 and same.content == b""


def test_lockbox_prices_only_go_to_a_running_final_check(client, conn, few_days):
    worker = futures_api(client, conn)
    fill(conn)
    fd.futures_periods(conn)
    assert client.get("/api/v1/data/futures-bars?through=lockbox", headers=auth(worker)).status_code == 403
    conn.execute("INSERT INTO models (id, name, module, market, description, how_it_works) "
                 "VALUES ('f1', 'F', 'opening_range', 'futures', 'd', 'h')")
    job = conn.execute("INSERT INTO jobs (kind, status, model_id, lease_worker_id) VALUES "
                       "('final_check', 'leased', 'f1', %s) RETURNING id", (worker["worker_id"],)).fetchone()["id"]
    bad = client.get("/api/v1/data/futures-bars?through=lockbox&job_id=not-a-uuid", headers=auth(worker))
    assert bad.status_code == 403
    ok = client.get(f"/api/v1/data/futures-bars?through=lockbox&job_id={job}", headers=auth(worker))
    assert ok.status_code == 200
    other = enroll(client, conn, "w2")
    assert client.get(f"/api/v1/data/futures-bars?through=lockbox&job_id={job}", headers=auth(other)).status_code == 403
    assert client.get("/api/v1/data/futures-bars?through=train").status_code == 401


def test_payload_before_any_prices(client, conn, few_days):
    worker = futures_api(client, conn)
    r = client.get("/api/v1/data/futures-bars?through=train", headers=auth(worker))
    assert r.status_code == 409 and "Load futures prices first" in r.json()["detail"]


def test_refresh_step_route_for_futures(client, conn):
    worker = futures_api(client, conn)
    r = client.post("/api/v1/data/refresh-step", json={"market": "futures", "symbol": "MES"}, headers=auth(worker))
    assert r.status_code == 200 and r.json()["market"] == "futures" and r.json()["feed"] == "synthetic"
    assert client.post("/api/v1/data/refresh-step", json={"market": "futures", "symbol": "SPY"},
                       headers=auth(worker)).status_code == 400


def test_the_cap_comes_from_topstep_toml(tmp_path):
    p = tmp_path / "topstep.toml"
    assert fd.max_download_usd(p) == 0.0  # no file: refuse every paid download
    p.write_text("[data]\nmax_download_usd = 7.5\n")
    assert fd.max_download_usd(p) == 7.5
    p.write_text("[data]\nmax_download_usd = 'lots'\n")
    assert fd.max_download_usd(p) == 0.0
    from coordinator.config import REPO_ROOT
    assert fd.max_download_usd(REPO_ROOT / "config" / "topstep.toml") > 0


# ------------------------------------------------------------------ the worker's grid and bars


def raw_series(days: list[date], symbol_gaps: dict[int, list[int]] | None = None) -> dict[str, np.ndarray]:
    """Bars at every session minute of `days`, price = minute index, minus some gaps."""
    t = np.concatenate([np.arange(cme.session_minutes(d)) * 60 + cme.session(d)[0] for d in days]).astype(np.int64)
    keep = np.ones(t.shape, dtype=bool)
    for i in (symbol_gaps or {}).get(0, []):
        keep[i] = False
    t = t[keep]
    px = np.arange(t.shape[0], dtype=float) + 100.0
    return {"t": t, "o": px, "h": px + 1, "l": px - 1, "c": px + 0.5, "v": np.ones(t.shape),
            "iid": np.where(t < cme.session(days[-1])[0], 7, 8).astype(np.int64)}


def test_the_grid_fills_quiet_minutes_with_the_last_price_only():
    days = [date(2024, 11, 27), date(2024, 11, 29)]  # a normal day, then a half day
    mes = raw_series(days, {0: [5, 6]})
    data = wfd.build("test", {"MES": mes})
    assert data.n_days == 2 and data.session.tolist() == [390, 210] and data.n_minutes == 600
    assert data.day_start.tolist() == [0, 390] and data.day_end.tolist() == [390, 600]
    assert data.real[0, 5] == data.real[0, 6] == False and data.real[0, 7]  # noqa: E712
    assert data.close[0, 5] == data.close[0, 4] == data.close[0, 6]  # carried forward, never back
    assert data.volume[0, 5] == 0 and data.open[0, 5] == data.close[0, 4]
    assert data.instrument[0, :390].tolist() == [7] * 390 and data.instrument[0, 390:].tolist() == [8] * 210
    assert data.minute[390] == 0 and data.minute[-1] == 209
    assert data.tradable.tolist() == [[True, True]]


def test_a_day_without_bars_is_not_tradable_for_that_symbol():
    days = [date(2024, 3, 11), date(2024, 3, 12)]
    mes = raw_series(days)
    mnq = raw_series(days[:1])
    data = wfd.build("test", {"MES": mes, "MNQ": mnq})
    assert data.tradable.tolist() == [[True, True], [True, False]]


def test_decision_bars_are_built_from_minutes():
    data = wfd.build("test", {"MES": raw_series([date(2024, 3, 11), date(2024, 11, 29)])})
    bars = wfd.resample(data, 15)
    assert bars.n == 26 + 14
    assert bars.start[:3].tolist() == [0, 15, 30] and bars.end[:3].tolist() == [15, 30, 45]
    assert bars.open[0, 1] == data.open[0, 15] and bars.close[0, 1] == data.close[0, 29]
    assert bars.high[0, 1] == data.high[0, 15:30].max() and bars.low[0, 1] == data.low[0, 15:30].min()
    assert bars.volume[0, 1] == 15 and bars.minute[:3].tolist() == [0, 15, 30]
    assert bars.first.nonzero()[0].tolist() == [0, 26] and bars.complete.all()
    assert bars.session.tolist() == [390] * 26 + [210] * 14
    cut = wfd.resample(data.until_minute(20), 15)
    assert cut.n == 2 and cut.complete.tolist() == [True, False] and cut.end.tolist() == [15, 20]
    with pytest.raises(ValueError, match="bar size"):
        wfd.resample(data, 7)


def test_until_day_and_until_minute():
    data = wfd.build("test", {"MES": raw_series([date(2024, 3, 11), date(2024, 3, 12), date(2024, 3, 13)])})
    two = data.until_day(2)
    assert two.n_days == 2 and two.n_minutes == 780 and two.days.tolist() == [20240311, 20240312]
    part = data.until_minute(800)
    assert part.n_days == 3 and part.day_end.tolist() == [390, 780, 800]
    assert data.day_index(20240312) == 1 and data.day_index(20240312, "right") == 2


class FakeOpener:
    """The coordinator's futures-bars route for PriceCache: answers 304 to a matching ETag."""

    def __init__(self, body: bytes, etag: str = '"one"') -> None:
        self.body, self.etag, self.seen = body, etag, []

    def __call__(self, req, timeout=None):
        self.seen.append((req.full_url, req.headers.get("If-none-match")))
        if req.headers.get("If-none-match") == self.etag:
            raise urllib.error.HTTPError(req.full_url, 304, "not modified", {}, io.BytesIO(b""))
        resp = _Resp(self.body)
        resp.headers = {"ETag": self.etag}
        return resp


def npz_of(series: dict[str, dict[str, np.ndarray]], feed: str = "test") -> bytes:
    buf = io.BytesIO()
    meta = {"feed": feed, "symbols": list(series), "through": "train"}
    np.savez_compressed(buf, meta=np.asarray(json.dumps(meta)),
                        **{f"{s}_{k}": v for s, d in series.items() for k, v in d.items()})
    return buf.getvalue()


def test_the_price_cache_downloads_once_per_etag():
    opener = FakeOpener(npz_of({"MES": raw_series([date(2024, 3, 11)])}))
    cache = wfd.PriceCache({"host_url": "http://h", "worker_token": "t"}, opener=opener)
    first = cache.get("train")
    second = cache.get("train")
    assert second is first and cache.downloads == 1
    assert opener.seen == [("http://h/api/v1/data/futures-bars?through=train", None),
                           ("http://h/api/v1/data/futures-bars?through=train", '"one"')]
    opener.etag = '"two"'
    assert cache.get("train") is not first and cache.downloads == 2
    cache.get("lockbox", job_id="j1")
    assert opener.seen[-1][0].endswith("through=lockbox&job_id=j1")


# ------------------------------------------------------------------ the Data refresh job and the dashboard


def test_the_futures_prices_job(client, conn, monkeypatch, test_db_url):
    enroll(client, conn, "w1")
    r = client.post("/api/jobs", json={"kind": "futures_prices", "target": "auto"})
    assert r.status_code == 201 and r.json()["message"].startswith("Data refresh"), r.text
    job = conn.execute("SELECT kind, params FROM jobs").fetchone()
    assert job["kind"] == "data_refresh" and job["params"] == {"markets": ["futures"]}
    from coordinator import cli
    monkeypatch.setattr(cli.Config, "from_env", classmethod(lambda cls, env=None: Config(database_url=test_db_url)))
    assert cli.main(["futures-prices"]) == 0
    assert conn.execute("SELECT count(*) AS n FROM jobs WHERE params = '{\"markets\": [\"futures\"]}'").fetchone()["n"] == 2


def test_the_data_refresh_job_drives_futures_steps(client, conn, monkeypatch):
    worker = futures_api(client, conn)
    calls = []

    def post_json(url, body, token=None, timeout=4.0):
        calls.append(body)
        resp = client.post(url.split("testserver", 1)[1], json=body, headers={"Authorization": "Bearer " + token})
        return resp.json()

    monkeypatch.setattr(data_job.http, "post_json", post_json)
    monkeypatch.setattr(fd, "datetime", type("D", (datetime,), {"now": staticmethod(lambda tz=None: NOW)}))
    rec = Recorder()
    params = {"markets": ["futures"], "_context": {"host_url": "http://testserver", "worker_token": worker["worker_token"]}}
    result = data_job.run_data_refresh(params, None, rec.emit, rec.should_stop)
    assert result["errors"] == {} and result["summary"].endswith("for 2 symbols")
    assert {c["market"] for c in calls} == {"futures"} and [c["symbol"] for c in calls][0] == "MES"
    assert rec.events[-1][2] == "Downloading MNQ (2 of 2)"
    n = conn.execute("SELECT count(*) AS n FROM bars WHERE timeframe = '1Min'").fetchone()["n"]
    assert n == result["bars_added"] == 2 * 34 * 390


def test_futures_models_and_the_forever_final_check_table(conn):
    conn.execute("INSERT INTO models (id, name, module, market, description, how_it_works) "
                 "VALUES ('f1', 'F', 'opening_range', 'futures', 'd', 'h')")
    conn.execute("INSERT INTO final_checks (model_id, feed, result) VALUES ('f1', 'databento', '{}')")
    with pytest.raises(Exception, match="stored forever"):
        conn.execute("UPDATE final_checks SET result = '{\"x\": 1}' WHERE model_id = 'f1'")
    with pytest.raises(Exception, match="stored forever"):
        conn.execute("DELETE FROM final_checks WHERE model_id = 'f1'")
    with pytest.raises(Exception):
        conn.execute("INSERT INTO final_checks (model_id, feed, result) VALUES ('f1', 'databento', '{}')")


def arrays(raw: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k != "meta"}


def test_later_periods_never_reach_an_earlier_one(conn, few_days):
    """Change every price after training (then after held-out): the training file (then
    the held-out file) stays exactly the same, bar for bar."""
    fill(conn)
    periods = fd.futures_periods(conn)
    sizes = {}
    for through in ("train", "held_out"):
        before = arrays(fd.payload(conn, through)[0])
        sizes[through] = before["MES_t"].shape[0]
        cutoff = fd.period_cutoff(periods, through)
        conn.execute("UPDATE bars SET open = open * 3, high = high * 3, low = low * 3, close = close * 3 "
                     "WHERE timeframe = '1Min' AND ts >= %s", (cutoff,))
        conn.execute("UPDATE bar_status SET refreshed_at = clock_timestamp() WHERE timeframe = '1Min'")
        after = arrays(fd.payload(conn, through)[0])
        assert after.keys() == before.keys()
        for k in before:
            assert np.array_equal(after[k], before[k]), k
    lockbox = arrays(fd.payload(conn, "lockbox")[0])
    assert lockbox["MES_t"].shape[0] > sizes["held_out"] > sizes["train"]


def test_run_again_never_reopens_a_kept_final_check(client, conn):
    from coordinator import queue
    conn.execute("INSERT INTO models (id, name, module, market, description, how_it_works) "
                 "VALUES ('f1', 'F', 'opening_range', 'futures', 'd', 'h')")
    job = conn.execute("INSERT INTO jobs (kind, status, model_id) VALUES ('final_check', 'failed', 'f1') "
                       "RETURNING id").fetchone()["id"]
    conn.execute("INSERT INTO final_checks (model_id, feed, result) VALUES ('f1', 'databento', '{}')")
    with pytest.raises(Exception, match="never runs again"):
        queue.run_again(conn, job)
