"""Futures models trading live (stage 6): the backtester's open-end mode gives the same
position the backtest held, minute by minute; live prices; the worker's decisions; the
coordinator's futures executor on Alpaca paper (SPY/QQQ share equivalents) with its
safety rules; the Alpaca paper record against the backtest's range; and Topstep
(through a stand-in TopstepX) with its loss-limit stop, contract limit and the ready gate."""
from __future__ import annotations

import io
import json
import urllib.error
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from coordinator import futures_live, futures_trading, futures_view, models, safety
from coordinator.broker import LIVE, FakeBroker
from coordinator.config import Config
from coordinator.futures_data import FakeFuturesSource
from coordinator.futures_trading import Venues
from coordinator.topstep_broker import FakeTopstep, TopstepError, TopstepLink, TopstepX, fingerprint, make_topstep
from fleet2.sim import cme_session as cme
from fleet2.sim import futures_backtest as fb
from fleet2.sim import futures_data as wfd
from fleet2.sim import topstep
from fleet2.worker import futures_live_job
from tests.conftest import enroll, heartbeat
from tests.test_futures_backtest import COSTS, momentum, scripted, wiggly
from tests.test_futures_screen import FEES, real_metrics, shaped

UTC = timezone.utc
DAY = date(2026, 10, 8)  # a Thursday


def ct(hh: int, mm: int, d: date = DAY) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=cme.CHICAGO).astimezone(UTC)


# ------------------------------------------------------------------ the backtester, live


def held_at(run: fb.FuturesRun, t: int) -> int:
    """Contracts the full backtest held from the open of the minute starting at t."""
    tl = run.trade_list
    out = 0
    for a, b, side, n, reason in zip(tl["entry_t"], tl["exit_t"], tl["side"], tl["contracts"], tl["reason"]):
        if a <= t and (b > t or (b == t and reason != "model")):
            out += int(side * n)
    return out


@pytest.mark.parametrize("stop,target,size", [(0, 0, 1), (12, 20, 1), (8, 0, 5)])
def test_live_decisions_match_the_backtest_minute_by_minute(stop, target, size):
    data = wiggly(days=4)
    model = scripted(momentum, stop_ticks=stop, target_ticks=target, bar_minutes=size)
    full = fb.run(data, model, None, COSTS)
    day = data.n_days - 1
    checked = 0
    for m in range(int(data.day_start[day]) + 1, int(data.day_end[day]), 7):
        live = fb.run(data.until_minute(m), model, None, COSTS, first_day=day, open_end=True)
        expected = held_at(full, int(data.times[m]))
        assert live.final["MES"] == expected, f"minute {m - int(data.day_start[day])}"
        checked += expected != 0
    assert checked > 5


def test_live_is_flat_after_the_cutoff_and_at_the_close():
    data = wiggly(days=2)
    always = scripted(np.ones(2 * 390))
    day_start = int(data.day_start[1])
    early = fb.run(data.until_minute(day_start + 100), always, None, COSTS, first_day=1, open_end=True)
    assert early.final == {"MES": 1}
    late = scripted(lambda bars: np.where(bars.minute >= 381, 1.0, 0.0))  # wants in at 14:51
    assert fb.run(data.until_minute(day_start + 385), late, None, COSTS, first_day=1, open_end=True).final == {"MES": 0}
    closed = fb.run(data, always, None, COSTS, first_day=1, open_end=True)  # the day is over
    assert closed.final == {"MES": 0}


# ------------------------------------------------------------------ the worker's job


def test_the_worker_decides_from_todays_newest_closed_minute():
    src = FakeFuturesSource(seed=4, first=date(2026, 9, 1), last=DAY)
    series = {s: src.series(s) for s in ("MES",)}
    cut = int(np.searchsorted(series["MES"]["t"], int(ct(10, 0).timestamp())))
    data = wfd.build("synthetic", {"MES": {k: v[:cut] for k, v in series["MES"].items()}})
    module = scripted(lambda bars: np.ones(bars.n))
    rules = replace(FEES, commission_per_side={"MES": 0.5, "MNQ": 0.5})
    d = futures_live_job.decide(data, module, module.DEFAULT_PARAMS, 2, rules, cme.as_int(DAY))
    assert d["minute_t"] == int(ct(9, 59).timestamp()) and d["contracts"] == {"MES": 2}
    assert d["reason"] == "After the 10:00 Chicago minute: long 2 MES"
    stale = futures_live_job.decide(data, module, module.DEFAULT_PARAMS, 2, rules, cme.as_int(DAY) + 1)
    assert stale["contracts"] == {"MES": 0} and "no prices for today" in stale["reason"]


# ------------------------------------------------------------------ live prices on the coordinator


def test_live_prices_keep_closed_session_minutes_of_recent_weeks(conn):
    src = FakeFuturesSource(seed=4)
    now = ct(10, 0, DAY) + timedelta(seconds=30)
    added = futures_live.refresh(conn, "synthetic", src, "MES", now)
    first = conn.execute("SELECT min(ts) AS a, max(ts) AS b FROM live_bars").fetchone()
    assert added > 30 * 390 and first["b"] == ct(9, 59)  # the 10:00 minute has not closed yet
    assert first["a"] >= now - timedelta(days=futures_live.LIVE_DAYS)
    assert futures_live.refresh(conn, "synthetic", src, "MES", now) == 0
    body, etag = futures_live.payload(conn, "synthetic")
    data = wfd.from_npz(body)
    assert data.feed == "synthetic" and data.symbols == ("MES",) and int(data.days[-1]) == cme.as_int(DAY)
    assert futures_live.payload(conn, "synthetic")[1] == etag
    assert futures_live.share_price(conn, "SPY") == pytest.approx(float(data.close[0, futures_live_job.newest_closed(data) - 1]) / 10)
    assert futures_live.source_for("alpaca_paper", False) == "alpaca" and futures_live.source_for("topstep", True) == "synthetic"


# ------------------------------------------------------------------ the executor on Alpaca paper


@pytest.fixture
def live(client, conn):
    """The app trading futures on a fake Alpaca paper account and a stand-in TopstepX,
    with synthetic live prices loaded up to 10:00 Chicago time."""
    broker = client.app.state.broker_status.broker
    broker.price_of = lambda s: {"SPY": 500.0, "QQQ": 400.0}[s]
    client.app.state.topstep = FEES
    venues = client.app.state.venues
    venues.fake = True
    venues.sources = {"synthetic": FakeFuturesSource(seed=4)}
    venues.topstep = TopstepLink(FakeTopstep(price_of=lambda s: 5000.0), None)
    futures_live.refresh(conn, "synthetic", venues.sources["synthetic"], "MES", ct(10, 1))
    store = shaped(0.5, 0.3, 1_000.0)
    store["held_out"]["daily_band"] = {"p01": -300.0, "p05": -150.0, "p50": 0.0, "p95": 150.0, "p99": 300.0}
    models.store_backtest(conn, "gap_fade", store, None)
    return venues


def start(client, conn, venue="alpaca_paper"):
    w = enroll(client, conn, f"w-{venue}")
    heartbeat(client, w)
    r = client.post("/api/models/gap_fade/futures/start", json={"venue": venue, "target": w["worker_id"]})
    assert r.status_code == 201, r.text
    job = heartbeat(client, w, want_job=True)["claimed"][0]
    return w, job, r.json()


def signal(client, w, job, contracts, minute_t, reason="test"):
    return client.post("/api/v1/futures/signal", json={"job_id": job["id"], "minute_t": minute_t,
                                                       "contracts": contracts, "reason": reason},
                       headers={"Authorization": "Bearer " + w["worker_token"]})


def run(client, conn, now):
    conn.execute("UPDATE futures_books SET target_at = %s WHERE target_at IS NOT NULL", (now - timedelta(seconds=20),))
    app = client.app.state
    return futures_trading.execute_pass(conn, app.venues, app.topstep, app.limits, now=now)


def fut_orders(conn):
    return conn.execute("SELECT * FROM futures_orders ORDER BY created_at").fetchall()


def test_a_futures_model_paper_trades_spy_shares_long_and_short(client, conn, live):
    w, job, reply = start(client, conn)
    assert job["kind"] == "paper_trade" and job["params"]["market"] == "futures"
    assert job["params"]["venue"] == "alpaca_paper" and job["params"]["source"] == "synthetic"
    assert "(50 SPY shares)" in reply["message"] and job["params"]["contracts"] == 1
    assert signal(client, w, job, {"MES": 1}, 100).json() == {"outcome": "kept"}
    assert signal(client, w, job, {"MES": 1}, 100).json() == {"outcome": "already have this decision"}
    assert signal(client, w, job, {"MES": 3}, 101).status_code == 400  # more than the model's size
    assert run(client, conn, ct(10, 2)) == 1
    (o,) = fut_orders(conn)
    assert (o["instrument"], o["side"], o["qty"], o["contracts"], o["status"]) == ("SPY", "buy", 50.0, 1.0, "filled")
    pos = conn.execute("SELECT * FROM futures_positions").fetchone()
    assert pos["qty"] == 50 and pos["instrument"] == "SPY"
    # A flip: closed first, then the short opened on the next pass.
    signal(client, w, job, {"MES": -1}, 102)
    client.app.state.broker_status.broker.price_of = lambda s: 502.0
    run(client, conn, ct(10, 3))
    assert conn.execute("SELECT count(*) AS n FROM futures_positions").fetchone()["n"] == 0
    run(client, conn, ct(10, 3))
    pos = conn.execute("SELECT * FROM futures_positions").fetchone()
    assert pos["qty"] == -50
    day = conn.execute("SELECT * FROM futures_days").fetchone()
    # Long 50 shares bought at 500.25 (5 bp), sold at 501.749 (5 bp): ~$75 in futures dollars, less
    # 3 sides x $0.62 commission (counted though Alpaca charges none).
    buy, sell = 500.0 * 1.0005, 502.0 * 0.9995
    assert day["pnl"] == pytest.approx(50 * (sell - buy) - 3 * 0.62) and day["trades"] == 2
    assert day["fees"] == pytest.approx(3 * 0.62)
    assert "short 1 MES (50 SPY)" in futures_trading.book_line(conn, conn.execute("SELECT * FROM futures_books").fetchone(),
                                                               ct(10, 3))["line"]


def test_the_executor_closes_by_the_close_even_when_paused(client, conn, live):
    w, job, _ = start(client, conn)
    signal(client, w, job, {"MES": 1}, 100)
    run(client, conn, ct(10, 2))
    client.post("/api/trading/pause")
    signal(client, w, job, {"MES": -1}, 101)
    run(client, conn, ct(11, 0))
    assert conn.execute("SELECT qty FROM futures_positions").fetchone()["qty"] == 50  # paused: nothing new
    book = conn.execute("SELECT waiting FROM futures_books").fetchone()
    assert book["waiting"] == "Waiting: trading is paused"
    run(client, conn, ct(14, 58))  # flat time (15:00 less the 2-minute margin): closing is allowed
    assert conn.execute("SELECT count(*) AS n FROM futures_positions").fetchone()["n"] == 0
    assert "flat by the close" in fut_orders(conn)[-1]["reason"]


def test_no_new_trades_near_the_close_and_none_without_fresh_decisions(client, conn, live):
    w, job, _ = start(client, conn)
    signal(client, w, job, {"MES": 1}, 100)
    run(client, conn, ct(14, 49))  # 14:48 is the cut-off (14:58 flat time less 10 minutes)
    assert fut_orders(conn) == []
    run(client, conn, ct(14, 40))
    assert len(fut_orders(conn)) == 1
    conn.execute("UPDATE futures_books SET target_at = %s", (ct(14, 40) - timedelta(minutes=10),))
    app = client.app.state
    futures_trading.execute_pass(conn, app.venues, app.topstep, app.limits, now=ct(14, 41))
    assert conn.execute("SELECT count(*) AS n FROM futures_positions").fetchone()["n"] == 0
    assert conn.execute("SELECT waiting FROM futures_books").fetchone()["waiting"].startswith("No fresh decision")


def test_futures_paper_limits_and_never_in_live_mode(client, conn, live):
    w, job, _ = start(client, conn)
    client.app.state.limits = replace(client.app.state.limits, futures_paper_max_dollars=10_000.0)
    signal(client, w, job, {"MES": 1}, 100)
    run(client, conn, ct(10, 2))
    (o,) = fut_orders(conn)
    assert o["status"] == "blocked" and "limit for all futures paper models" in o["error"]
    client.app.state.limits = replace(client.app.state.limits, futures_paper_max_dollars=150_000.0)
    client.app.state.broker_status.broker.mode = LIVE
    run(client, conn, ct(10, 4))
    assert "only paper trade on Alpaca" in fut_orders(conn)[-1]["error"]
    client.post("/api/models/gap_fade/futures/stop", json={"venue": "alpaca_paper"})
    r = client.post("/api/models/gap_fade/futures/start", json={"venue": "alpaca_paper"})
    assert r.status_code == 409 and "still closing" in r.json()["detail"]
    run(client, conn, ct(10, 5))  # nothing held: the book closes
    assert conn.execute("SELECT status FROM futures_books").fetchone()["status"] == "closed"
    r = client.post("/api/models/gap_fade/futures/start", json={"venue": "alpaca_paper"})
    assert r.status_code == 409 and "live mode" in r.json()["detail"]


def test_starting_needs_a_tested_model_and_the_fees(client, conn, live):
    assert client.post("/api/models/pullback/futures/start", json={}).status_code == 409  # untested
    client.app.state.topstep = topstep.Rules()
    r = client.post("/api/models/gap_fade/futures/start", json={})
    assert r.status_code == 409 and "Set the fee" in r.json()["detail"]


def test_a_dead_worker_is_restarted_by_run_again(client, conn, live):
    from coordinator import queue
    w, job, _ = start(client, conn)
    conn.execute("UPDATE jobs SET status = 'failed', finished_at = now() WHERE id = %s", (job["id"],))
    again = queue.run_again(conn, job["id"])
    book = conn.execute("SELECT job_id FROM futures_books").fetchone()
    assert str(book["job_id"]) == str(again.jobs[0]["id"]) and again.jobs[0]["params"]["venue"] == "alpaca_paper"


# ------------------------------------------------------------------ the paper record and the checklist


def test_paper_days_are_compared_with_the_backtests_range(conn):
    rules = FEES
    band = {"p01": -300.0, "p99": 300.0}
    conn.execute("INSERT INTO models (id, name, module, market, description, how_it_works) "
                 "VALUES ('f1', 'F', 'gap_fade', 'futures', 'd', 'h')")
    book = conn.execute("INSERT INTO futures_books (model_id, venue, contracts, status) VALUES "
                        "('f1', 'alpaca_paper', 2, 'closed') RETURNING id").fetchone()["id"]
    days = [date(2026, 9, 1) + timedelta(days=i) for i in range(25)]
    for i, d in enumerate(days):
        conn.execute("INSERT INTO futures_days (book_id, day, pnl) VALUES (%s, %s, %s)", (book, d, 50.0 if i else 700.0))
    rec = futures_trading.paper_record(conn, "f1", band, 2, rules, date(2026, 10, 8))
    # At 2 contracts the range is -$600 to +$600: the $700 day is outside it.
    assert rec == {"days": 25, "total": 700.0 + 24 * 50.0, "outside": 1, "low": -600.0, "high": 600.0}
    assert futures_trading.paper_ok(rec, rules) == "ok"
    assert futures_trading.paper_ok({**rec, "outside": 3}, rules) == "no"  # more than 10% of 25 days
    assert futures_trading.paper_ok({**rec, "days": 2}, rules) == "pending"
    assert futures_trading.paper_ok({**rec, "total": -1.0}, rules) == "no"
    assert futures_trading.paper_record(conn, "f1", band, 2, rules, days[5])["days"] == 5  # today is not finished
    assert futures_trading.paper_ok(futures_trading.paper_record(conn, "f1", None, 2, rules, days[5]), rules) == "no"


def test_the_checklist_needs_alpaca_paper_days(client, conn, live):
    v = futures_view.model_verdict(conn, "gap_fade", FEES)
    line = v["items"][5]
    assert line["label"].startswith("Alpaca paper trading: 3+ days") and line["state"] == "pending"
    assert not v["ready"]


# ------------------------------------------------------------------ Topstep


def test_only_a_ready_model_trades_on_topstep(client, conn, live, monkeypatch):
    r = client.post("/api/models/gap_fade/futures/start", json={"venue": "topstep"})
    assert r.status_code == 409 and "not ready for a Combine" in r.json()["detail"]
    html = client.get("/models?market=futures&id=gap_fade").text
    assert "Only a model ready for a Combine" in html
    monkeypatch.setattr(futures_view, "model_verdict", lambda conn, mid, rules, now=None: {"ready": True})
    w, job, reply = start(client, conn, "topstep")
    assert job["params"]["source"] == "synthetic" and "on Topstep at 1 contract" in reply["message"]
    signal(client, w, job, {"MES": 1}, 100)
    run(client, conn, ct(10, 2))
    (o,) = fut_orders(conn)
    assert (o["venue"], o["instrument"], o["qty"], o["status"]) == ("topstep", "CON.F.US.MES.DEMO", 1.0, "filled")
    day = conn.execute("SELECT fees FROM futures_days").fetchone()
    assert day["fees"] == pytest.approx(0.62)


def test_topstep_stops_at_80_percent_of_the_loss_limit(client, conn, live, monkeypatch):
    monkeypatch.setattr(futures_view, "model_verdict", lambda conn, mid, rules, now=None: {"ready": True})
    w, job, _ = start(client, conn, "topstep")
    signal(client, w, job, {"MES": 1}, 100)
    run(client, conn, ct(10, 2))
    venues = client.app.state.venues
    # Floor $48,000: 80% of the $2,000 limit is used up at a balance of $48,400.
    venues.topstep.client.balance = 48_401.0
    assert futures_trading.topstep_check(conn, venues, FEES, ct(10, 3))["room"] == pytest.approx(401.0, abs=200)
    assert safety.topstep_paused(conn) is None
    venues.topstep.client.balance = 48_390.0
    futures_trading.topstep_check(conn, venues, FEES, ct(10, 4))
    assert safety.topstep_paused(conn).startswith("Topstep loss limit")
    signal(client, w, job, {"MES": 1}, 101)
    run(client, conn, ct(10, 5))
    assert conn.execute("SELECT count(*) AS n FROM futures_positions").fetchone()["n"] == 0  # closed out
    assert client.post("/api/models/gap_fade/futures/start", json={"venue": "topstep"}).status_code == 409
    assert client.post("/api/topstep/resume").json() == {"message": "Topstep trading resumed"}


def test_the_floor_follows_topsteps_end_of_day_balances(conn):
    rules = FEES
    assert safety.topstep_floor(conn, "a1", rules) == 48_000
    conn.execute("INSERT INTO topstep_days VALUES ('a1', '2026-10-06', 50500), ('a1', '2026-10-07', 50100)")
    assert safety.topstep_floor(conn, "a1", rules) == 48_500
    conn.execute("INSERT INTO topstep_days VALUES ('a1', '2026-10-08', 53000)")
    assert safety.topstep_floor(conn, "a1", rules) == 50_000


def test_the_topstep_contract_limit(client, conn, live, monkeypatch):
    monkeypatch.setattr(futures_view, "model_verdict", lambda conn, mid, rules, now=None: {"ready": True})
    client.app.state.topstep = replace(FEES, max_micro_contracts=1)
    conn.execute("INSERT INTO futures_books (model_id, venue, contracts, target, target_at) VALUES "
                 "('pullback', 'topstep', 1, '{\"MNQ\": 1}', now())")  # holding 1 MNQ and wanting to keep it
    other = conn.execute("SELECT id FROM futures_books").fetchone()["id"]
    conn.execute("INSERT INTO futures_positions VALUES (%s, 'MNQ', 'CON.F.US.MNQ.DEMO', 1, 20000)", (other,))
    w, job, _ = start(client, conn, "topstep")
    signal(client, w, job, {"MES": 1}, 100)
    run(client, conn, ct(10, 2))
    blocked = [o for o in fut_orders(conn) if o["model_id"] == "gap_fade"]
    assert blocked and "over Topstep's limit of 1 micro contracts" in blocked[0]["error"]


def test_topstep_needs_keys_and_a_confirmation_for_exactly_those_keys(client, conn):
    assert make_topstep(Config(), None).problem.startswith("Add TOPSTEPX_USERNAME")
    cfg = Config(topstepx_username="me", topstepx_api_key="k1", topstepx_account="50KTC")
    assert "Type TRADE ON TOPSTEP" in make_topstep(cfg, None).problem
    assert make_topstep(cfg, {"key": fingerprint(cfg)}).on
    assert not make_topstep(replace(cfg, topstepx_api_key="k2"), {"key": fingerprint(cfg)}).on
    assert "k1" not in repr(cfg)
    r = client.post("/api/topstep/confirm", json={"confirm": "TRADE ON TOPSTEP"})
    assert r.status_code == 409 and "TOPSTEPX_USERNAME" in r.json()["detail"]
    client.app.state.config = replace(client.app.state.config, topstepx_username="me", topstepx_api_key="k1",
                                      topstepx_account="50KTC")
    cfg = client.app.state.config
    assert client.post("/api/topstep/confirm", json={"confirm": "yes"}).status_code == 400
    r = client.post("/api/topstep/confirm", json={"confirm": "TRADE ON TOPSTEP"})
    assert r.status_code == 200 and "Restart" in r.json()["message"]
    stored = conn.execute("SELECT value FROM settings WHERE key = 'topstep_confirmed'").fetchone()["value"]
    assert stored == {"key": fingerprint(cfg)}


# ------------------------------------------------------------------ the TopstepX client


class ProjectX:
    """Stands in for TopstepX's API: records requests, answers like the ProjectX docs."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, str | None]] = []
        self.expire_once = False

    def __call__(self, req, timeout=None):
        path = req.full_url.split("topstepx.com", 1)[1]
        body = json.loads(req.data)
        auth = req.headers.get("Authorization")
        self.calls.append((path, body, auth))
        if self.expire_once and path != "/api/Auth/loginKey":
            self.expire_once = False
            raise urllib.error.HTTPError(req.full_url, 401, "expired", {}, io.BytesIO(b""))
        answers = {
            "/api/Auth/loginKey": {"token": f"tok{len(self.calls)}"},
            "/api/Account/search": {"accounts": [{"id": 77, "name": "50KTC-1", "balance": 50210.5, "canTrade": True}]},
            "/api/Contract/search": {"contracts": [{"id": "CON.F.US.MES.Z26", "name": "MESZ6", "activeContract": True},
                                                   {"id": "CON.F.US.MES.H27", "name": "MESH7", "activeContract": False}]},
            "/api/History/retrieveBars": {"bars": [{"t": "2026-10-08T14:31:00+00:00", "o": 5000.25, "h": 5001,
                                                    "l": 4999.5, "c": 5000.75, "v": 120},
                                                   {"t": "2026-10-08T14:30:00+00:00", "o": 5000, "h": 5000.5,
                                                    "l": 4999.75, "c": 5000.25, "v": 90}]},
            "/api/Order/place": {"orderId": 9056},
            "/api/Order/search": {"orders": [{"id": 9056, "status": 2, "fillVolume": 1, "filledPrice": 5000.5,
                                              "customTag": "abc"}]},
            "/api/Position/searchOpen": {"positions": [{"contractId": "CON.F.US.MES.Z26", "type": 2, "size": 1,
                                                        "averagePrice": 5000.5}]},
        }
        out = {"success": True, "errorCode": 0, "errorMessage": None, **answers[path]}
        if path == "/api/Order/place" and body["size"] > 50:
            out = {"success": False, "errorCode": 2, "errorMessage": "Size too large"}
        from tests.test_futures_data import _Resp
        return _Resp(json.dumps(out).encode())


def test_the_topstepx_client_speaks_the_projectx_api():
    api = ProjectX()
    cfg = Config(topstepx_username="me", topstepx_api_key="key", topstepx_account="50KTC-1")
    tx = TopstepX(cfg, opener=api)
    tx.limiter.acquire = tx.bars_limiter.acquire = lambda: 0.0
    acc = tx.account()
    assert (acc.id, acc.name, acc.balance, acc.can_trade) == ("77", "50KTC-1", 50210.5, True)
    login, search = api.calls[0], api.calls[1]
    assert login[0] == "/api/Auth/loginKey" and login[1] == {"userName": "me", "apiKey": "key"} and login[2] is None
    assert search[1] == {"onlyActiveAccounts": True} and search[2] == "Bearer tok1"
    assert tx.contract("MES") == "CON.F.US.MES.Z26"
    bars = tx.bars("MES", datetime(2026, 10, 8, 14, 0, tzinfo=UTC), datetime(2026, 10, 8, 15, 0, tzinfo=UTC))
    assert [b["t"] for b in bars] == [int(datetime(2026, 10, 8, 14, 30, tzinfo=UTC).timestamp()),
                                      int(datetime(2026, 10, 8, 14, 31, tzinfo=UTC).timestamp())]
    req = next(c for c in api.calls if c[0] == "/api/History/retrieveBars")[1]
    assert req["contractId"] == "CON.F.US.MES.Z26" and req["unit"] == 2 and req["unitNumber"] == 1
    assert req["live"] is False and req["startTime"] == "2026-10-08T14:00:00.000Z" and req["includePartialBar"] is False
    state = tx.place_order("abc", "CON.F.US.MES.Z26", "sell", 1)
    place = next(c for c in api.calls if c[0] == "/api/Order/place")[1]
    assert place == {"accountId": 77, "contractId": "CON.F.US.MES.Z26", "type": 2, "side": 1, "size": 1,
                     "customTag": "abc"}
    assert state.status == "submitted" and state.broker_order_id == "9056"
    done = tx.get_order("abc", "9056")
    assert (done.status, done.filled_qty, done.filled_avg_price) == ("filled", 1.0, 5000.5)
    assert tx.positions() == {"CON.F.US.MES.Z26": (-1, 5000.5)}
    refused = tx.place_order("big", "CON.F.US.MES.Z26", "buy", 60)
    assert refused.status == "rejected" and "Size too large" in refused.error
    api.expire_once = True  # an expired token: logs in again once and carries on
    assert tx.account(refresh=True).id == "77"
    assert [c[0] for c in api.calls][-3:] == ["/api/Account/search", "/api/Auth/loginKey", "/api/Account/search"]
    with pytest.raises(TopstepError, match="TOPSTEPX_ACCOUNT"):
        TopstepX(replace(cfg, topstepx_account=""), opener=api).account()
