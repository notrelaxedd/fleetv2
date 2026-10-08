"""Regression tests for the findings of the independent review of the backtester and
the safety controls (each test fails on the code as it was before the fix)."""
from __future__ import annotations

import types

import numpy as np
import pytest

from coordinator import queue, safety, trading
from coordinator.broker import NEVER_REACHED, AlpacaBroker, OrderState
from coordinator.limits import Limits
from fleet2.sim.backtest import Run
from fleet2.sim.metrics import summarize
from tests.conftest import enroll, heartbeat
from tests.test_safety import orders, paper_model, run_executor, seed_prices, send_signal

LIMITS = Limits()


def _book(conn, model_id="momentum"):
    return trading.open_book(conn, model_id)


# 1. an unclear answer from Alpaca is never booked as "rejected"


class _ApiError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.status_code = status


def _alpaca(raises: Exception | None = None, lookup: Exception | None = None, order: object | None = None) -> AlpacaBroker:
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker.mode = "paper"

    def submit(request):
        if raises:
            raise raises
        return order

    def by_client_id(cid):
        if lookup:
            raise lookup
        return order

    broker._client = types.SimpleNamespace(submit_order=submit, get_order_by_client_id=by_client_id)
    return broker


def test_alpaca_answers_are_sorted_into_refused_or_unknown():
    assert _alpaca(raises=_ApiError(403)).place_order("c", "AAPL", "buy", notional=10).status == "rejected"
    unknown = _alpaca(raises=TimeoutError("read timed out")).place_order("c", "AAPL", "buy", notional=10)
    assert unknown.status == "submitted" and "checking" in unknown.error
    assert _alpaca(raises=_ApiError(503)).place_order("c", "AAPL", "buy", notional=10).status == "submitted"
    assert _alpaca(lookup=_ApiError(404)).get_order("c") == OrderState("rejected", error=NEVER_REACHED)


def test_an_order_with_no_clear_answer_is_checked_and_not_sent_twice(client, conn):
    seed_prices(conn, {"AAPL": 100.0})
    w, job = paper_model(client, conn)
    status = client.app.state.broker_status
    broker = status.broker
    broker.price_of = lambda s: 100.0
    real_place = broker.place_order

    def accepted_but_timed_out(cid, symbol, side, notional=None, qty=None):
        real_place(cid, symbol, side, notional=notional, qty=qty)  # Alpaca has it and fills it
        return OrderState("submitted", error="no clear answer from Alpaca (read timed out); checking")

    broker.place_order = accepted_but_timed_out
    send_signal(client, w, job, {"AAPL": 0.1})
    trading.execute_pass(conn, status, LIMITS)
    for _ in range(3):
        trading.poll_fills(conn, status)
        trading.execute_pass(conn, status, LIMITS)
    placed = orders(conn)
    assert len(placed) == 1 and placed[0]["status"] == "filled" and placed[0]["error"] is None
    assert _book(conn)["cash"] == pytest.approx(9_000.0)


# 2. a crash between the record and Alpaca's answer does not freeze the book


def test_a_submitting_order_left_by_a_crash_is_resolved(client, conn):
    seed_prices(conn, {"AAPL": 100.0})
    w, job = paper_model(client, conn)
    status = client.app.state.broker_status
    status.broker.price_of = lambda s: 100.0
    book = _book(conn)
    row = conn.execute(
        "INSERT INTO orders (book_id, model_id, mode, symbol, side, notional, reason, status, created_at)"
        " VALUES (%s, 'momentum', 'paper', 'AAPL', 'buy', 500, 'test', 'submitting', now() - interval '1 minute') RETURNING id",
        (book["id"],),
    ).fetchone()
    status.broker.place_order(str(row["id"]), "AAPL", "buy", notional=500.0)  # Alpaca did get it
    ghost = conn.execute(
        "INSERT INTO orders (book_id, model_id, mode, symbol, side, notional, reason, status, created_at)"
        " VALUES (%s, 'momentum', 'paper', 'AAPL', 'buy', 500, 'test', 'submitting', now() - interval '1 minute') RETURNING id",
        (book["id"],),
    ).fetchone()  # this one never reached Alpaca
    conn.execute("UPDATE books SET targets_bar_t = 1, executed_bar_t = 1 WHERE id = %s", (book["id"],))
    trading.poll_fills(conn, status)
    got = {str(o["id"]): o for o in orders(conn)}
    assert got[str(row["id"])]["status"] == "filled"
    assert got[str(ghost["id"])]["status"] == "rejected" and got[str(ghost["id"])]["error"] == NEVER_REACHED
    assert _book(conn)["executed_bar_t"] is None  # the decision is planned again
    assert _book(conn)["cash"] == pytest.approx(9_500.0)


# 3. the daily loss check fails closed


def test_no_account_reading_means_no_order(client, conn):
    seed_prices(conn, {"AAPL": 100.0})
    paper_model(client, conn)
    status = client.app.state.broker_status
    broker = status.broker

    def down():
        from coordinator.broker import BrokerUnavailable
        raise BrokerUnavailable("Alpaca account: 503")

    broker.account = down
    row = safety.approve_and_place(conn, status, LIMITS, _book(conn),
                                   {"symbol": "AAPL", "side": "buy", "notional": 500.0, "reason": "t"}, {"AAPL": 100.0})
    assert row["status"] == "blocked" and row["error"] == safety.NO_ACCOUNT
    assert status.account_info is None  # the old reading is not kept
    assert not safety.is_paused(conn)  # a missing reading waits; it does not pause


def test_no_market_clock_means_no_stock_order(client, conn):
    seed_prices(conn, {"AAPL": 100.0})
    paper_model(client, conn)
    status = client.app.state.broker_status

    def no_clock():
        from coordinator.broker import BrokerUnavailable
        raise BrokerUnavailable("Alpaca clock: timeout")

    status.broker.clock = no_clock
    row = safety.approve_and_place(conn, status, LIMITS, _book(conn),
                                   {"symbol": "AAPL", "side": "buy", "notional": 500.0, "reason": "t"}, {"AAPL": 100.0})
    assert row["error"] == safety.NO_CLOCK


# 5. a refused sale of a stopping model backs off instead of repeating every pass


def test_a_refused_sale_of_a_stopping_model_backs_off(client, conn):
    seed_prices(conn, {"AAPL": 100.0})
    w, job = paper_model(client, conn)
    send_signal(client, w, job, {"AAPL": 0.1})
    run_executor(client, conn)
    client.post("/api/models/momentum/paper/stop")
    client.app.state.broker_status.broker.refuse = "Alpaca refused the order: asset not tradable"
    for _ in range(5):
        run_executor(client, conn)
    sells = [o for o in orders(conn) if o["side"] == "sell"]
    assert len(sells) == 1 and sells[0]["status"] == "rejected"
    assert _book(conn)["waiting"] == "Alpaca refused a sale; trying again in a few minutes"


# 8. a job handed back for an update can move to another worker


def test_an_update_hand_off_is_not_pinned_to_a_worker_that_never_returns(client, conn):
    w = enroll(client, conn, "w20")
    heartbeat(client, w)
    client.post("/api/jobs", json={"kind": "sleep", "target": "auto"})
    job = heartbeat(client, w, want_job=True)["claimed"][0]
    heartbeat(client, w, released=[{"id": job["id"], "lease_token": job["lease_token"], "reason": "update"}])
    conn.execute("UPDATE workers SET last_heartbeat_at = now() - interval '10 minutes'")
    other = enroll(client, conn, "w21")
    heartbeat(client, other)
    queue.release_dead_targets(conn)
    queue.dispatch(conn)
    assert heartbeat(client, other, want_job=True)["claimed"][0]["id"] == job["id"]


# 9 and 11. crypto booking and fills without a price


def test_crypto_fees_come_out_of_the_coins_and_cash_never_goes_negative(client, conn):
    from coordinator import models

    seed_prices(conn, {"BTC/USD": 50_000.0, "ETH/USD": 2_000.0, "SOL/USD": 100.0})
    models.store_backtest(conn, "crypto_trend", {"held_out": {"roi": 0.1, "trades": 200}}, None)
    w = enroll(client, conn, "w30")
    heartbeat(client, w)
    client.post("/api/models/crypto_trend/paper/start", json={"target": w["worker_id"]})
    job = heartbeat(client, w, want_job=True)["claimed"][0]
    conn.execute("UPDATE books SET cash = 2000")
    send_signal(client, w, job, {"BTC/USD": 0.3, "ETH/USD": 0.3, "SOL/USD": 0.3})
    run_executor(client, conn)
    book = _book(conn, "crypto_trend")
    assert book["cash"] >= -1e-6
    btc = conn.execute("SELECT qty, cost FROM positions WHERE symbol = 'BTC/USD'").fetchone()
    filled = conn.execute("SELECT filled_qty, filled_avg_price FROM orders WHERE symbol = 'BTC/USD'").fetchone()
    assert btc["qty"] == pytest.approx(filled["filled_qty"] * (1 - 0.0025))
    assert btc["cost"] == pytest.approx(filled["filled_qty"] * filled["filled_avg_price"])


def test_a_fill_without_a_price_is_not_marked_booked(client, conn):
    seed_prices(conn, {"AAPL": 100.0})
    paper_model(client, conn)
    status = client.app.state.broker_status
    book = _book(conn)
    row = conn.execute(
        "INSERT INTO orders (book_id, model_id, mode, symbol, side, notional, reason, status)"
        " VALUES (%s, 'momentum', 'paper', 'AAPL', 'buy', 500, 't', 'submitted') RETURNING id", (book["id"],)).fetchone()
    status.broker._orders = {str(row["id"]): OrderState("partially_filled", 2.0, None)}
    trading.poll_fills(conn, status)
    assert conn.execute("SELECT booked_qty FROM orders WHERE id = %s", (row["id"],)).fetchone()["booked_qty"] == 0
    status.broker._orders = {str(row["id"]): OrderState("filled", 5.0, 100.0)}
    trading.poll_fills(conn, status)
    assert conn.execute("SELECT qty FROM positions").fetchone()["qty"] == pytest.approx(5.0)


# 12. pause waits for an order in flight


def test_pause_waits_for_an_order_in_flight(client, conn, test_db_url):
    import threading

    import psycopg
    from psycopg.rows import dict_row

    seed_prices(conn, {"AAPL": 100.0})
    paper_model(client, conn)
    status = client.app.state.broker_status
    real_place = status.broker.place_order
    in_flight, release = threading.Event(), threading.Event()
    events: list[str] = []

    def slow_place(*a, **kw):
        in_flight.set()
        release.wait(5)
        events.append("order answered")
        return real_place(*a, **kw)

    status.broker.place_order = slow_place

    def place():
        with psycopg.connect(test_db_url, row_factory=dict_row) as c:
            safety.approve_and_place(c, status, LIMITS, _book(conn),
                                     {"symbol": "AAPL", "side": "buy", "notional": 500.0, "reason": "t"}, {"AAPL": 100.0})

    t = threading.Thread(target=place)
    t.start()
    assert in_flight.wait(5)

    def pause():
        with psycopg.connect(test_db_url, row_factory=dict_row) as c:
            safety.pause_trading(c, "owner", "Paused by you")
            c.commit()
        events.append("pause returned")

    p = threading.Thread(target=pause)
    p.start()
    p.join(0.5)
    assert p.is_alive()  # pause is waiting for the order in flight
    release.set()
    t.join(5)
    p.join(5)
    assert events == ["order answered", "pause returned"]


# 13. drawdown and Sharpe count the first bar


def test_drawdown_counts_a_loss_on_the_first_bar():
    run = Run(times=np.arange(3), equity=np.array([9_500.0, 9_400.0, 9_000.0]), benchmark=np.full(3, np.nan),
              money=10_000.0)
    assert summarize(run)["max_drawdown"] == pytest.approx(0.10)
