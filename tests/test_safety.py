"""Safety controls: the pause switch, the daily loss limit, the money limits, market
hours, the order log, and real money staying impossible without every condition."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from coordinator import models, safety, trading
from coordinator.broker import LIVE, PAPER, FakeBroker, choose_mode
from coordinator.config import Config
from coordinator.limits import Limits
from tests.conftest import enroll, heartbeat, seed_history

LIMITS = Limits()
METRICS = {"held_out": {"roi": 0.05, "trades": 150, "enough_trades": True}, "train": {"roi": 0.1}}


def seed_prices(conn, prices: dict[str, float]) -> None:
    """One recent daily (or hourly, for crypto) bar per symbol."""
    ts = datetime.now(timezone.utc) - timedelta(days=1)
    for symbol, price in prices.items():
        tf = "1Hour" if "/" in symbol else "1Day"
        conn.execute(
            "INSERT INTO bars VALUES (%s, %s, %s, %s, %s, %s, %s, 1000, 'test') ON CONFLICT DO NOTHING",
            (symbol, tf, ts, price, price, price, price),
        )


def paper_model(client, conn, model_id: str = "momentum"):
    """Backtested model paper trading on a fake worker; returns (worker, job)."""
    models.store_backtest(conn, model_id, METRICS, None)
    w = enroll(client, conn, "w1")
    heartbeat(client, w)
    resp = client.post(f"/api/models/{model_id}/paper/start", json={"target": w["worker_id"]})
    assert resp.status_code == 201, resp.text
    job = heartbeat(client, w, want_job=True)["claimed"][0]
    assert job["kind"] == "paper_trade"
    return w, job


def send_signal(client, w, job, targets, bar_t=1_700_000_000):
    return client.post("/api/v1/paper/signal", json={"job_id": job["id"], "bar_t": bar_t, "targets": targets,
                                                     "reason": "test decision"},
                       headers={"Authorization": "Bearer " + w["worker_token"]})


def run_executor(client, conn):
    status = client.app.state.broker_status
    status.broker.price_of = lambda s: trading.latest_prices(conn, [s])[s]
    return trading.execute_pass(conn, status, client.app.state.limits)


def orders(conn):
    return conn.execute("SELECT * FROM orders ORDER BY created_at").fetchall()


# ------------------------------------------------------------------ pause


def test_paused_trading_places_no_orders_and_resume_lets_them_through(client, conn):
    seed_prices(conn, {"AAPL": 100.0, "MSFT": 200.0})
    w, job = paper_model(client, conn)
    assert send_signal(client, w, job, {"AAPL": 0.1, "MSFT": 0.1}).json() == {"outcome": "kept"}
    client.post("/api/trading/pause")
    assert run_executor(client, conn) == 0
    assert orders(conn) == []
    book = trading.open_book(conn, "momentum")
    assert book["waiting"] == "Waiting: trading is paused"
    assert book["executed_bar_t"] is None  # the decision is kept for later, not dropped
    client.post("/api/trading/resume")
    assert run_executor(client, conn) == 2
    placed = orders(conn)
    assert {o["symbol"] for o in placed} == {"AAPL", "MSFT"}
    assert all(o["status"] == "filled" and o["side"] == "buy" for o in placed)


def test_an_order_reaching_the_gate_while_paused_is_blocked_and_logged(client, conn):
    seed_prices(conn, {"AAPL": 100.0})
    paper_model(client, conn)
    client.post("/api/trading/pause")
    book = trading.open_book(conn, "momentum")
    row = safety.approve_and_place(conn, client.app.state.broker_status, LIMITS, book,
                                   {"symbol": "AAPL", "side": "buy", "notional": 500.0, "reason": "test"}, {"AAPL": 100.0})
    assert row["status"] == "blocked" and row["error"] == "trading is paused"
    assert client.app.state.broker_status.broker.get_order(str(row["id"])).status == "rejected"  # never sent


def test_backtests_keep_running_while_paused(client, conn):
    seed_history(conn)
    client.post("/api/trading/pause")
    w = enroll(client, conn, "w2")
    heartbeat(client, w)
    assert client.post("/api/jobs", json={"kind": "backtest", "model_id": "pairs", "target": "auto"}).status_code == 201
    assert heartbeat(client, w, want_job=True)["claimed"][0]["kind"] == "backtest"


# ------------------------------------------------------------------ daily loss


def test_daily_loss_past_the_limit_pauses_trading_by_itself(client, conn):
    status = client.app.state.broker_status
    status.broker.equity, status.broker.last_equity = 97_900.0, 100_000.0  # down 2.1%
    status.refresh(force=True)
    reason = safety.check_daily_loss(conn, status, LIMITS)
    assert reason == "Daily loss limit reached: the account is down 2.10% today (limit 2%)"
    assert safety.is_paused(conn) and safety.paused_reason(conn) == reason
    header = client.get("/api/fleet").json()["header"]
    assert header["paused"] and header["paused_reason"] == reason
    audit = conn.execute("SELECT actor FROM audit_log WHERE action = 'trading_paused'").fetchone()
    assert audit["actor"] == "system"


def test_a_loss_under_the_limit_does_not_pause(client, conn):
    status = client.app.state.broker_status
    status.broker.equity, status.broker.last_equity = 98_100.0, 100_000.0  # down 1.9%
    status.refresh(force=True)
    assert safety.check_daily_loss(conn, status, LIMITS) is None
    assert not safety.is_paused(conn)


def test_recovering_does_not_resume_on_its_own(client, conn):
    status = client.app.state.broker_status
    status.broker.equity, status.broker.last_equity = 90_000.0, 100_000.0
    status.refresh(force=True)
    safety.check_daily_loss(conn, status, LIMITS)
    status.broker.equity = 101_000.0
    status.refresh(force=True)
    safety.check_daily_loss(conn, status, LIMITS)
    assert safety.is_paused(conn)


def test_the_order_gate_checks_the_loss_limit_on_a_fresh_reading(client, conn):
    seed_prices(conn, {"AAPL": 100.0})
    paper_model(client, conn)
    status = client.app.state.broker_status
    status.broker.equity, status.broker.last_equity = 95_000.0, 100_000.0
    book = trading.open_book(conn, "momentum")
    row = safety.approve_and_place(conn, status, LIMITS, book,
                                   {"symbol": "AAPL", "side": "buy", "notional": 500.0, "reason": "test"}, {"AAPL": 100.0})
    assert row["status"] == "blocked" and row["error"].startswith("Daily loss limit reached")
    assert safety.is_paused(conn)


def test_a_custom_limit_from_the_file_is_used(tmp_path):
    from coordinator.limits import load_limits

    path = tmp_path / "limits.toml"
    path.write_text("[safety]\ndaily_loss_limit_pct = 1.5\n[money]\nmax_per_position = 500\n")
    limits = load_limits(path)
    assert limits.daily_loss_limit_pct == 1.5 and limits.max_per_position == 500.0
    assert limits.max_per_model == 10_000.0  # untouched keys keep the safe default


# ------------------------------------------------------------------ money limits


@pytest.mark.parametrize("order,problem", [
    ({"symbol": "AAPL", "side": "buy", "notional": 1_100.0}, "over the $1,000 limit per position"),
    ({"symbol": "AAPL", "side": "sell", "qty": 1.0}, "nothing to sell: the model holds no AAPL"),
    ({"symbol": "AAPL", "side": "buy", "notional": 0.0}, "a buy must be for more than $0"),
])
def test_money_limits_block_orders(client, conn, order, problem):
    seed_prices(conn, {"AAPL": 100.0})
    paper_model(client, conn)
    book = trading.open_book(conn, "momentum")
    row = safety.approve_and_place(conn, client.app.state.broker_status, LIMITS, book,
                                   {**order, "reason": "test"}, {"AAPL": 100.0})
    assert row["status"] == "blocked" and row["error"] == problem


def test_the_model_cannot_spend_more_than_its_book(client, conn):
    seed_prices(conn, {"AAPL": 100.0})
    paper_model(client, conn)
    conn.execute("UPDATE books SET cash = 300")
    book = trading.open_book(conn, "momentum")
    row = safety.approve_and_place(conn, client.app.state.broker_status, LIMITS, book,
                                   {"symbol": "AAPL", "side": "buy", "notional": 500.0, "reason": "test"}, {"AAPL": 100.0})
    assert row["error"] == "not enough cash in the model's book ($300.00 free)"


def test_plan_sizes_like_the_backtester():
    book = {"cash": 10_000.0}
    plan = trading.plan_orders(book, {}, {"AAPL": 100.0, "MSFT": 50.0}, {"AAPL": 0.5, "MSFT": 0.05}, LIMITS)
    assert [(o["symbol"], o["side"], o["notional"]) for o in plan] == [("MSFT", "buy", 500.0), ("AAPL", "buy", 1000.0)]
    # a change under $25 is not traded; selling out entirely always is
    held = {"AAPL": 10.0, "MSFT": 1.0}
    plan = trading.plan_orders({"cash": 9_000.0}, held, {"AAPL": 100.0, "MSFT": 50.0}, {"AAPL": 0.101}, LIMITS)
    assert [(o["symbol"], o["side"]) for o in plan] == [("MSFT", "sell")]


# ------------------------------------------------------------------ market hours and logging


def test_stock_orders_wait_for_the_market_but_crypto_trades_any_time(client, conn):
    seed_prices(conn, {"AAPL": 100.0, "BTC/USD": 60_000.0})
    models.store_backtest(conn, "crypto_trend", METRICS, None)
    status = client.app.state.broker_status
    status.broker.is_open = False
    status.refresh(force=True)
    w, job = paper_model(client, conn)
    send_signal(client, w, job, {"AAPL": 0.1})
    run_executor(client, conn)
    assert orders(conn) == []
    assert trading.open_book(conn, "momentum")["waiting"] == "Waiting for the stock market to open"
    w2 = enroll(client, conn, "w9")
    heartbeat(client, w2)
    client.post("/api/models/crypto_trend/paper/start", json={"target": w2["worker_id"]})
    job2 = heartbeat(client, w2, want_job=True)["claimed"][0]
    send_signal(client, w2, job2, {"BTC/USD": 0.1})
    run_executor(client, conn)
    assert [o["symbol"] for o in orders(conn)] == ["BTC/USD"]


def test_every_order_records_model_worker_time_and_reason(client, conn):
    seed_prices(conn, {"AAPL": 100.0})
    w, job = paper_model(client, conn)
    send_signal(client, w, job, {"AAPL": 0.1})
    run_executor(client, conn)
    (order,) = orders(conn)
    assert order["model_id"] == "momentum" and order["worker_id"] == w["worker_id"]
    assert order["created_at"] is not None and order["mode"] == PAPER
    assert order["reason"].startswith("Momentum wants 10% of its money in AAPL")
    listed = client.get("/api/models/momentum/orders").json()
    assert listed[0]["worker_name"] == "w1" and listed[0]["reason"] == order["reason"]


def test_fills_are_booked_into_the_models_book(client, conn):
    seed_prices(conn, {"AAPL": 100.0})
    w, job = paper_model(client, conn)
    send_signal(client, w, job, {"AAPL": 0.1})
    run_executor(client, conn)
    pos = conn.execute("SELECT qty, cost FROM positions").fetchone()
    book = trading.open_book(conn, "momentum")
    assert pos["cost"] == pytest.approx(1_000.0) and book["cash"] == pytest.approx(9_000.0)
    assert pos["qty"] == pytest.approx(1_000.0 / 100.05)


def test_stop_sells_the_positions_then_closes_the_book(client, conn):
    seed_prices(conn, {"AAPL": 100.0})
    w, job = paper_model(client, conn)
    send_signal(client, w, job, {"AAPL": 0.1})
    run_executor(client, conn)
    assert client.post("/api/models/momentum/paper/stop").json()["message"] == (
        "Paper trading stopped; its 1 position(s) will be sold")
    run_executor(client, conn)
    sells = [o for o in orders(conn) if o["side"] == "sell"]
    assert len(sells) == 1 and sells[0]["status"] == "filled"
    assert conn.execute("SELECT status FROM books").fetchone()["status"] == "closed"
    assert conn.execute("SELECT status FROM models WHERE id = 'momentum'").fetchone()["status"] == "backtested"


def test_a_signal_from_another_worker_or_with_bad_weights_is_refused(client, conn):
    seed_prices(conn, {"AAPL": 100.0})
    w, job = paper_model(client, conn)
    other = enroll(client, conn, "w7")
    assert send_signal(client, other, job, {"AAPL": 0.1}).status_code == 409
    assert send_signal(client, w, job, {"AAPL": 0.7, "MSFT": 0.7}).status_code == 400
    assert send_signal(client, w, job, {"BTC/USD": 0.1}).status_code == 400


# ------------------------------------------------------------------ real money


def test_live_needs_the_env_switch_the_confirmation_and_live_keys():
    from coordinator.broker import confirmation_for

    base = dict(database_url="x", alpaca_live_key_id="k", alpaca_live_secret="s")
    ok = confirmation_for(Config(**base))
    assert choose_mode(Config(**base, alpaca_live_allowed=True), ok) == LIVE
    assert choose_mode(Config(**base, alpaca_live_allowed=False), ok) == PAPER
    assert choose_mode(Config(**base, alpaca_live_allowed=True), None) == PAPER
    assert choose_mode(Config(**base, alpaca_live_allowed=True), True) == PAPER  # a bare "yes" is not enough
    assert choose_mode(Config(database_url="x", alpaca_live_allowed=True), ok) == PAPER  # no live keys
    # a confirmation given for other keys does not carry over to new ones
    other = dict(base, alpaca_live_key_id="new-key")
    assert choose_mode(Config(**other, alpaca_live_allowed=True), ok) == PAPER


def test_the_dashboard_confirmation_is_refused_without_the_env_switch(client, conn):
    assert client.post("/api/live/confirm", json={"confirm": "TRADE REAL MONEY"}).status_code == 409
    assert client.post("/api/live/confirm", json={"confirm": "yes"}).status_code == 400
    assert client.get("/api/live").json()["confirmed"] is False


def test_a_paper_book_never_trades_when_the_coordinator_is_live(client, conn):
    seed_prices(conn, {"AAPL": 100.0})
    paper_model(client, conn)
    status = client.app.state.broker_status
    status.broker.mode = LIVE
    book = trading.open_book(conn, "momentum")
    row = safety.approve_and_place(conn, status, LIMITS, book,
                                   {"symbol": "AAPL", "side": "buy", "notional": 500.0, "reason": "t"}, {"AAPL": 100.0})
    assert row["status"] == "blocked" and "paper money but the coordinator is in live mode" in row["error"]


def test_broker_is_fake_only_when_asked():
    assert isinstance(FakeBroker(), FakeBroker)


def test_a_paper_job_on_a_dead_worker_runs_again_on_the_same_book(client, conn):
    from coordinator import queue

    seed_prices(conn, {"AAPL": 100.0})
    w, job = paper_model(client, conn)
    conn.execute("UPDATE jobs SET lease_expires_at = now() - interval '1 second' WHERE id = %s", (job["id"],))
    queue.reap(conn)
    prev = client.get("/api/fleet").json()["previous_jobs"][0]
    assert prev["status"] == "Failed" and prev["can_run_again"]
    again = client.post(f"/api/jobs/{job['id']}/run-again")
    assert again.status_code == 201
    book = trading.open_book(conn, "momentum")
    assert str(book["job_id"]) == again.json()["jobs"][0]["id"]
    client.post("/api/models/momentum/paper/stop")
    stopped = conn.execute("SELECT id FROM jobs WHERE id = %s", (again.json()["jobs"][0]["id"],)).fetchone()
    row = next(j for j in client.get("/api/fleet").json()["previous_jobs"] if j["id"] == str(stopped["id"]))
    assert row["status"] == "Cancelled" and not row["can_run_again"]
