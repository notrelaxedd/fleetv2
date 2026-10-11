"""The news strategies' executor (coordinator.news_trading) end to end with the fake
broker: entries, filters, exits, stops, stopping a strategy and the dashboard summary."""
from __future__ import annotations

import itertools
import types
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from coordinator import news_scoring, news_trading, safety
from coordinator.broker import BrokerStatus, BrokerUnavailable, FakeBroker
from coordinator.errors import Conflict
from coordinator.limits import Limits

LIMITS = Limits()
_ids = itertools.count(1)


@pytest.fixture
def env(conn, monkeypatch):
    news_trading.ensure_models(conn)
    monkeypatch.setattr(news_scoring, "gate", lambda c, market, limits: (True, "its predictions beat the coin flip"))
    prices = {"AAPL": 100.0, "MSFT": 200.0, "KO": 50.0, "JPM": 150.0, "SPY": 500.0, "BTC/USD": 60_000.0,
              "ETH/USD": 3_000.0}
    broker = FakeBroker()
    broker.price_of = lambda s: prices[s]
    status = BrokerStatus(broker, ttl=0)
    status.refresh(force=True)
    e = types.SimpleNamespace(conn=conn, broker=broker, status=status, prices=prices, now=datetime.now(timezone.utc))
    e.price_now = lambda symbols: {s: prices[s] for s in symbols if s in prices}
    return e


def start(env, market="stocks"):
    return news_trading.start(env.conn, market, "paper", LIMITS)


def predict(env, symbol, direction, move=3.0, confidence=0.8, horizon=3, priced_in=False, ago_min=5.0,
            market=None, source="live", headline=None):
    """One article, one analysis and one prediction about `symbol`."""
    market = market or ("crypto" if "/" in symbol else "stocks")
    news_id = 1_000 + next(_ids)
    decided = env.now - timedelta(minutes=ago_min)
    env.conn.execute("INSERT INTO news (id, headline, symbols, ours, created_at) VALUES (%s, %s, %s, %s, %s)",
                     (news_id, headline or f"Big news about {symbol}", [symbol.replace("/", "")], [symbol], decided))
    analysis = env.conn.execute(
        "INSERT INTO news_analyses (news_id, source, requested_model, served_model, prompt_version, context)"
        " VALUES (%s, %s, 'claude-opus-5-5', 'claude-opus-5-5', 'v1', '{}'::jsonb) RETURNING id",
        (news_id, source)).fetchone()
    return env.conn.execute(
        """
        INSERT INTO news_predictions (analysis_id, news_id, source, model, symbol, market, direction, expected_move_pct,
                                      horizon_days, confidence, already_priced_in, decided_at)
        VALUES (%s, %s, %s, 'claude-opus-5-5', %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *
        """,
        (analysis["id"], news_id, source, symbol, market, direction, move, horizon, confidence, priced_in, decided),
    ).fetchone()


def run(env, limits=LIMITS, at=None):
    return news_trading.news_pass(env.conn, env.status, limits, env.price_now, at or env.now)


def positions(env):
    return env.conn.execute("SELECT * FROM news_positions ORDER BY id").fetchall()


def orders(env):
    return env.conn.execute("SELECT * FROM orders ORDER BY created_at").fetchall()


def book(env, market="stocks"):
    return env.conn.execute("SELECT * FROM books WHERE model_id = %s ORDER BY id DESC LIMIT 1",
                            (news_trading.NEWS_MODELS[market],)).fetchone()


def held(env, symbol):
    row = env.conn.execute("SELECT qty FROM positions WHERE symbol = %s", (symbol,)).fetchone()
    return row["qty"] if row else 0.0


# ------------------------------------------------------------------ start


def test_start_waits_for_the_gate(env, monkeypatch):
    monkeypatch.setattr(news_scoring, "gate", lambda c, m, l: (False, "Only 12 scored predictions so far; 100 needed"))
    with pytest.raises(Conflict, match="Only 12 scored predictions"):
        start(env)
    assert book(env) is None
    monkeypatch.setattr(news_scoring, "gate", lambda c, m, l: (True, "ok"))
    b = start(env)
    assert b["status"] == "active" and b["cash"] == LIMITS.starting_balance_per_model and b["job_id"] is None
    assert env.conn.execute("SELECT status FROM models WHERE id = 'news-stocks'").fetchone()["status"] == "paper_trading"
    with pytest.raises(Conflict, match="already trading"):
        start(env)


# ------------------------------------------------------------------ entries


def test_an_up_prediction_opens_a_long(env):
    start(env)
    p = predict(env, "AAPL", "up", move=3.0, confidence=0.8, horizon=3, headline="Apple beats on earnings")
    assert run(env) == 1
    (o,) = orders(env)
    assert o["side"] == "buy" and o["notional"] == LIMITS.max_per_position and o["status"] == "filled"
    assert o["reason"] == ("News: Claude expects AAPL to rise 3.0% within 3 days (confidence 0.80): "
                           "Apple beats on earnings")
    (np,) = positions(env)
    assert np["status"] == "open" and np["side"] == "long" and np["prediction_id"] == p["id"]
    assert np["entry_price"] == pytest.approx(100.05)
    assert np["stop_price"] == pytest.approx(100.05 * 0.95)
    assert np["qty"] == pytest.approx(1_000.0 / 100.05) == pytest.approx(held(env, "AAPL"))
    assert np["exit_due_at"] == news_trading.horizon_end("stocks", env.now, 3)
    assert book(env)["waiting"] is None


def test_a_down_prediction_opens_a_whole_share_short(env):
    start(env)
    predict(env, "MSFT", "down", move=2.5, confidence=0.72, horizon=3)
    assert run(env) == 1
    (o,) = orders(env)
    assert o["side"] == "sell" and o["qty"] == 2.0  # floor($500 / $200)
    assert o["reason"].startswith("News: Claude expects MSFT to fall 2.5% within 3 days (confidence 0.72): ")
    (np,) = positions(env)
    assert np["side"] == "short" and np["status"] == "open" and np["qty"] == 2.0
    assert np["stop_price"] == pytest.approx(np["entry_price"] * 1.05)
    assert held(env, "MSFT") == -2.0


def test_crypto_down_never_shorts_but_closes_a_held_coin(env):
    start(env, "crypto")
    predict(env, "ETH/USD", "down", move=5.0)
    assert run(env) == 0 and orders(env) == []
    predict(env, "BTC/USD", "up", move=3.0, ago_min=10)
    run(env)
    assert held(env, "BTC/USD") > 0
    predict(env, "BTC/USD", "down", move=3.0, ago_min=1)
    run(env)
    sells = [o for o in orders(env) if o["side"] == "sell"]
    assert len(sells) == 1 and sells[0]["symbol"] == "BTC/USD" and sells[0]["status"] == "filled"
    assert held(env, "BTC/USD") == 0
    btc = [p for p in positions(env) if p["symbol"] == "BTC/USD"]
    assert len(btc) == 1 and btc[0]["status"] == "closed" and btc[0]["close_reason"].startswith("a newer prediction")


@pytest.mark.parametrize("kwargs", [
    {"symbol": "AAPL", "direction": "up", "confidence": 0.5},            # not confident enough
    {"symbol": "AAPL", "direction": "up", "priced_in": True},            # already in the price
    {"symbol": "AAPL", "direction": "up", "move": 0.15},                 # under 2 x 0.10% costs
    {"symbol": "AAPL", "direction": "up", "ago_min": 40},                # stale
    {"symbol": "AAPL", "direction": "up", "source": "replay"},           # replays never trade
    {"symbol": "AAPL", "direction": "none"},
    {"symbol": "SPY", "direction": "up"},                                # left to the futures books
    {"symbol": "SPY", "direction": "down"},
    {"symbol": "BTC/USD", "direction": "up"},                            # another market's book
])
def test_predictions_that_are_not_traded(env, kwargs):
    start(env)
    predict(env, **kwargs)
    assert run(env) == 0
    assert orders(env) == [] and positions(env) == []


def test_crypto_needs_a_bigger_move_for_its_costs(env):
    start(env, "crypto")
    assert news_trading.round_trip_cost_pct("crypto") == pytest.approx(0.70)
    assert news_trading.round_trip_cost_pct("stocks") == pytest.approx(0.10)
    predict(env, "BTC/USD", "up", move=1.3)  # under 2 x 0.70%
    assert run(env) == 0
    predict(env, "ETH/USD", "up", move=1.5)
    assert run(env) == 1


def test_a_pass_is_idempotent(env):
    start(env)
    predict(env, "AAPL", "up")
    assert run(env) == 1
    assert run(env) == 0
    assert len(orders(env)) == 1 and len(positions(env)) == 1


def test_one_position_per_symbol_and_the_same_view_holds_it_longer(env):
    start(env)
    predict(env, "AAPL", "up", horizon=1, ago_min=20)
    run(env)
    due = positions(env)[0]["exit_due_at"]
    predict(env, "AAPL", "up", horizon=5, ago_min=1)
    assert run(env) == 0
    (np,) = positions(env)
    assert np["exit_due_at"] > due
    assert np["exit_due_at"] == news_trading.horizon_end("stocks", env.now - timedelta(minutes=1), 5)


def test_at_most_max_trades_per_day(env):
    start(env)
    limits = replace(LIMITS, max_trades_per_day=2.0)
    for symbol in ("AAPL", "MSFT", "KO"):
        predict(env, symbol, "up")
    assert run(env, limits) == 2
    assert len(positions(env)) == 2
    assert book(env)["waiting"].startswith("Done for today: 2 new news positions")


def test_the_stock_book_waits_for_the_market_and_while_paused(env):
    start(env)
    predict(env, "AAPL", "up")
    env.broker.is_open = False
    env.status.refresh(force=True)
    assert run(env) == 0 and book(env)["waiting"] == "Waiting for the stock market to open"
    env.broker.is_open = True
    env.status.refresh(force=True)
    safety.pause_trading(env.conn, "owner", "Paused by you")
    assert run(env) == 0 and book(env)["waiting"] == "Waiting: trading is paused"
    safety.resume_trading(env.conn, "owner")
    assert run(env) == 1


# ------------------------------------------------------------------ exits


def test_a_position_closes_when_its_horizon_has_passed(env):
    start(env)
    predict(env, "AAPL", "up", horizon=2)
    run(env)
    assert run(env, at=env.now + timedelta(days=1)) == 0  # not yet
    env.prices["AAPL"] = 104.0
    assert run(env, at=env.now + timedelta(days=5)) == 1
    (np,) = positions(env)
    assert np["status"] == "closed" and np["close_reason"] == "its 2-day horizon has passed"
    sell = orders(env)[-1]
    assert sell["side"] == "sell" and sell["reason"] == "News: closing the AAPL long because its 2-day horizon has passed"
    assert np["pnl"] == pytest.approx(1_000.0 / 100.05 * 103.948 - 1_000.0)
    assert held(env, "AAPL") == 0
    assert book(env)["cash"] == pytest.approx(LIMITS.starting_balance_per_model + np["pnl"])


def test_the_stop_closes_a_long(env):
    start(env)
    predict(env, "AAPL", "up")
    run(env)
    env.prices["AAPL"] = 96.0
    assert run(env) == 0  # above the stop at 95.05
    env.prices["AAPL"] = 95.0
    assert run(env) == 1
    (np,) = positions(env)
    assert np["status"] == "closed" and np["close_reason"].startswith("the price fell to 95.00, through its stop")
    assert np["pnl"] < 0


def test_the_stop_closes_a_short_by_buying_the_shares_back(env):
    start(env)
    predict(env, "MSFT", "down")
    run(env)
    env.prices["MSFT"] = 211.0  # stop at 199.9 x 1.05 = 209.9
    assert run(env) == 1
    cover = orders(env)[-1]
    assert cover["side"] == "buy" and cover["qty"] == 2.0 and cover["notional"] is None
    (np,) = positions(env)
    assert np["status"] == "closed" and np["close_reason"].startswith("the price rose to 211.00")
    assert np["pnl"] == pytest.approx(2 * 200.0 * (1 - 0.0005) - 2 * 211.0 * (1 + 0.0005))
    assert held(env, "MSFT") == 0


def test_no_price_means_no_stop_check_and_no_entry(env):
    start(env)
    predict(env, "AAPL", "up")
    env.price_now = lambda symbols: {}
    assert run(env) == 0
    assert book(env)["waiting"] == "No current price for AAPL; its prediction waits"

    def broken(symbols):
        raise RuntimeError("Alpaca data: 503")

    env.price_now = broken
    assert run(env) == 0  # never raises


def test_a_newer_opposite_prediction_closes_and_turns_the_position(env):
    start(env)
    predict(env, "AAPL", "up", ago_min=20)
    run(env)
    predict(env, "AAPL", "down", confidence=0.5, ago_min=10)  # not confident enough to count
    run(env)
    assert [p["status"] for p in positions(env)] == ["open"]
    predict(env, "AAPL", "down", move=4.0, confidence=0.75, ago_min=1)
    run(env)
    first, second = positions(env)
    assert first["status"] == "closed"
    assert first["close_reason"] == "a newer prediction expects it to fall 4.0% (confidence 0.75)"
    assert second["side"] == "short" and second["status"] == "open" and held(env, "AAPL") == -5.0


def test_stop_closes_every_position_then_the_book(env):
    start(env)
    predict(env, "AAPL", "up")
    predict(env, "MSFT", "down")
    run(env)
    assert held(env, "AAPL") > 0 and held(env, "MSFT") < 0
    news_trading.stop(env.conn, "stocks")
    assert env.conn.execute("SELECT status FROM models WHERE id = 'news-stocks'").fetchone()["status"] is None
    predict(env, "KO", "up")  # a stopping book opens nothing
    assert run(env) == 2
    assert {p["status"] for p in positions(env)} == {"closed"}
    assert all(p["close_reason"] == "the news strategy was stopped" for p in positions(env))
    assert book(env)["status"] == "closed"
    assert env.conn.execute("SELECT count(*) AS n FROM positions").fetchone()["n"] == 0
    with pytest.raises(Conflict):
        news_trading.stop(env.conn, "stocks")


# ------------------------------------------------------------------ blocked and refused orders


def test_a_blocked_opening_order_fails_the_position_with_the_reason(env):
    conn = env.conn
    conn.execute("INSERT INTO models (id, name, module, market, description, how_it_works)"
                 " VALUES ('plain', 'Plain', 'momentum', 'stocks', 'd', 'h')")
    plain = conn.execute("INSERT INTO books (model_id, mode, starting_balance, cash) VALUES ('plain', 'paper', 1e4, 1e4)"
                         " RETURNING id").fetchone()
    conn.execute("INSERT INTO positions (book_id, symbol, qty, cost) VALUES (%s, 'MSFT', 2, 400)", (plain["id"],))
    start(env)
    predict(env, "MSFT", "down")
    assert run(env) == 0
    (np,) = positions(env)
    assert np["status"] == "failed"
    assert np["close_reason"] == ("Blocked: Plain holds MSFT, and Alpaca keeps one position per symbol: "
                                  "no short until it is sold")
    assert run(env) == 0 and len(orders(env)) == 1  # tried once, not every pass


def test_a_refused_opening_order_fails_the_position(env):
    start(env)
    env.broker.refuse = "Alpaca refused the order: insufficient buying power"
    predict(env, "AAPL", "up")
    run(env)
    (np,) = positions(env)
    assert np["status"] == "failed" and np["close_reason"] == "Alpaca refused the order: insufficient buying power"
    assert held(env, "AAPL") == 0


def test_an_order_blocked_for_now_is_tried_again_while_fresh(env):
    start(env)
    predict(env, "AAPL", "up")
    real = env.broker.account

    def down():
        raise BrokerUnavailable("Alpaca account: 503")

    env.broker.account = down
    assert run(env) == 0
    (np,) = positions(env)
    assert np["status"] == "failed" and np["close_reason"] == "Blocked: " + safety.NO_ACCOUNT
    assert book(env)["waiting"] == "Waiting: " + safety.NO_ACCOUNT
    env.broker.account = real
    assert run(env) == 1
    assert [p["status"] for p in positions(env)] == ["failed", "open"]


def test_a_position_left_opening_by_a_crash_is_resolved(env):
    b = start(env)
    p = predict(env, "AAPL", "up")
    env.conn.execute("INSERT INTO news_positions (book_id, prediction_id, symbol, side, opened_at)"
                     " VALUES (%s, %s, 'AAPL', 'long', %s)", (b["id"], p["id"], env.now))
    assert run(env) == 0
    (np,) = positions(env)
    assert np["status"] == "failed" and np["close_reason"] == "the coordinator stopped before the order was sent"


# ------------------------------------------------------------------ dashboard


def test_book_summary_and_open_positions(env):
    summary = news_trading.book_summary(env.conn)
    keys = {"model_id", "status", "book_id", "cash", "equity", "gross_exposure", "short_exposure", "open_positions",
            "trades_today", "waiting"}
    assert set(summary) == {"stocks", "crypto"}
    assert keys <= set(summary["stocks"]) and summary["stocks"]["status"] == "not_started"
    start(env)
    predict(env, "AAPL", "up", headline="Apple beats")
    predict(env, "MSFT", "down")
    run(env)
    s = news_trading.book_summary(env.conn, prices=env.prices, now=env.now)["stocks"]
    assert s["status"] == "active" and s["model_id"] == "news-stocks" and s["open_positions"] == 2
    assert s["trades_today"] == 2 and s["short_exposure"] == pytest.approx(400.0)
    assert s["gross_exposure"] == pytest.approx(1_000.0 / 100.05 * 100.0 + 400.0)
    assert s["equity"] == pytest.approx(s["cash"] + 1_000.0 / 100.05 * 100.0 - 400.0)
    rows = news_trading.open_positions(env.conn, "stocks", prices=env.prices)
    assert {r["symbol"] for r in rows} == {"AAPL", "MSFT"}
    msft = next(r for r in rows if r["symbol"] == "MSFT")
    assert msft["market"] == "stocks" and msft["value"] == pytest.approx(-400.0) and msft["headline"]
    assert msft["open_pnl"] == pytest.approx(2 * (msft["entry_price"] - 200.0))
    assert news_trading.open_positions(env.conn, "crypto") == []


def test_horizon_end_counts_weekdays_for_stocks():
    friday = datetime(2026, 10, 9, 15, 0, tzinfo=timezone.utc)
    assert news_trading.horizon_end("stocks", friday, 1) == datetime(2026, 10, 12, 15, 0, tzinfo=timezone.utc)
    assert news_trading.horizon_end("stocks", friday, 3) == datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)
    assert news_trading.horizon_end("crypto", friday, 2) == friday + timedelta(hours=48)


def test_ensure_models_is_repeatable_and_keeps_the_status(conn):
    news_trading.ensure_models(conn)
    conn.execute("UPDATE models SET status = 'paper_trading' WHERE id = 'news-crypto'")
    news_trading.ensure_models(conn)
    rows = {r["id"]: r for r in conn.execute("SELECT * FROM models WHERE origin = 'news'").fetchall()}
    assert set(rows) == {"news-stocks", "news-crypto"}
    assert rows["news-crypto"]["status"] == "paper_trading" and rows["news-stocks"]["module"] == "news"
    assert rows["news-stocks"]["market"] == "stocks" and rows["news-crypto"]["market"] == "crypto"


def test_everything_is_committed_on_a_pooled_connection(env, test_db_url):
    """The loop's connections are not autocommit: every step must commit its own work."""
    import psycopg
    from psycopg.rows import dict_row

    start(env)
    predict(env, "AAPL", "up")
    predict(env, "MSFT", "down")
    with psycopg.connect(test_db_url, row_factory=dict_row) as c:
        c.execute("SET TIME ZONE 'UTC'")
        c.commit()
        assert news_trading.news_pass(c, env.status, LIMITS, env.price_now, env.now) == 2
        c.rollback()  # anything left uncommitted would be lost here
        news_trading.stop(c, "stocks")
        c.commit()
        assert news_trading.news_pass(c, env.status, LIMITS, env.price_now, env.now) == 2
        c.rollback()  # anything left uncommitted would be lost here
    assert [p["status"] for p in positions(env)] == ["closed", "closed"]
    assert book(env)["status"] == "closed"
