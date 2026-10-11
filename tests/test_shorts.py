"""Short selling in the core (coordinator.safety, coordinator.trading): who may short,
the caps, the rules across books sharing one Alpaca account, and how a short is booked."""
from __future__ import annotations

import types
from dataclasses import replace

import pytest

from coordinator import models, news_trading, safety, trading
from coordinator.broker import AlpacaBroker, AssetInfo, BrokerStatus, BrokerUnavailable, FakeBroker, NoBroker
from coordinator.data import FakeBarSource
from coordinator.errors import BadRequest, Conflict
from coordinator.limits import Limits

LIMITS = Limits()
PRICES = {"AAPL": 100.0, "MSFT": 200.0, "KO": 50.0, "SPY": 500.0, "QQQ": 400.0, "BTC/USD": 60_000.0}


def _book(conn, model_id: str, status: str = "active") -> dict:
    return conn.execute("INSERT INTO books (model_id, mode, starting_balance, cash, status) VALUES (%s, 'paper', 10000,"
                        " 10000, %s) RETURNING *", (model_id, status)).fetchone()


@pytest.fixture
def env(conn):
    news_trading.ensure_models(conn)
    conn.execute("INSERT INTO models (id, name, module, market, description, how_it_works, origin)"
                 " VALUES ('plain', 'Plain', 'momentum', 'stocks', 'd', 'h', 'starter')")
    broker = FakeBroker()
    broker.price_of = lambda s: PRICES[s]
    status = BrokerStatus(broker, ttl=0)
    status.refresh(force=True)
    return types.SimpleNamespace(conn=conn, broker=broker, status=status, news=_book(conn, "news-stocks"),
                                 plain=_book(conn, "plain"))


def place(env, book: dict, limits: Limits = LIMITS, prices: dict | None = None, **order) -> dict:
    fresh = env.conn.execute("SELECT * FROM books WHERE id = %s", (book["id"],)).fetchone()
    row = safety.approve_and_place(env.conn, env.status, limits, fresh, {**order, "reason": "test"}, prices or PRICES)
    trading.poll_fills(env.conn, env.status, book_id=book["id"])
    return env.conn.execute("SELECT * FROM orders WHERE id = %s", (row["id"],)).fetchone()


def held(env, book: dict, symbol: str) -> dict | None:
    return env.conn.execute("SELECT qty, cost FROM positions WHERE book_id = %s AND symbol = %s",
                            (book["id"], symbol)).fetchone()


def cash(env, book: dict) -> float:
    return env.conn.execute("SELECT cash FROM books WHERE id = %s", (book["id"],)).fetchone()["cash"]


# ------------------------------------------------------------------ who may short


def test_a_news_book_shorts_whole_shares_and_the_fill_is_booked_negative(env):
    row = place(env, env.news, symbol="AAPL", side="sell", qty=3.0)
    assert row["status"] == "filled" and row["error"] is None
    fill = 100.0 * (1 - 0.0005)
    pos = held(env, env.news, "AAPL")
    assert pos["qty"] == pytest.approx(-3.0)
    assert pos["cost"] == pytest.approx(-3 * fill)  # cost keeps the sign of qty: cost / qty is the entry price
    assert cash(env, env.news) == pytest.approx(10_000 + 3 * fill)
    # the book's value is cash + qty x price: the short's open loss shows at a higher price
    value, _ = safety.book_value(env.conn, env.news["id"], {"AAPL": 110.0})
    assert cash(env, env.news) + value == pytest.approx(10_000 + 3 * fill - 330.0)
    ex = safety.exposure(env.conn, env.news["id"], {"AAPL": 110.0})
    assert ex["short"] == pytest.approx(330.0) and ex["long"] == 0 and ex["gross"] == pytest.approx(330.0)


def test_extending_a_short_adds_to_it(env):
    place(env, env.news, symbol="AAPL", side="sell", qty=2.0)
    assert place(env, env.news, symbol="AAPL", side="sell", qty=1.0)["status"] == "filled"
    assert held(env, env.news, "AAPL")["qty"] == pytest.approx(-3.0)


@pytest.mark.parametrize("book_name,order,problem", [
    ("plain", {"symbol": "AAPL", "side": "sell", "qty": 1.0}, "nothing to sell: the model holds no AAPL"),
    ("news", {"symbol": "AAPL", "side": "sell", "qty": 1.5}, "shorts are whole shares only, not 1.5"),
    ("news", {"symbol": "AAPL", "side": "sell", "qty": 0.4}, "shorts are whole shares only, not 0.4"),
    ("news", {"symbol": "AAPL", "side": "sell", "qty": 0.0}, "a sale must be for more than 0 shares"),
])
def test_only_news_books_short_and_only_whole_shares(env, book_name, order, problem):
    row = place(env, getattr(env, book_name), **order)
    assert row["status"] == "blocked" and row["error"] == problem


def test_crypto_is_never_shorted(env):
    crypto = _book(env.conn, "news-crypto")
    row = place(env, crypto, symbol="BTC/USD", side="sell", qty=1.0)
    assert row["status"] == "blocked" and row["error"] == "Alpaca cannot short crypto"


@pytest.mark.parametrize("symbol", ["SPY", "QQQ"])
def test_news_books_never_trade_spy_or_qqq_either_way(env, symbol):
    for order in ({"side": "sell", "qty": 1.0}, {"side": "buy", "notional": 100.0}):
        row = place(env, env.news, symbol=symbol, **order)
        assert row["status"] == "blocked" and row["error"].startswith(f"the news strategies never trade {symbol}")
    # the other models still trade them
    assert place(env, env.plain, symbol=symbol, side="buy", notional=100.0)["status"] == "filled"


def test_not_shortable_or_hard_to_borrow_is_refused(env):
    env.broker.assets = {"AAPL": AssetInfo(True, False, True, True), "MSFT": AssetInfo(True, True, False, True)}
    assert place(env, env.news, symbol="AAPL", side="sell", qty=1.0)["error"] == "Alpaca says AAPL cannot be shorted"
    assert place(env, env.news, symbol="MSFT", side="sell", qty=1.0)["error"] == (
        "MSFT is hard to borrow at Alpaca, so it is not shorted")


def test_an_asset_lookup_failure_fails_closed(env):
    env.broker.asset_problem = "Alpaca asset AAPL: read timed out"
    row = place(env, env.news, symbol="AAPL", side="sell", qty=1.0)
    assert row["status"] == "blocked"
    assert row["error"] == "could not check with Alpaca that AAPL can be shorted (Alpaca asset AAPL: read timed out)"
    with pytest.raises(BrokerUnavailable):
        NoBroker().asset_info("AAPL")
    # no broker at all: no short
    fresh = env.conn.execute("SELECT * FROM books WHERE id = %s", (env.news["id"],)).fetchone()
    assert safety.limit_problem(env.conn, fresh, {"symbol": "AAPL", "side": "sell", "qty": 1.0}, PRICES, LIMITS) == (
        "could not check with Alpaca that AAPL can be shorted")


def test_a_short_with_no_price_is_refused(env):
    row = place(env, env.news, prices={"MSFT": 200.0}, symbol="AAPL", side="sell", qty=1.0)
    assert row["error"] == "no current price for AAPL, so the short cannot be checked against the limits"


def test_alpaca_asset_flags_are_read_once_an_hour():
    calls = []

    def get_asset(symbol):
        calls.append(symbol)
        return types.SimpleNamespace(tradable=True, shortable=True, easy_to_borrow=False, fractionable=True)

    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._client = types.SimpleNamespace(get_asset=get_asset)
    assert broker.asset_info("AAPL") == AssetInfo(True, True, False, True)
    broker.asset_info("AAPL")
    assert calls == ["AAPL"]

    def down(symbol):
        raise TimeoutError("read timed out")

    broker._client = types.SimpleNamespace(get_asset=down)
    with pytest.raises(BrokerUnavailable):
        broker.asset_info("MSFT")


# ------------------------------------------------------------------ caps


def test_the_per_position_short_cap(env):
    row = place(env, env.news, symbol="AAPL", side="sell", qty=6.0)  # $600 > $500 + $25
    assert row["error"] == "over the $500 limit per short position"
    place(env, env.news, symbol="AAPL", side="sell", qty=5.0)
    assert place(env, env.news, symbol="AAPL", side="sell", qty=1.0)["error"] == "over the $500 limit per short position"


def test_the_total_short_cap(env):
    limits = replace(LIMITS, max_short_per_model=1_000.0)
    assert place(env, env.news, limits, symbol="AAPL", side="sell", qty=5.0)["status"] == "filled"
    assert place(env, env.news, limits, symbol="MSFT", side="sell", qty=2.0)["status"] == "filled"
    row = place(env, env.news, limits, symbol="KO", side="sell", qty=3.0)  # $1,050: past the $25 band
    assert row["error"] == "over the $1,000 limit for all of a model's shorts"


def test_gross_exposure_counts_longs_and_shorts(env):
    limits = replace(LIMITS, max_per_model=1_000.0, max_short_per_model=1_000.0)
    assert place(env, env.news, limits, symbol="MSFT", side="buy", notional=700.0)["status"] == "filled"
    row = place(env, env.news, limits, symbol="AAPL", side="sell", qty=4.0)
    assert row["error"] == "over the $1,000 limit per model (longs and shorts together)"
    assert place(env, env.news, limits, symbol="AAPL", side="sell", qty=3.0)["status"] == "filled"
    # and a buy counts the shorts too
    row = place(env, env.news, limits, symbol="KO", side="buy", notional=100.0)
    assert row["error"] == "over the $1,000 limit per model"


# ------------------------------------------------------------------ one account, many books


def test_no_short_while_another_book_holds_the_symbol(env):
    place(env, env.plain, symbol="AAPL", side="buy", notional=300.0)
    row = place(env, env.news, symbol="AAPL", side="sell", qty=1.0)
    assert row["error"] == "Plain holds AAPL, and Alpaca keeps one position per symbol: no short until it is sold"
    env.conn.execute("UPDATE books SET status = 'closing' WHERE id = %s", (env.plain["id"],))
    assert place(env, env.news, symbol="AAPL", side="sell", qty=1.0)["status"] == "blocked"  # closing books count


def test_no_short_while_another_book_has_a_buy_in_flight(env):
    env.conn.execute("INSERT INTO orders (book_id, model_id, mode, symbol, side, notional, reason, status)"
                     " VALUES (%s, 'plain', 'paper', 'AAPL', 'buy', 300, 't', 'submitted')", (env.plain["id"],))
    row = place(env, env.news, symbol="AAPL", side="sell", qty=1.0)
    assert row["error"] == "Plain has a buy of AAPL in flight; no short until it settles"


def test_no_model_buys_a_symbol_another_book_is_short(env):
    place(env, env.news, symbol="AAPL", side="sell", qty=2.0)
    row = place(env, env.plain, symbol="AAPL", side="buy", notional=300.0)
    assert row["status"] == "blocked" and row["error"].startswith("News strategy: stocks is short AAPL")
    assert place(env, env.plain, symbol="MSFT", side="buy", notional=300.0)["status"] == "filled"


def test_no_model_buys_while_a_short_sale_is_in_flight(env):
    env.conn.execute("INSERT INTO orders (book_id, model_id, mode, symbol, side, qty, reason, status)"
                     " VALUES (%s, 'news-stocks', 'paper', 'AAPL', 'sell', 2, 't', 'submitted')", (env.news["id"],))
    row = place(env, env.plain, symbol="AAPL", side="buy", notional=300.0)
    assert row["error"] == "News strategy: stocks has a short sale of AAPL in flight; no model buys it until that settles"


# ------------------------------------------------------------------ order meanings


def test_a_sale_cannot_cross_from_long_to_short(env):
    place(env, env.news, symbol="AAPL", side="buy", notional=500.0)
    row = place(env, env.news, symbol="AAPL", side="sell", qty=6.0)
    assert row["status"] == "blocked" and "would cross from long to short in one order" in row["error"]
    qty = held(env, env.news, "AAPL")["qty"]
    assert place(env, env.news, symbol="AAPL", side="sell", qty=qty)["status"] == "filled"
    assert held(env, env.news, "AAPL") is None


def test_a_dollar_buy_while_short_is_refused(env):
    place(env, env.news, symbol="AAPL", side="sell", qty=2.0)
    row = place(env, env.news, symbol="AAPL", side="buy", notional=100.0)
    assert row["error"] == "cover the short first: the model is short 2 AAPL"


def test_cover_by_shares(env):
    place(env, env.news, symbol="AAPL", side="sell", qty=3.0)
    before = cash(env, env.news)
    assert place(env, env.news, symbol="AAPL", side="buy", qty=2.0)["status"] == "filled"
    fill_in, fill_out = 100.0 * (1 - 0.0005), 100.0 * (1 + 0.0005)
    pos = held(env, env.news, "AAPL")
    assert pos["qty"] == pytest.approx(-1.0) and pos["cost"] == pytest.approx(-fill_in)
    assert cash(env, env.news) == pytest.approx(before - 2 * fill_out)
    assert place(env, env.news, symbol="AAPL", side="buy", qty=2.0)["error"] == "buy of 2 is more than the 1 AAPL short"
    assert place(env, env.news, symbol="AAPL", side="buy", qty=1.0)["status"] == "filled"
    assert held(env, env.news, "AAPL") is None
    # round trip: the book lost exactly the spread
    assert cash(env, env.news) == pytest.approx(10_000 + 3 * fill_in - 3 * fill_out)
    row = place(env, env.plain, symbol="AAPL", side="buy", qty=1.0)
    assert row["error"] == "a buy by shares only covers a short, and the model is not short AAPL"


def test_close_if_flat_waits_for_shorts(env):
    place(env, env.news, symbol="AAPL", side="sell", qty=2.0)
    env.conn.execute("UPDATE books SET status = 'closing' WHERE id = %s", (env.news["id"],))
    assert not trading.close_if_flat(env.conn, env.news["id"])
    place(env, env.news, symbol="AAPL", side="buy", qty=2.0)
    assert trading.close_if_flat(env.conn, env.news["id"])


# ------------------------------------------------------------------ the existing safety still applies


def test_pause_and_the_daily_loss_limit_block_shorts(env):
    safety.pause_trading(env.conn, "owner", "Paused by you")
    assert place(env, env.news, symbol="AAPL", side="sell", qty=1.0)["error"] == safety.PAUSED
    safety.resume_trading(env.conn, "owner")
    env.broker.equity, env.broker.last_equity = 97_000.0, 100_000.0
    row = place(env, env.news, symbol="AAPL", side="sell", qty=1.0)
    assert row["status"] == "blocked" and row["error"].startswith("Daily loss limit reached")


def test_the_market_must_be_open_for_a_short(env):
    env.broker.is_open = False
    assert place(env, env.news, symbol="AAPL", side="sell", qty=1.0)["error"] == safety.MARKET_CLOSED


# ------------------------------------------------------------------ news models stay out of the models' paths


def test_execute_pass_leaves_news_books_alone(env):
    place(env, env.news, symbol="AAPL", side="sell", qty=2.0)
    env.conn.execute("UPDATE books SET status = 'closing', targets = '{}'::jsonb, targets_bar_t = 5 WHERE id = %s",
                     (env.news["id"],))
    assert trading.execute_pass(env.conn, env.status, LIMITS) == 0
    assert held(env, env.news, "AAPL")["qty"] == pytest.approx(-2.0)
    assert env.conn.execute("SELECT status FROM books WHERE id = %s", (env.news["id"],)).fetchone()["status"] == "closing"


def test_news_models_are_hidden_and_refuse_model_actions(env):
    ids = {m["id"] for m in models.list_models(env.conn, include_retired=True)}
    assert "news-stocks" not in ids and "news-crypto" not in ids and "plain" in ids
    assert {"news-stocks", "news-crypto"} <= {m["id"] for m in models.list_models(env.conn, include_news=True)}
    env.conn.execute("UPDATE books SET status = 'closed' WHERE id = %s", (env.news["id"],))
    with pytest.raises(Conflict):
        trading.start(env.conn, "news-stocks", "paper", LIMITS)
    with pytest.raises(Conflict):
        trading.stop(env.conn, "news-stocks")
    with pytest.raises(BadRequest):
        models.job_params(env.conn, "backtest", "news-crypto", {}, LIMITS)


def test_news_models_are_not_on_the_models_screen_or_in_assign_a_job(client, conn):
    news_trading.ensure_models(conn)
    assert "news-stocks" not in {m["id"] for m in client.get("/api/models").json()}
    assert "news-stocks" not in {m["id"] for m in client.get("/api/fleet").json()["assign"]["models"]}
    assert "News strategy" not in client.get("/models?id=news-stocks").text
    resp = client.post("/api/jobs", json={"kind": "backtest", "model_id": "news-stocks", "target": "auto"})
    assert resp.status_code == 400
    resp = client.post("/api/models/news-stocks/paper/start", json={"target": "auto"})
    assert resp.status_code == 409


# ------------------------------------------------------------------ latest prices


def test_the_fake_bar_source_gives_latest_prices():
    source = FakeBarSource(seed=7)
    prices = source.latest_prices(["AAPL", "BTC/USD"])
    assert set(prices) == {"AAPL", "BTC/USD"} and all(p > 0 for p in prices.values())
    assert source.latest_prices([]) == {}
