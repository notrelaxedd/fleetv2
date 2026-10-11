"""The News screen: status line, scores, paper trading buttons, the article feed, escaping, owner-only routes.

Tests look for data-* hooks (not CSS classes) in the rendered HTML. coordinator.news_trading is
written separately, so these tests replace its functions with monkeypatch (a stand-in module is
installed when the real one is missing)."""
from __future__ import annotations

import re
import sys
import types
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from html import unescape
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from psycopg.types.json import Jsonb

import coordinator
from coordinator import haiku
from coordinator.api.app import create_app
from coordinator.broker import BrokerStatus
from tests.test_dashboard import attr_tags, element, text

UTC = timezone.utc
HOSTILE = "<script>alert('headline')</script>"
HOSTILE_WHY = "<img src=x onerror=alert('why')>"


def summary(**over):
    base = {"model_id": "news-stocks", "status": "not_started", "book_id": None, "cash": 0.0, "equity": 0.0,
            "gross_exposure": 0.0, "short_exposure": 0.0, "open_positions": 0, "trades_today": 0, "waiting": ""}
    return {"stocks": {**base, **over.get("stocks", {})}, "crypto": {**base, "model_id": "news-crypto", **over.get("crypto", {})}}


@pytest.fixture
def trading(monkeypatch):
    """coordinator.news_trading with recording stand-ins for the functions the screen uses."""
    module = sys.modules.get("coordinator.news_trading")
    if module is None:
        try:
            from coordinator import news_trading as module
        except ImportError:
            module = types.ModuleType("coordinator.news_trading")
            monkeypatch.setitem(sys.modules, "coordinator.news_trading", module)
            monkeypatch.setattr(coordinator, "news_trading", module, raising=False)
    calls: list[tuple] = []
    state = {"summary": summary(), "positions": []}
    monkeypatch.setattr(module, "book_summary", lambda conn: state["summary"], raising=False)
    monkeypatch.setattr(module, "open_positions", lambda conn, market=None, prices=None: state["positions"], raising=False)
    monkeypatch.setattr(module, "start", lambda conn, market, mode, limits: calls.append(("start", market, mode)) or {"book_id": 1}, raising=False)
    monkeypatch.setattr(module, "stop", lambda conn, market: calls.append(("stop", market)) or {"book_id": 1}, raising=False)
    return SimpleNamespace(calls=calls, state=state)


def add_article(conn, id=1, headline="Apple unveils a new chip", ours=("AAPL",), status="pending", url="https://example.com/a",
                created=None, note=None):
    conn.execute(
        "INSERT INTO news (id, headline, symbols, ours, created_at, url, source, analysis_status, analysis_note)"
        " VALUES (%s, %s, %s, %s, %s, %s, 'Benzinga', %s, %s)",
        (id, headline, list(ours), list(ours), created or datetime.now(UTC) - timedelta(minutes=5), url, status, note))


def add_analysis(conn, news_id=1, source="live") -> int:
    return conn.execute(
        "INSERT INTO news_analyses (news_id, source, requested_model, served_model, prompt_version, context)"
        " VALUES (%s, %s, 'claude-opus-5-5', 'claude-opus-5-5', 'v1', %s) RETURNING id",
        (news_id, source, Jsonb({}))).fetchone()["id"]


def add_prediction(conn, analysis_id, news_id=1, symbol="AAPL", direction="up", rationale="Strong demand", source="live",
                   market="stocks", priced_in=False, **scores) -> int:
    cols = {"analysis_id": analysis_id, "news_id": news_id, "source": source, "model": "claude-opus-5-5", "symbol": symbol,
            "market": market, "direction": direction, "expected_move_pct": 2.5, "horizon_days": 3, "confidence": 0.7,
            "already_priced_in": priced_in, "rationale": rationale, "decided_at": datetime.now(UTC) - timedelta(days=10), **scores}
    return conn.execute(f"INSERT INTO news_predictions ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) RETURNING id",
                        list(cols.values())).fetchone()["id"]


def scored(conn, aid, hit, signed, twin, **kw):
    kw.setdefault("twin_direction", "down")
    return add_prediction(conn, aid, hit=hit, signed_pct=signed, twin_signed_pct=twin, move_pct=signed, bench_pct=0.0,
                          entry_price=100.0, exit_price=101.0, scored_at=datetime.now(UTC), **kw)


def tag_of(html: str, hook: str) -> str:
    return attr_tags(html, hook)[0]


def card(html: str, hook: str) -> str:
    """The <article> that carries the hook (a status card, a metric card, ...), as text."""
    return text(re.search(rf"{re.escape(hook)}.*?</article>", html, re.S).group(0))


def value(html: str, hook: str) -> str:
    return element(re.search(rf"{re.escape(hook)}.*?</article>", html, re.S).group(0), "data-value")


# ------------------------------------------------------------------ pages and nav


def test_the_page_renders_on_an_empty_database(client, trading):
    r = client.get("/news")
    assert r.status_code == 200
    html = r.text
    assert 'data-nav="news" class="current" aria-current="page"' in html
    assert element(html, 'data-nav="news"') == "News"
    for key in ("feed", "analyst", "spend-today", "spend-month", "articles", "waiting"):
        assert f'data-status-card="{key}"' in html
    assert "No articles yet" in element(html, "data-no-news")
    assert "No open positions" in element(html, "data-no-positions")
    assert element(html, "data-demo") == "demo data"  # the test broker is the fake one
    assert "Needs 100 independent scored calls it would trade; has 0." in element(html, 'data-score="stocks"')


def test_the_fragment_has_the_news_regions(client, trading):
    r = client.get("/fragments/news")
    assert r.status_code == 200
    regions = re.findall(r'data-region="([\w-]+)"', r.text)
    assert regions == ["status", "banner", "news-status", "news-scores", "news-paper", "news-feed"]
    assert "<html" not in r.text


def test_every_screen_links_to_news(client):
    for path in ("/fleet", "/models"):
        html = client.get(path).text
        assert element(html, 'data-nav="news"') == "News"
        assert 'data-nav="news" class="current"' not in html


def test_the_pages_need_the_owner_login(config, broker, trading):
    strict = replace(config, dev=False, owner_login="owner@example.com")
    status = BrokerStatus(broker, ttl=0)
    status.refresh(force=True)
    login = {"Tailscale-User-Login": "owner@example.com"}
    with TestClient(create_app(strict, status)) as c:
        assert c.get("/news").status_code == 401
        assert c.get("/fragments/news").status_code == 401
        assert c.get("/news", headers=login).status_code == 200
        assert c.post("/api/news/stocks/start").status_code == 401
        assert c.post("/api/news/stocks/stop").status_code == 401
        assert c.post("/api/news/stocks/start", headers=login).status_code == 201
        assert c.post("/api/news/stocks/stop", headers=login).status_code == 200
    assert [c[0] for c in trading.calls] == ["start", "stop"]


# ------------------------------------------------------------------ status line


def test_missing_feed_and_analyst_say_not_set_up(client, trading):
    del client.app.state.news_feed, client.app.state.analyst
    html = client.get("/news").text
    assert "Not set up" in card(html, 'data-status-card="feed"') and "Not set up" in card(html, 'data-status-card="analyst"')


def test_feed_and_analyst_state_and_problem(client, trading):
    state = client.app.state
    state.news_feed = SimpleNamespace(connected=False, problem="Alpaca keys are missing")
    state.analyst = SimpleNamespace(connected=True, problem=None, model="claude-opus-5-5")
    html = client.get("/news").text
    feed, analyst = card(html, 'data-status-card="feed"'), card(html, 'data-status-card="analyst"')
    assert "Not connected" in feed and "Alpaca keys are missing" in feed
    assert "Ready" in analyst and "Model: claude-opus-5-5" in analyst


def test_spend_today_and_this_month_against_their_caps(client, conn, trading):
    haiku.record_call(conn, "news", "claude-opus-5-5", 1000, 500, 1.25, "ok")
    haiku.record_call(conn, "market_note", "claude-haiku", 1000, 500, 0.50, "ok")
    ai = client.app.state.ai
    html = client.get("/news").text
    today, month = card(html, 'data-status-card="spend-today"'), card(html, 'data-status-card="spend-month"')
    assert f"$1.25 of ${ai.news_daily_cap_usd:,.2f}" in today  # only the news calls
    assert f"$1.75 of ${ai.monthly_cap_usd:,.2f}" in month  # every feature
    assert "Over the cap" not in today


def test_a_cap_that_is_used_up_is_flagged(client, conn, trading):
    haiku.record_call(conn, "news", "claude-opus-5-5", 1000, 500, client.app.state.ai.news_daily_cap_usd, "ok")
    html = client.get("/news").text
    card = re.search(r'data-status-card="spend-today".*?</article>', html, re.S).group(0)
    assert 'data-tone="warn"' in card and "Over the cap" in text(card)


def test_articles_today_and_waiting_counts(client, conn, trading):
    add_article(conn, 1, status="pending")
    add_article(conn, 2, status="analysed")
    add_article(conn, 3, status="pending", created=datetime.now(UTC) - timedelta(days=3))
    html = client.get("/news").text
    assert value(html, 'data-status-card="articles"') == "2"  # the third article is three days old
    assert value(html, 'data-status-card="waiting"') == "2"  # articles 1 and 3 are pending


def test_a_switched_off_analyst_is_said_plainly(client, trading):
    client.app.state.ai = replace(client.app.state.ai, news_enabled=False)
    assert "switched off" in element(client.get("/news").text, "data-news-off")


def test_the_switch_note_is_absent_when_enabled(client, trading):
    assert "data-news-off" not in client.get("/news").text


# ------------------------------------------------------------------ scores and the gate


def test_scores_show_hit_rate_result_twin_edge_and_gate(client, conn, trading):
    add_article(conn)
    aid = add_analysis(conn)
    scored(conn, aid, True, 2.0, 0.5)
    scored(conn, aid, True, 1.0, -0.5)
    scored(conn, aid, True, 3.0, 0.0)
    scored(conn, aid, False, -1.0, 1.0, direction="down")
    add_prediction(conn, aid, direction="none")
    add_prediction(conn, aid)  # waiting to be scored
    html = client.get("/news").text
    block = re.search(r'data-score="stocks".*?</article>\s*<article class="card nblock" data-score="crypto"', html, re.S).group(0)
    values = {m: value(block, f'data-metric="{m}"') for m in ("hit-rate", "result", "twin", "edge", "scored")}
    body = text(block)
    assert values["hit-rate"] == "75%" and values["scored"] == "4"
    assert values["result"] == "+1.25%"  # (2 + 1 + 3 - 1) / 4
    assert values["twin"] == "+0.25%"  # (0.5 - 0.5 + 0 + 1.0) / 4
    # Against the market, twins all "down": (direction - twin) x move = 4, 2, 6 and 0 for the down call.
    assert re.fullmatch(r"\+3\.00% ± \d\.\d\d%", values["edge"])
    assert "1 waiting for their time to pass" in body and "1 where Claude saw nothing to trade" in body
    assert "coin-flip twin" in body.lower()
    assert "Needs 100 independent scored calls it would trade; has 4." in element(block, "data-gate")
    basis = element(block, "data-gate-basis")
    for words in ("Counted on the calls it would trade", "confidence 0.6 or more", "not already priced in",
                  "at least 2 times the 0.10% round-trip cost", "never SPY or QQQ", "one per symbol and day",
                  "against the stock market (SPY)", "different model than the one asked never count",
                  "The cards below count every call on its own."):
        assert words in basis
    assert 'data-ok="false"' in tag_of(block, "data-gate")
    assert "4 live" in element(block, "data-source-counts") and "0 replay" in element(block, "data-source-counts")


def test_live_and_replay_are_counted_apart(client, conn, trading):
    add_article(conn)
    live, replay = add_analysis(conn), add_analysis(conn, source="replay")
    scored(conn, live, True, 1.0, 0.0)
    scored(conn, replay, True, 1.0, 0.0, source="replay")
    scored(conn, replay, False, -1.0, 0.0, source="replay")
    block = re.search(r'data-score="stocks".*?</article>', client.get("/news").text, re.S).group(0)
    counts = element(block, "data-source-counts")
    assert "1 live" in counts and "2 replay" in counts


def test_crypto_scores_stay_apart_from_stocks(client, conn, trading):
    add_article(conn, 1, ours=("BTC/USD",))
    aid = add_analysis(conn)
    scored(conn, aid, True, 4.0, 0.0, symbol="BTC/USD", market="crypto")
    html = client.get("/news").text
    crypto = html.split('data-score="crypto"', 1)[1]
    stocks = re.search(r'data-score="stocks".*?data-score="crypto"', html, re.S).group(0)
    assert value(crypto, 'data-metric="result"') == "+4.00%"
    assert value(stocks, 'data-metric="result"') == "-"
    basis = element(crypto, "data-gate-basis")
    assert "0.70% round-trip cost" in basis and "rises only" in basis and "against Bitcoin" in basis


def test_replays_answered_by_a_fallback_model_are_left_out(client, conn, trading):
    add_article(conn)
    aid = add_analysis(conn, source="replay")
    conn.execute("UPDATE news_analyses SET served_model = 'another-model' WHERE id = %s", (aid,))
    scored(conn, aid, True, 2.0, 0.0, source="replay")
    block = re.search(r'data-score="stocks".*?data-score="crypto"', client.get("/news").text, re.S).group(0)
    assert value(block, 'data-metric="scored"') == "0"
    assert "1 replay left out (answered by a fallback model)" in card(block, 'data-metric="scored"')


def test_the_up_down_and_confidence_tables_are_there(client, conn, trading):
    add_article(conn)
    aid = add_analysis(conn)
    scored(conn, aid, True, 2.0, 0.0)
    html = client.get("/news").text
    stocks = re.search(r'data-score="stocks".*?data-score="crypto"', html, re.S).group(0)
    assert "Up calls" in stocks and "Down calls" in stocks and "How sure Claude was" in stocks


# ------------------------------------------------------------------ paper trading


def test_start_is_disabled_with_the_gates_reason_when_the_gate_fails(client, trading):
    html = client.get("/news").text
    book = re.search(r'data-book="stocks".*?</article>\s*(?=<article class="card nblock" data-book="crypto")', html, re.S).group(0)
    start = tag_of(book, 'data-action="news-start"')
    assert " disabled" in start and 'data-market="stocks"' in start
    assert "Cannot start yet: Needs 100 independent scored calls it would trade; has 0." in element(book, "data-start-reason")
    assert " disabled" in tag_of(book, 'data-action="news-stop"')  # nothing to stop
    assert element(book, "data-book-status") == "Not started"


def test_start_is_enabled_when_the_gate_passes(client, trading, monkeypatch):
    monkeypatch.setattr("coordinator.news_scoring.gate", lambda conn, market, limits: (True, f"Ready: {market}"))
    html = client.get("/news").text
    assert all(" disabled" not in t for t in attr_tags(html, 'data-action="news-start"'))
    assert "data-start-reason" not in html
    assert "Ready: stocks" in element(html, "data-gate")


def test_a_running_book_shows_its_numbers_and_can_be_stopped(client, trading, monkeypatch):
    monkeypatch.setattr("coordinator.news_scoring.gate", lambda conn, market, limits: (True, "Ready"))
    trading.state["summary"] = summary(stocks={"status": "active", "book_id": 3, "cash": 7000.0, "equity": 10250.5,
                                               "gross_exposure": 3250.5, "short_exposure": 500.0, "open_positions": 2,
                                               "trades_today": 4, "waiting": "Waiting for the market to open"})
    html = client.get("/news").text
    book = re.search(r'data-book="stocks".*?data-book="crypto"', html, re.S).group(0)
    assert element(book, "data-book-status") == "Paper trading"
    assert value(book, 'data-book-metric="equity"') == "$10,250.50"
    assert value(book, 'data-book-metric="cash"') == "$7,000.00"
    assert value(book, 'data-book-metric="short"') == "$500.00"
    assert "Waiting for the market to open" in element(book, "data-waiting")
    assert " disabled" in tag_of(book, 'data-action="news-start"')  # already running
    assert " disabled" not in tag_of(book, 'data-action="news-stop"')


def test_the_market_notes(client, trading):
    html = client.get("/news").text
    notes = [re.sub(r"\s+", " ", unescape(m)) for m in re.findall(r"data-market-note>(.*?)</p>", html, re.S)]
    assert notes == ["Shorts allowed (whole shares, easy-to-borrow only); never SPY or QQQ",
                     "Long only: Alpaca cannot short crypto"]


def test_open_positions_are_listed_with_what_they_are_worth(client, trading):
    later = datetime.now(UTC) + timedelta(days=2)
    trading.state["positions"] = [
        {"id": 1, "symbol": "AAPL", "side": "long", "qty": 10.0, "entry_price": 100.0, "stop_price": 95.0, "price": 110.0,
         "value": 1100.0, "open_pnl": 100.0, "exit_due_at": later, "status": "open", "opened_at": datetime.now(UTC),
         "market": "stocks", "headline": "Apple " + HOSTILE},
        {"id": 2, "symbol": "TSLA", "side": "short", "qty": 5.0, "entry_price": 100.0, "stop_price": 105.0, "price": 110.0,
         "value": -550.0, "open_pnl": -50.0, "exit_due_at": later, "status": "open", "opened_at": datetime.now(UTC),
         "market": "stocks", "headline": "Tesla news"},
        {"id": 3, "symbol": "ETH/USD", "side": "long", "qty": 0.5, "entry_price": 2000.0, "stop_price": 1900.0, "price": None,
         "value": None, "open_pnl": None, "exit_due_at": later, "status": "opening", "opened_at": datetime.now(UTC),
         "market": "crypto", "headline": "Ether news"},
    ]
    html = client.get("/news").text
    assert len(attr_tags(html, "data-position")) == 3
    rows = {m: text(re.search(rf'data-position="{m}".*?</tr>', html, re.S).group(0)) for m in ("AAPL", "TSLA", "ETH/USD")}
    assert "Long" in rows["AAPL"] and "$110.00" in rows["AAPL"] and "$1,100.00" in rows["AAPL"] and "+$100.00" in rows["AAPL"]
    assert "Short (profits if it falls)" in rows["TSLA"] and "$550.00" in rows["TSLA"] and "−$50.00" in rows["TSLA"]
    assert 'data-tone="loss"' in re.search(r'data-position="TSLA".*?</tr>', html, re.S).group(0)
    assert "Crypto" in rows["ETH/USD"] and "$2,000.00" in rows["ETH/USD"]
    assert "<script>alert('headline')" not in html and "&lt;script&gt;" in html


def test_without_the_trading_module_the_screen_still_works(client, monkeypatch):
    monkeypatch.setitem(sys.modules, "coordinator.news_trading", None)
    monkeypatch.delattr(coordinator, "news_trading", raising=False)
    r = client.get("/news")
    assert r.status_code == 200 and "not built into this version yet" in element(r.text, "data-paper-missing")
    assert "news-start" not in r.text


# ------------------------------------------------------------------ the routes


def test_start_and_stop_call_news_trading(client, trading):
    r = client.post("/api/news/crypto/start")
    assert r.status_code == 201 and "crypto" in r.json()["message"]
    r = client.post("/api/news/stocks/stop")
    assert r.status_code == 200 and "stocks" in r.json()["message"]
    assert trading.calls == [("start", "crypto", "paper"), ("stop", "stocks")]


def test_an_unknown_market_is_a_404(client, trading):
    r = client.post("/api/news/bonds/start")
    assert r.status_code == 404 and "stocks or crypto" in r.json()["detail"]
    assert trading.calls == []


def test_a_refusal_from_news_trading_comes_back_as_its_message(client, trading, monkeypatch):
    from coordinator.errors import Conflict

    def refuse(conn, market, mode, limits):
        raise Conflict("Does not beat the coin-flip twin yet")
    monkeypatch.setattr("coordinator.news_trading.start", refuse)
    r = client.post("/api/news/stocks/start")
    assert r.status_code == 409 and r.json()["detail"] == "Does not beat the coin-flip twin yet"


def test_a_wrong_origin_is_refused(client, trading):
    r = client.post("/api/news/stocks/start", headers={"Origin": "https://box1.example.ts.net"})
    assert r.status_code == 403 and "FLEET_PUBLIC_URL" in r.text
    assert trading.calls == []
    ok = client.post("/api/news/stocks/start", headers={"Origin": "http://127.0.0.1:8090/"})
    assert ok.status_code == 201


def test_live_mode_needs_the_typed_phrase(client, trading):
    client.app.state.broker_status.broker.mode = "live"
    r = client.post("/api/news/stocks/start")
    assert r.status_code == 400 and "TRADE REAL MONEY" in r.json()["detail"]
    assert trading.calls == []
    assert client.post("/api/news/stocks/start", json={"confirm": "TRADE REAL MONEY"}).status_code == 201


# ------------------------------------------------------------------ the feed


def test_the_feed_shows_articles_newest_first_with_claudes_calls(client, conn, trading):
    add_article(conn, 1, "Older headline", created=datetime.now(UTC) - timedelta(hours=3), status="analysed")
    add_article(conn, 2, "Newer headline", ours=("AAPL", "BTC/USD"), created=datetime.now(UTC) - timedelta(minutes=10), status="analysed")
    aid = add_analysis(conn, 2)
    add_prediction(conn, aid, 2, "AAPL", "up", "Strong demand for the chip", priced_in=True)
    add_prediction(conn, aid, 2, "BTC/USD", "down", "Regulators are circling", market="crypto", hit=True, signed_pct=-1.5,
                   twin_signed_pct=0.25, scored_at=datetime.now(UTC))
    html = client.get("/news").text
    assert re.findall(r'data-article="(\d+)"', html) == ["2", "1"]
    article = re.search(r'data-article="2".*?</li>', html, re.S).group(0)
    assert re.findall(r'data-symbol>([^<]+)<', article) == ["AAPL", "BTC/USD"]
    assert element(article, "data-analysis-status") == "Read by Claude"
    aapl, btc = (re.search(rf'data-call="{s}".*?</div>', article, re.S).group(0) for s in ("AAPL", "BTC/USD"))
    assert element(aapl, "data-dir-word") == "Up" and "about 2.5% in 3 days" in element(aapl, "data-move")
    assert element(aapl, "data-confidence") == "70% sure" and element(aapl, "data-priced-in") == "Already priced in"
    assert element(aapl, "data-score-line") == "Not scored yet."
    assert "Strong demand for the chip" in element(aapl, "data-rationale") and "<details" in aapl
    assert element(btc, "data-dir-word") == "Down" and "data-priced-in" not in btc
    score = element(btc, "data-score-line")
    assert "−1.50%" in score and "Right" in score and "+0.25%" in score
    link = tag_of(article, "rel")
    assert 'href="https://example.com/a"' in link and 'rel="noopener noreferrer"' in link and 'target="_blank"' in link


def test_a_call_of_none_is_shown_as_no_call(client, conn, trading):
    add_article(conn, 1, status="analysed")
    add_prediction(conn, add_analysis(conn), 1, direction="none", rationale="Nothing new in this")
    article = re.search(r'data-article="1".*?</li>', client.get("/news").text, re.S).group(0)
    assert element(article, "data-dir-word") == "No call" and "data-move" not in article and "data-score-line" not in article


def test_unreadable_articles_say_why(client, conn, trading):
    add_article(conn, 1, status="over_budget", note="today's news cap of $5.00 is reached")
    article = re.search(r'data-article="1".*?</li>', client.get("/news").text, re.S).group(0)
    assert element(article, "data-analysis-status") == "Not read: over the spending cap"
    assert "cap of $5.00" in element(article, "data-analysis-note")


def test_the_feed_is_capped_at_fifty(client, conn, trading):
    for i in range(1, 56):
        add_article(conn, i, f"Headline {i}", created=datetime.now(UTC) - timedelta(minutes=i))
    ids = re.findall(r'data-article="(\d+)"', client.get("/news").text)
    assert len(ids) == 50 and ids[0] == "1" and "55" not in ids


def test_hostile_article_text_is_escaped(client, conn, trading):
    add_article(conn, 1, HOSTILE, url="javascript:alert(1)", status="analysed", note="<b>note</b>")
    add_article(conn, 2, "Second", url='https://example.com/"><script>alert(2)</script>', status="analysed")
    aid = add_analysis(conn, 1)
    add_prediction(conn, aid, 1, rationale=HOSTILE_WHY)
    for path in ("/news", "/fragments/news"):
        html = client.get(path).text
        assert "<script>alert" not in html and "<img src=x" not in html and "<b>note</b>" not in html
        assert "&lt;script&gt;alert(&#39;headline&#39;)&lt;/script&gt;" in html
        assert "&lt;img src=x onerror=alert(&#39;why&#39;)&gt;" in html
        assert 'href="javascript:' not in html  # a non-web address never becomes a link
        assert 'href="https://example.com/&#34;&gt;&lt;script&gt;' in html  # a web address is escaped inside the attribute
