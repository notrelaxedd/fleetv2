"""News ingest: symbol mapping, HTML to text, storing, polling, backfill, the fake source."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from coordinator import data, news_feed
from coordinator.news_feed import (
    Article, FakeNewsSource, NoNewsSource, backfill, from_universe, html_to_text, make_news_source,
    parse_news_response, poll_step, recent, store_articles, to_universe,
)
from fleet2 import universe

NOW = datetime(2026, 10, 12, 15, 0, tzinfo=timezone.utc)


def article(i: int, symbols=("AAPL",), created: datetime | None = None, **kw) -> Article:
    base = dict(id=i, headline=f"Headline {i}", summary="s", content="c", author="a", source="src",
                url=f"https://example.invalid/{i}", symbols=tuple(symbols),
                created_at=created or NOW - timedelta(minutes=30), updated_at=None)
    base.update(kw)
    return Article(**base)


class ScriptedSource:
    """Serves a fixed list of pages and records every request."""

    connected = True
    problem = None

    def __init__(self, pages: list[list[Article]], fail_on: int | None = None) -> None:
        self.pages = pages
        self.fail_on = fail_on
        self.calls: list[tuple] = []

    def fetch(self, start, end, symbols, page_token):
        self.calls.append((start, end, list(symbols), page_token))
        n = int(page_token or 0)
        if self.fail_on is not None and n == self.fail_on:
            raise RuntimeError("Alpaca is down")
        more = str(n + 1) if n + 1 < len(self.pages) else None
        return self.pages[n], more


def cursor_of(conn):
    row = conn.execute("SELECT value FROM settings WHERE key = 'news_cursor'").fetchone()
    return None if row is None else datetime.fromisoformat(row["value"].replace("Z", "+00:00"))


def status_of(conn, i: int) -> str:
    return conn.execute("SELECT analysis_status FROM news WHERE id = %s", (i,)).fetchone()["analysis_status"]


# ------------------------------------------------------------------ symbols


def test_symbol_mapping():
    assert to_universe("BTCUSD") == "BTC/USD"
    assert to_universe("btcusd") == "BTC/USD"
    assert to_universe("BTC/USD") == "BTC/USD"
    assert to_universe("AAPL") == "AAPL"
    assert to_universe("V") == "V"
    assert to_universe("TSLA") is None
    assert to_universe("") is None
    assert from_universe("BTC/USD") == "BTCUSD"
    assert from_universe("AAPL") == "AAPL"
    for s in (*universe.STOCKS, *universe.CRYPTO):
        assert to_universe(from_universe(s)) == s


# ------------------------------------------------------------------ text


def test_html_to_text_keeps_paragraphs_and_unescapes():
    html = ("<div><p>Apple &amp; friends   beat <b>estimates</b>.</p><p>Second&nbsp;paragraph<br/>new line</p>"
            "<ul><li>one</li><li>two</li></ul></div>")
    assert html_to_text(html) == "Apple & friends beat estimates.\n\nSecond paragraph\nnew line\n\none\ntwo"


def test_html_to_text_drops_scripts_and_styles():
    html = "<style>p {color:red}</style><p>Hello</p><script>alert('x')</script><p>World &lt;3</p>"
    assert html_to_text(html) == "Hello\n\nWorld <3"


def test_html_to_text_handles_plain_and_empty():
    assert html_to_text(None) == ""
    assert html_to_text("") == ""
    assert html_to_text("just text") == "just text"


def test_store_caps_content_and_strips_nul(conn):
    store_articles(conn, [article(1, content="x" * 25_000, headline="bad\x00head")])
    row = conn.execute("SELECT headline, length(content) AS n FROM news WHERE id = 1").fetchone()
    assert row["n"] == 20_000
    assert row["headline"] == "badhead"


def test_parse_caps_content():
    raw = {"news": [{"id": 1, "headline": "h", "created_at": "2026-10-12T14:00:00Z",
                     "symbols": ["AAPL"], "content": "<p>" + "y" * 30_000 + "</p>"}]}
    (a,), _ = parse_news_response(raw)
    assert len(a.content) == 20_000


# ------------------------------------------------------------------ parsing


def test_parse_news_response():
    raw = {
        "news": [
            {"id": 24843171, "headline": "Apple &amp; Co. Beat   Estimates", "summary": "<p>Short.</p>",
             "content": "<p>Long &amp; detailed.</p><p>Second.</p>", "author": "Jane Doe", "source": "benzinga",
             "url": "https://example.invalid/a", "symbols": ["AAPL", "BTCUSD", "TSLA"],
             "created_at": "2026-10-12T14:01:02.123456789Z", "updated_at": "2026-10-12T14:05:00Z",
             "images": []},
            {"id": "77", "headline": "No symbols", "created_at": "2026-10-12T14:02:00+00:00", "symbols": None},
            {"headline": "no id", "created_at": "2026-10-12T14:02:00Z"},
            {"id": 5, "headline": "", "created_at": "2026-10-12T14:02:00Z"},
            {"id": 6, "headline": "no time"},
            "garbage",
        ],
        "next_page_token": "abc123",
    }
    articles, token = parse_news_response(raw)
    assert token == "abc123"
    assert [a.id for a in articles] == [24843171, 77]
    a = articles[0]
    assert a.headline == "Apple & Co. Beat Estimates"
    assert a.summary == "Short."
    assert a.content == "Long & detailed.\n\nSecond."
    assert a.author == "Jane Doe" and a.source == "benzinga"
    assert a.symbols == ("AAPL", "BTCUSD", "TSLA")
    assert a.created_at == datetime(2026, 10, 12, 14, 1, 2, 123456, tzinfo=timezone.utc)
    assert a.updated_at == datetime(2026, 10, 12, 14, 5, tzinfo=timezone.utc)
    assert articles[1].symbols == () and articles[1].updated_at is None


def test_parse_empty_and_last_page():
    assert parse_news_response({"news": [], "next_page_token": None}) == ([], None)
    assert parse_news_response({"news": None}) == ([], None)
    assert parse_news_response(None) == ([], None)


class RecordingClient:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.requests = response, error, []

    def get(self, path, data=None, **kw):
        self.requests.append((path, data))
        if self.error:
            raise self.error
        return self.response


def test_alpaca_source_builds_request_and_parses(monkeypatch):
    waits = []
    monkeypatch.setattr(data.ALPACA_LIMITER, "acquire", lambda: waits.append(1) or 0.0)
    client = RecordingClient({"news": [{"id": 1, "headline": "h", "created_at": "2026-10-12T14:00:00Z",
                                        "symbols": ["AAPL"]}], "next_page_token": "tok2"})
    src = news_feed.AlpacaNewsSource("k", "s", client=client)
    assert src.connected and src.problem is None
    start, end = NOW - timedelta(hours=1), NOW
    articles, token = src.fetch(start, end, ["AAPL", "BTCUSD"], "tok1")
    assert [a.id for a in articles] == [1] and token == "tok2"
    assert waits == [1]
    path, params = client.requests[0]
    assert path == "/news"
    assert params["symbols"] == "AAPL,BTCUSD"
    assert params["sort"] == "asc" and params["limit"] == 50
    assert params["include_content"] is True and params["exclude_contentless"] is False
    assert params["page_token"] == "tok1"
    assert params["start"] == start.isoformat() and params["end"] == end.isoformat()


def test_alpaca_source_first_page_sends_no_token(monkeypatch):
    monkeypatch.setattr(data.ALPACA_LIMITER, "acquire", lambda: 0.0)
    client = RecordingClient({"news": []})
    news_feed.AlpacaNewsSource("k", "s", client=client).fetch(NOW - timedelta(hours=1), NOW, ["AAPL"], None)
    assert "page_token" not in client.requests[0][1]


def test_alpaca_source_wraps_errors(monkeypatch):
    from alpaca.common.exceptions import APIError

    monkeypatch.setattr(data.ALPACA_LIMITER, "acquire", lambda: 0.0)

    class Denied(APIError):
        status_code = 403

        def __init__(self):
            Exception.__init__(self, "forbidden")

    with pytest.raises(RuntimeError, match="refused the keys"):
        news_feed.AlpacaNewsSource("k", "s", client=RecordingClient(error=Denied())).fetch(
            NOW - timedelta(hours=1), NOW, ["AAPL"], None)
    with pytest.raises(RuntimeError, match="Could not read Alpaca's news feed: boom"):
        news_feed.AlpacaNewsSource("k", "s", client=RecordingClient(error=OSError("boom"))).fetch(
            NOW - timedelta(hours=1), NOW, ["AAPL"], None)


# ------------------------------------------------------------------ sources


def test_no_news_source():
    src = NoNewsSource("no keys")
    assert not src.connected and src.problem == "no keys"
    with pytest.raises(RuntimeError, match="no keys"):
        src.fetch(NOW - timedelta(hours=1), NOW, [], None)


def test_make_news_source():
    fake = make_news_source(SimpleNamespace(fake_broker=True, alpaca_paper_key_id="", alpaca_paper_secret=""))
    assert isinstance(fake, FakeNewsSource)
    none = make_news_source(SimpleNamespace(fake_broker=False, alpaca_paper_key_id="", alpaca_paper_secret=""))
    assert isinstance(none, NoNewsSource) and not none.connected
    assert none.problem == "Add your Alpaca paper keys to .env on box1 (the news feed needs them)"
    half = make_news_source(SimpleNamespace(fake_broker=False, alpaca_paper_key_id="k", alpaca_paper_secret=""))
    assert isinstance(half, NoNewsSource)
    real = make_news_source(SimpleNamespace(fake_broker=False, alpaca_paper_key_id="k", alpaca_paper_secret="s"))
    assert isinstance(real, news_feed.AlpacaNewsSource) and real.connected


def fetch_all(src, start, end, symbols):
    out, token = [], None
    while True:
        page, token = src.fetch(start, end, symbols, token)
        out += page
        if token is None:
            return out


def test_fake_source_is_deterministic_and_stable():
    start, end = NOW - timedelta(hours=6), NOW
    tags = news_feed.universe_tags()
    a = fetch_all(FakeNewsSource(seed=3), start, end, tags)
    b = fetch_all(FakeNewsSource(seed=3), start, end, tags)
    assert a == b and a
    assert a != fetch_all(FakeNewsSource(seed=4), start, end, tags)
    # a sub-window is a slice of the same articles, with the same ids
    sub = fetch_all(FakeNewsSource(seed=3), start + timedelta(hours=2), end - timedelta(hours=1), tags)
    by_id = {x.id: x for x in a}
    assert sub and all(by_id[x.id] == x for x in sub)
    assert len({x.id for x in a}) == len(a)
    assert all(start <= x.created_at <= end for x in a)
    assert [x.created_at for x in a] == sorted(x.created_at for x in a)


def test_fake_source_content_mix():
    arts = fetch_all(FakeNewsSource(), NOW - timedelta(days=3), NOW, [])
    assert len(arts) >= 3 * 24 * 2  # a few an hour
    assert all(x.headline.startswith("Demo:") for x in arts)
    ours = [bool(news_feed.ours_of(x)) for x in arts]
    assert any(ours) and not all(ours)  # some name only symbols we do not trade
    assert any("BTCUSD" in x.symbols or "ETHUSD" in x.symbols for x in arts)
    assert any(len(x.symbols) >= 2 for x in arts)
    assert any("<p>" in x.content for x in arts)


def test_fake_source_paging_and_filter():
    src = FakeNewsSource(page_size=5)
    start, end = NOW - timedelta(hours=10), NOW
    page, token = src.fetch(start, end, [], None)
    assert len(page) == 5 and token == "5"
    everything = fetch_all(src, start, end, [])
    assert len({x.id for x in everything}) == len(everything) > 5
    only = fetch_all(src, start, end, ["BTCUSD"])
    assert only and all("BTCUSD" in x.symbols for x in only)
    with pytest.raises(RuntimeError):
        src.fetch(start, end, [], "nope")


# ------------------------------------------------------------------ store_articles


def test_store_skips_unmapped_dedups_and_sorts(conn):
    a = article(1, symbols=("TSLA",))
    b = article(2, symbols=("MSFT", "BTCUSD", "AAPL", "AAPL", "TSLA"))
    assert store_articles(conn, [a, b]) == 1
    assert conn.execute("SELECT count(*) AS n FROM news").fetchone()["n"] == 1
    row = conn.execute("SELECT * FROM news WHERE id = 2").fetchone()
    assert row["ours"] == ["AAPL", "BTC/USD", "MSFT"]
    assert row["symbols"] == ["MSFT", "BTCUSD", "AAPL", "AAPL", "TSLA"]
    assert row["analysis_status"] == "pending"
    assert row["created_at"] == b.created_at


def test_store_upserts_text_but_never_resets_status(conn):
    assert store_articles(conn, [article(1, headline="old", content="full text")]) == 1
    conn.execute("UPDATE news SET analysis_status = 'analysed', analysis_note = 'done' WHERE id = 1")
    again = article(1, headline="new", content="", symbols=("AAPL", "MSFT"))
    assert store_articles(conn, [again], status="pending") == 0
    assert store_articles(conn, [again], status="backfill") == 0
    row = conn.execute("SELECT * FROM news WHERE id = 1").fetchone()
    assert row["headline"] == "new"
    assert row["ours"] == ["AAPL", "MSFT"]
    assert row["content"] == "full text"  # an empty re-fetch does not erase the text
    assert row["analysis_status"] == "analysed" and row["analysis_note"] == "done"


def test_store_counts_only_new_rows_and_same_batch_duplicates(conn):
    batch = [article(1), article(2), article(2)]
    assert store_articles(conn, batch) == 2
    assert store_articles(conn, batch + [article(3)]) == 1


def test_store_status_argument(conn):
    store_articles(conn, [article(1)], status="backfill")
    assert status_of(conn, 1) == "backfill"


# ------------------------------------------------------------------ poll_step


def test_poll_first_run_starts_an_hour_back(conn):
    newest = NOW - timedelta(minutes=5)
    src = ScriptedSource([[article(1, created=NOW - timedelta(minutes=20)), article(2, created=newest)]])
    out = poll_step(conn, src, NOW)
    assert out == {"added": 2, "error": None}
    start, end, symbols, token = src.calls[0]
    assert start == NOW - timedelta(hours=1) and end == NOW and token is None
    assert "BTCUSD" in symbols and "AAPL" in symbols and "BTC/USD" not in symbols
    assert len(symbols) == len(universe.STOCKS) + len(universe.CRYPTO)
    assert cursor_of(conn) == newest
    assert status_of(conn, 1) == "pending"


def test_poll_overlaps_the_cursor_and_does_not_double_count(conn):
    first = article(1, created=NOW - timedelta(minutes=20))
    poll_step(conn, ScriptedSource([[first]]), NOW)
    later = NOW + timedelta(minutes=1)
    src = ScriptedSource([[first, article(2, created=NOW - timedelta(minutes=1))]])
    out = poll_step(conn, src, later)
    assert out["added"] == 1  # the overlap re-served article 1
    assert src.calls[0][0] == first.created_at - timedelta(minutes=10)
    assert src.calls[0][1] == later
    assert cursor_of(conn) == NOW - timedelta(minutes=1)


def test_poll_cursor_never_moves_backwards_or_past_now(conn):
    poll_step(conn, ScriptedSource([[article(1, created=NOW - timedelta(minutes=2))]]), NOW)
    before = cursor_of(conn)
    # only an older (late-published) article comes back: the cursor stays
    poll_step(conn, ScriptedSource([[article(2, created=NOW - timedelta(minutes=8))]]), NOW)
    assert cursor_of(conn) == before
    # nothing comes back: the cursor stays
    poll_step(conn, ScriptedSource([[]]), NOW)
    assert cursor_of(conn) == before
    # an article stamped in the future (clock skew) is capped at now
    poll_step(conn, ScriptedSource([[article(3, created=NOW + timedelta(hours=3))]]), NOW)
    assert cursor_of(conn) == NOW


def test_poll_cursor_counts_skipped_articles_and_writes_no_audit(conn):
    audits = conn.execute("SELECT count(*) AS n FROM audit_log").fetchone()["n"]
    out = poll_step(conn, ScriptedSource([[article(1, symbols=("TSLA",), created=NOW - timedelta(minutes=3))]]), NOW)
    assert out["added"] == 0
    assert cursor_of(conn) == NOW - timedelta(minutes=3)
    assert conn.execute("SELECT count(*) AS n FROM audit_log").fetchone()["n"] == audits


def test_poll_error_leaves_cursor_alone(conn):
    poll_step(conn, ScriptedSource([[article(1, created=NOW - timedelta(minutes=20))]]), NOW)
    before = cursor_of(conn)
    pages = [[article(2, created=NOW - timedelta(minutes=2))], [article(3, created=NOW - timedelta(minutes=1))]]
    out = poll_step(conn, ScriptedSource(pages, fail_on=1), NOW)
    assert out["error"] == "Alpaca is down"
    assert out["added"] == 1  # page one was stored, and a later poll sees it again harmlessly
    assert cursor_of(conn) == before


def test_poll_error_on_first_run_sets_no_cursor(conn):
    out = poll_step(conn, ScriptedSource([[]], fail_on=0), NOW)
    assert out == {"added": 0, "error": "Alpaca is down"}
    assert cursor_of(conn) is None


def test_poll_not_connected_returns_problem(conn):
    out = poll_step(conn, NoNewsSource("no keys"), NOW)
    assert out == {"added": 0, "error": "no keys"}
    assert cursor_of(conn) is None


def test_poll_follows_pages_up_to_the_bound(conn):
    pages = [[article(i + 1, created=NOW - timedelta(minutes=60 - i))] for i in range(15)]
    src = ScriptedSource(pages)
    out = poll_step(conn, src, NOW)
    assert len(src.calls) == news_feed.POLL_MAX_PAGES == 10
    assert [c[3] for c in src.calls] == [None] + [str(i) for i in range(1, 10)]
    assert out == {"added": 10, "error": None}
    assert cursor_of(conn) == NOW - timedelta(minutes=51)  # carries on from here next poll
    # the next poll starts 10 minutes behind that and reaches the rest
    src2 = ScriptedSource(pages[9:])
    assert poll_step(conn, src2, NOW)["added"] == 5


def test_poll_stops_at_last_page(conn):
    src = ScriptedSource([[article(1)], [article(2)], [article(3)]])
    assert poll_step(conn, src, NOW)["added"] == 3
    assert len(src.calls) == 3


def test_poll_with_fake_source_end_to_end(conn):
    src = FakeNewsSource(seed=1)
    out = poll_step(conn, src, NOW)
    assert out["error"] is None and out["added"] > 0
    stored = conn.execute("SELECT count(*) AS n FROM news").fetchone()["n"]
    assert stored == out["added"]
    again = poll_step(conn, src, NOW)
    assert again["added"] == 0
    assert conn.execute("SELECT count(*) AS n FROM news").fetchone()["n"] == stored


# ------------------------------------------------------------------ backfill


def test_backfill_reads_every_page_and_marks_backfill(conn):
    src = ScriptedSource([[article(1)], [article(2), article(3, symbols=("TSLA",))], [article(4)]])
    start, end = NOW - timedelta(days=2), NOW - timedelta(days=1)
    assert backfill(conn, src, start, end) == 3
    assert [c[3] for c in src.calls] == [None, "1", "2"]
    assert all(c[0] == start and c[1] == end for c in src.calls)
    assert {r["analysis_status"] for r in conn.execute("SELECT analysis_status FROM news")} == {"backfill"}
    assert cursor_of(conn) is None  # a backfill never touches the live cursor


def test_backfill_does_not_requeue_or_reset_existing(conn):
    store_articles(conn, [article(1), article(2)])
    conn.execute("UPDATE news SET analysis_status = 'analysed' WHERE id = 1")
    assert backfill(conn, ScriptedSource([[article(1), article(2), article(3)]]), NOW - timedelta(days=1), NOW) == 1
    assert status_of(conn, 1) == "analysed"
    assert status_of(conn, 2) == "pending"
    assert status_of(conn, 3) == "backfill"


def test_backfill_is_bounded(conn):
    pages = [[article(i + 1)] for i in range(20)]
    src = ScriptedSource(pages)
    assert backfill(conn, src, NOW - timedelta(days=1), NOW, max_pages=4) == 4
    assert len(src.calls) == 4


def test_backfill_raises_source_errors(conn):
    with pytest.raises(RuntimeError, match="Alpaca is down"):
        backfill(conn, ScriptedSource([[article(1)], [article(2)]], fail_on=1), NOW - timedelta(days=1), NOW)


# ------------------------------------------------------------------ recent


def test_recent_newest_first_and_limited(conn):
    store_articles(conn, [article(i, created=NOW - timedelta(minutes=i)) for i in range(1, 6)])
    rows = recent(conn, limit=3)
    assert [r["id"] for r in rows] == [1, 2, 3]
    assert rows[0]["created_at"] == (NOW - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
    assert rows[0]["ours"] == ["AAPL"] and rows[0]["analysis_status"] == "pending"
    assert "content" not in rows[0]
    assert len(recent(conn)) == 5
