"""Claude Haiku's daily market note (coordinator/market_note.py, docs/AI_PLAN.md stage D),
with stand-ins for Anthropic and Alpaca's news: nothing here calls either.

What these prove: the note is written once per trading day, from note_time Chicago time,
from the headlines since the last close; a note written after the open is kept but does
not count in the forward test; paper results on flagged days are compared with the other
days only over notes written before the open; and a news problem is retried, not paid
for."""
from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from coordinator import market_note
from coordinator.haiku import AiSettings
from tests.test_ai_ideas import FakeClient

CT = ZoneInfo("America/Chicago")
NOTE = {"summary": "Inflation report before the open; otherwise a quiet night.",
        "events": [{"time": "07:30", "what": "Consumer prices report"}], "news_level": "normal", "flag": True,
        "reason": "A big report lands an hour before the open."}


def at(day: str, hhmm: str) -> datetime:
    h, m = (int(x) for x in hhmm.split(":"))
    return datetime.combine(date.fromisoformat(day), datetime.min.time(), CT).replace(hour=h, minute=m).astimezone(timezone.utc)


class News:
    name = "test headlines"

    def __init__(self, items=None, error=None):
        self.items = items if items is not None else [
            {"time": "2026-10-08T23:05:00Z", "headline": "Futures steady ahead of inflation data",
             "summary": "Investors wait for the consumer prices report.", "source": "Benzinga", "symbols": []},
        ]
        self.error = error
        self.asked = []

    def headlines(self, start, end, limit):
        self.asked.append((start, end, limit))
        if self.error:
            raise self.error
        return self.items


def write(conn, now, news=None, client=None, settings=AiSettings(), key="sk-test"):
    return market_note.write_note(conn, settings, key, news or News(), client or FakeClient(answer=NOTE), now)


def test_the_note_is_due_from_its_time_on_trading_days_only(conn):
    s = AiSettings()
    assert market_note.due(conn, s, at("2026-10-09", "07:44")) is None      # a Friday, before 07:45
    assert market_note.due(conn, s, at("2026-10-09", "07:45")) == date(2026, 10, 9)
    assert market_note.due(conn, s, at("2026-10-10", "09:00")) is None      # Saturday
    assert market_note.due(conn, s, at("2026-11-26", "09:00")) is None      # Thanksgiving
    assert market_note.due(conn, s, at("2026-10-09", "15:01")) is None      # the session is over


def test_a_note_is_written_from_the_headlines_since_the_last_close(conn):
    news, client = News(), FakeClient(answer=NOTE)
    reply = write(conn, at("2026-10-12", "07:50"), news, client)            # a Monday
    assert reply["written"] and reply["flagged"]
    start, end, limit = news.asked[0]
    assert start == at("2026-10-09", "15:00") and end == at("2026-10-12", "07:50") and limit == 60  # since Friday's close
    req = client.requests[0]
    text = req["messages"][0]["content"]
    assert "Monday, October 12, 2026" in text and "Futures steady ahead of inflation data" in text
    assert "Thu 18:05 [Benzinga]" in text and "opens at 08:30" in text
    assert req["output_config"]["effort"] == "low" and req["max_tokens"] == 4000
    row = conn.execute("SELECT * FROM market_notes").fetchone()
    assert row["day"] == date(2026, 10, 12) and row["before_open"] and row["flagged"] and row["headlines"] == 1
    assert row["note"]["events"][0]["what"] == "Consumer prices report"
    assert conn.execute("SELECT feature FROM ai_calls").fetchone()["feature"] == "market_note"
    assert write(conn, at("2026-10-12", "08:10"))["skipped"] == "not due"  # once a day


def test_a_note_written_after_the_open_does_not_count(conn):
    write(conn, at("2026-10-12", "09:15"))
    row = conn.execute("SELECT * FROM market_notes").fetchone()
    assert not row["before_open"]
    assert market_note.comparison(conn)["days"] == 0


def test_a_news_problem_is_tried_again_and_costs_nothing(conn):
    client = FakeClient(answer=NOTE)
    reply = write(conn, at("2026-10-12", "07:50"), News(error=ConnectionError("down")), client)
    assert reply["error"].startswith("reading the news failed") and client.requests == []
    assert conn.execute("SELECT count(*) AS n FROM market_notes").fetchone()["n"] == 0
    assert write(conn, at("2026-10-12", "08:00"))["written"]


def test_no_key_no_news_or_turned_off_means_no_note(conn):
    now = at("2026-10-12", "07:50")
    assert write(conn, now, key="")["skipped"] == "off"
    assert write(conn, now, settings=AiSettings(note_enabled=False))["skipped"] == "off"
    assert market_note.write_note(conn, AiSettings(), "sk-test", None, FakeClient(), now)["skipped"].startswith("no news")


def test_a_problem_is_shown_until_the_note_is_written(conn):
    now = at("2026-10-12", "07:50")
    write(conn, now, client=FakeClient(error=ConnectionError("down")))
    view = market_note.view(conn, AiSettings(), "sk-test", News(), now)
    assert view["problem"] == "Last note call: error: ConnectionError: down"
    conn.execute("UPDATE ai_calls SET at = at - interval '1 hour'")  # the next try is in a new hour
    write(conn, at("2026-10-12", "08:00"))
    write(conn, at("2026-10-12", "08:05"), client=FakeClient(error=ConnectionError("down")))  # not due: no call
    assert "problem" not in market_note.view(conn, AiSettings(), "sk-test", News(), at("2026-10-12", "08:10"))


def test_an_answer_without_a_news_level_is_not_kept(conn):
    reply = write(conn, at("2026-10-12", "07:50"), client=FakeClient(answer={**NOTE, "news_level": "wild"}))
    assert not reply["written"] and conn.execute("SELECT count(*) AS n FROM market_notes").fetchone()["n"] == 0


def test_paper_results_on_flagged_days_are_compared_with_the_other_days(conn):
    from coordinator import models

    models.sync_starters(conn)
    conn.execute("INSERT INTO futures_books (model_id, venue, contracts) VALUES ('gap_fade', 'alpaca_paper', 1)")
    book = conn.execute("SELECT id FROM futures_books").fetchone()["id"]
    days = {"2026-10-12": (True, -50.0), "2026-10-13": (False, 30.0), "2026-10-14": (False, 10.0)}
    for day, (flag, pnl) in days.items():
        write(conn, at(day, "07:50"), client=FakeClient(answer={**NOTE, "flag": flag}))
        conn.execute("INSERT INTO futures_days (book_id, day, pnl) VALUES (%s, %s, %s)", (book, day, pnl))
    write(conn, at("2026-10-15", "09:00"), client=FakeClient(answer={**NOTE, "flag": True}))  # after the open
    conn.execute("INSERT INTO futures_days (book_id, day, pnl) VALUES (%s, '2026-10-15', -999)", (book,))
    c = market_note.comparison(conn)
    assert c["days"] == 3 and c["flagged_days"] == 1 and not c["enough"]
    assert c["flagged"] == {"n": 1, "avg": -50.0} and c["other"] == {"n": 2, "avg": 20.0}
    view = market_note.view(conn, AiSettings(), "sk-test", News(), at("2026-10-15", "10:00"))
    assert "Futures paper results on flagged days: 1 model-day, average $-50.00" in view["comparison"]
    assert "Too few days to mean anything yet (40 needed)" in view["comparison"]
    assert view["title"] == "Today's market note" and "after the open" in view["written"]


def test_alpaca_news_is_read_newest_first_with_a_limit():
    class Client:
        def __init__(self):
            self.requests = []

        def get_news(self, req):
            self.requests.append(req)
            return {"news": [{"headline": "Stocks rise", "summary": "A summary", "created_at": "2026-10-12T11:00:00Z",
                              "source": "benzinga", "symbols": ["SPY"]}, {"headline": "", "summary": "no headline"}]}

    client = Client()
    got = market_note.AlpacaNews("k", "s", client=client).headlines(at("2026-10-09", "15:00"), at("2026-10-12", "07:45"), 60)
    assert got == [{"time": "2026-10-12T11:00:00Z", "headline": "Stocks rise", "summary": "A summary",
                    "source": "benzinga", "symbols": ["SPY"]}]
    req = client.requests[0]
    assert req.limit == 60 and req.sort == "desc" and req.start == at("2026-10-09", "15:00")


def test_the_futures_view_shows_the_note(client, conn):
    client.app.state.ai_key = ""
    html = client.get("/models?market=futures").text
    assert "Claude Haiku can write a short note before each session" in html
    client.app.state.ai_key = "sk-test"
    client.app.state.news = News()
    write(conn, at("2026-10-12", "07:50"))
    html = client.get("/models?market=futures").text
    assert "Inflation report before the open" in html and "Flagged day" in html and "Consumer prices report" in html
    assert "Advice only: nothing trades on it." in html and "Forward test: 1 notes written before the open" in html


@pytest.mark.parametrize("config, kind", [
    ({"fake_broker": True, "alpaca_paper_key_id": "", "alpaca_paper_secret": ""}, market_note.FakeNews),
    ({"fake_broker": False, "alpaca_paper_key_id": "", "alpaca_paper_secret": ""}, type(None)),
])
def test_the_news_source_follows_the_config(config, kind):
    from types import SimpleNamespace

    assert isinstance(market_note.make_news(SimpleNamespace(**config)), kind)
