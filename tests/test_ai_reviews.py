"""Claude Haiku's reviews of futures models (coordinator/ai_reviews.py, docs/AI_PLAN.md
stage C), with a stand-in for the Anthropic client: nothing here calls Anthropic.

What these prove: reviews are automatic for models worth a look whenever their results
change, within a daily limit, and on request from the model's page; Haiku is shown the
page's own numbers; a failed review is shown and not retried on its own; reviews never
reach the recipe writer; and nothing about the model changes."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from coordinator import ai_ideas, ai_reviews, futures_models, models
from coordinator.haiku import AiSettings
from tests.test_ai_ideas import FakeClient, futures_search
from tests.test_futures_screen import FEES, shaped

NOW = datetime(2026, 10, 9, 15, 0, tzinfo=timezone.utc)
REVIEW = {"headline": "Beats its coin flip, but on few attempts.", "verdict": "unclear",
          "luck": ["Only 40 attempts: SECRET-REVIEW-TEXT"], "fragile": ["Half the profit came on one day."],
          "watch": ["Whether paper days stay inside the backtest's range."]}


@pytest.fixture
def tested(conn):
    """gap_fade beats its coin flip; trend_day does not."""
    models.sync_starters(conn)
    models.store_backtest(conn, "gap_fade", shaped(0.5, 0.2, 3000.0), None)
    models.store_backtest(conn, "trend_day", shaped(0.1, 0.3, 100.0), None)
    return conn


def tick(conn, client=None, settings=AiSettings(), key="sk-test", now=NOW):
    return ai_reviews.tick(conn, settings, key, FEES, client or FakeClient(answer=REVIEW), now)


def model(conn, model_id):
    return conn.execute("SELECT * FROM models WHERE id = %s", (model_id,)).fetchone()


def test_without_a_key_nothing_is_reviewed(tested):
    assert tick(tested, key="")["reviewed"] is None
    assert tested.execute("SELECT count(*) AS n FROM model_reviews").fetchone()["n"] == 0
    view = ai_reviews.view(tested, model(tested, "gap_fade"), AiSettings(), "")
    assert view["state"] == "off" and "ANTHROPIC_API_KEY" in view["text"]


def test_models_worth_a_look_are_reviewed_automatically_with_their_own_numbers(tested):
    assert ai_reviews.worth_reviewing(tested, model(tested, "gap_fade"), FEES)
    assert not ai_reviews.worth_reviewing(tested, model(tested, "trend_day"), FEES)
    before = dict(model(tested, "gap_fade"))
    client = FakeClient(answer=REVIEW)
    reply = tick(tested, client)
    assert reply["reviewed"] == "gap_fade" and reply["verdict"] == "unclear"
    req = client.requests[0]
    text = req["messages"][0]["content"]
    assert "Model: Gap fade" in text and "Pass rate: 50%" in text and "Coin-flip twin: 20%" in text
    assert "Checklist for a Combine" in text and "Chance this is luck" in text
    assert req["output_config"]["effort"] == "low" and req["max_tokens"] == 6000
    assert req["output_config"]["format"]["schema"]["properties"]["verdict"]["enum"] == ["promising", "unclear", "weak"]
    row = tested.execute("SELECT * FROM model_reviews").fetchone()
    assert row["status"] == "done" and row["asked_by"] == "auto" and row["review"]["luck"][0].startswith("Only 40")
    assert tested.execute("SELECT feature FROM ai_calls").fetchone()["feature"] == "reviews"
    after = dict(model(tested, "gap_fade"))
    assert after == before  # a review changes nothing about the model
    view = ai_reviews.view(tested, model(tested, "gap_fade"), AiSettings(), "sk-test")
    assert view["state"] == "done" and view["verdict"] == "Unclear" and not view["stale"]
    assert view["asked_by"] == "automatically"


def test_a_model_is_reviewed_again_only_when_its_results_change(tested):
    tick(tested)
    assert tick(tested)["reviewed"] is None  # same results: no second review
    tested.execute("UPDATE models SET backtested_at = backtested_at + interval '1 hour' WHERE id = 'gap_fade'")
    view = ai_reviews.view(tested, model(tested, "gap_fade"), AiSettings(), "sk-test")
    assert view["stale"]  # the page says the results changed since
    assert tick(tested)["reviewed"] == "gap_fade"
    assert tested.execute("SELECT count(*) AS n FROM model_reviews WHERE status = 'done'").fetchone()["n"] == 2


def test_paper_days_bring_a_new_review_every_five_days(tested):
    tick(tested)
    tested.execute("INSERT INTO futures_books (model_id, venue, contracts) VALUES ('gap_fade', 'alpaca_paper', 1)")
    book = tested.execute("SELECT id FROM futures_books").fetchone()["id"]
    for i in range(4):
        tested.execute("INSERT INTO futures_days (book_id, day, pnl) VALUES (%s, %s, 10)", (book, f"2026-10-0{i + 1}"))
    assert tick(tested)["reviewed"] is None
    tested.execute("INSERT INTO futures_days (book_id, day, pnl) VALUES (%s, '2026-10-05', 10)", (book,))
    assert tick(tested)["reviewed"] == "gap_fade"


def test_a_final_check_brings_a_new_review(tested):
    tick(tested)
    tested.execute("INSERT INTO final_checks (model_id, feed, result) VALUES ('gap_fade', 'databento', '{}')")
    assert tick(tested)["reviewed"] == "gap_fade"
    models.store_backtest(tested, "trend_day", shaped(0.1, 0.3, 100.0), None)
    tested.execute("INSERT INTO final_checks (model_id, feed, result) VALUES ('trend_day', 'databento', '{}')")
    assert ai_reviews.worth_reviewing(tested, model(tested, "trend_day"), FEES)  # a Final check makes it worth a look


def test_at_most_reviews_per_day(tested):
    settings = AiSettings(reviews_per_day=1)
    models.store_backtest(tested, "pullback", shaped(0.6, 0.1, 3000.0), None)
    assert tick(tested, settings=settings)["reviewed"] is not None
    assert tick(tested, settings=settings)["reviewed"] is None
    assert tested.execute("SELECT count(*) AS n FROM model_reviews").fetchone()["n"] == 1  # no second one queued
    later = NOW + timedelta(days=1, minutes=1)
    assert tick(tested, settings=settings, now=later)["reviewed"] is not None


def test_turning_automatic_reviews_off_leaves_only_the_button(tested):
    assert tick(tested, settings=AiSettings(review_auto=False))["reviewed"] is None
    ai_reviews.request(tested, "trend_day")  # not worth an automatic look, but the owner asked
    assert tick(tested, settings=AiSettings(review_auto=False))["reviewed"] == "trend_day"


def test_asked_for_reviews_go_first(tested):
    ai_reviews.queue_due(tested, AiSettings(), FEES, NOW)
    ai_reviews.request(tested, "trend_day")
    assert ai_reviews.request(tested, "trend_day")["message"].endswith("already on its way")
    assert tick(tested)["reviewed"] == "trend_day"
    assert tick(tested)["reviewed"] == "gap_fade"
    with pytest.raises(ValueError, match="Run a backtest first"):
        ai_reviews.request(tested, "pullback")


@pytest.mark.parametrize("client, error", [
    (FakeClient(answer={**REVIEW, "verdict": "amazing"}), "bad: no verdict in the answer"),
    (FakeClient(stop_reason="refusal"), "refused"),
    (FakeClient(error=ConnectionError("down")), "error: ConnectionError: down"),
])
def test_a_failed_review_is_shown_and_not_retried_on_its_own(tested, client, error):
    assert tick(tested, client)["error"] == error
    view = ai_reviews.view(tested, model(tested, "gap_fade"), AiSettings(), "sk-test")
    assert view["failed"].endswith(error) and view["state"] == "none"
    assert tick(tested)["reviewed"] is None  # same results: the button is there to try again


def test_reviews_never_reach_the_recipe_writer(tested):
    tick(tested)
    futures_search(tested)
    client = FakeClient()
    ai_ideas.write_recipes(tested, AiSettings(), "sk-test", client=client, now=NOW)
    sent = client.requests[0]["system"] + client.requests[0]["messages"][0]["content"]
    assert "SECRET-REVIEW-TEXT" not in sent and "coin flip" not in sent.lower()


def test_the_cap_holds_for_reviews_too(tested):
    from coordinator import haiku

    haiku.record_call(tested, "recipes", "claude-haiku-5-5", 0, 0, 4.999, "ok", at=NOW)
    client = FakeClient(answer=REVIEW)
    reply = tick(tested, client)
    assert reply["reviewed"] is None and "cap" in reply["skipped"] and client.requests == []
    assert tested.execute("SELECT status FROM model_reviews").fetchone()["status"] == "requested"  # waits


def test_the_model_page_shows_the_review_and_the_button(client, conn):
    client.app.state.topstep = FEES
    client.app.state.ai_key = "sk-test"
    models.store_backtest(conn, "gap_fade", shaped(0.5, 0.2, 3000.0), None)
    html = client.get("/models?market=futures&id=gap_fade").text
    assert "Claude Haiku's review" in html and 'data-action="ask-review"' in html
    r = client.post("/api/models/gap_fade/review")
    assert r.status_code == 201 and "within a minute or two" in r.json()["message"]
    html = client.get("/models?market=futures&id=gap_fade").text
    assert "Claude Haiku is writing a review of this model" in html and 'data-action="ask-review"' not in html
    ai_reviews.run_one(conn, AiSettings(), "sk-test", FEES, FakeClient(answer=REVIEW), datetime.now(timezone.utc))
    html = client.get("/models?market=futures&id=gap_fade").text
    assert "Beats its coin flip, but on few attempts." in html and "What could be luck" in html
    assert "Advice only: it changes nothing." in html and "Ask for a new review" in html
    client.app.state.ai_key = ""
    assert client.post("/api/models/gap_fade/review").status_code == 400
