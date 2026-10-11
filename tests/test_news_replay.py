"""The post-cutoff replay (coordinator/news_replay.py, docs/NEWS_PLAN.md section 1).

What these prove: a window before no_replay_before, in the future or empty is refused, and
so is a replay with no Claude; only 'backfill' articles of the window are replayed, once,
oldest first, deciding five minutes after publication on prices known by then; the
predictions are saved as 'replay' and scored; a second run resumes instead of repeating;
the owner's --max-usd and the monthly cap each stop it and say so; the live daily cap is
neither used nor consumed; and the command line without --yes spends nothing."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from coordinator import cli, haiku, news_feed, news_replay, research
from coordinator.config import Config
from coordinator.haiku import AiSettings
from coordinator.limits import Limits
from coordinator.research import AnalysisResult, Prediction

UTC = timezone.utc
NOW = datetime(2026, 10, 14, 15, 0, tzinfo=UTC)
SINCE = datetime(2026, 8, 3, tzinfo=UTC)
UNTIL = datetime(2026, 8, 10, tzinfo=UTC)
LIMITS = Limits()                      # no_replay_before 2026-08-01
SETTINGS = AiSettings(monthly_cap_usd=50.0, news_daily_cap_usd=5.0)


class EmptySource:
    """A news source with nothing to give: the articles are already in the database."""

    connected = True
    problem = None

    def fetch(self, start, end, symbols, page_token):
        return [], None


class StubAnalyst:
    """Says "up" for every symbol; every call costs `cost`, and could cost `worst`."""

    connected = True
    problem = None
    model = "stub-opus"

    def __init__(self, cost=0.10, worst=0.30, error=None, connected=True):
        self.cost, self.worst, self.error, self.connected = cost, worst, error, connected
        self.calls: list[tuple[int, dict]] = []

    def worst_cost(self, article, context):
        return self.worst

    def analyse(self, article, context):
        self.calls.append((article["id"], context))
        preds = [] if self.error else [
            Prediction(s, "up", 2.0, 1, 0.65, False, "stub") for s in research.ours(article)]
        return AnalysisResult(predictions=preds, requested_model=self.model, served_model=self.model,
                              prompt_version="stub", input_tokens=1000, output_tokens=200, cache_read_tokens=0,
                              cache_write_tokens=0, cost_usd=self.cost, stop_reason="end_turn", raw={"p": 1},
                              error=self.error)


def add(conn, news_id, created_at, status="backfill", ours=("AAPL",)):
    conn.execute(
        "INSERT INTO news (id, headline, summary, content, source, symbols, ours, created_at, analysis_status) "
        "VALUES (%s, 'h', 's', 'c', 'test', %s, %s, %s, %s)",
        (news_id, [s.replace("/", "") for s in ours], list(ours), created_at, status))


def add_window(conn, n, start=SINCE + timedelta(hours=9)):
    """n 'backfill' articles inside the window, an hour apart, ids 1..n, oldest first."""
    for i in range(n):
        add(conn, i + 1, start + timedelta(hours=i))


def go(conn, analyst, source=None, settings=SETTINGS, max_usd=100.0, since=SINCE, until=UNTIL, now=NOW, limits=LIMITS):
    lines: list[str] = []
    out = news_replay.replay(conn, source or EmptySource(), analyst, settings, limits, since, until, max_usd, now,
                             log=lines.append)
    out["lines"] = lines
    return out


def count(conn, sql, *args):
    return conn.execute(sql, args).fetchone()["n"]


# ------------------------------------------------------------------ refusals


def test_a_window_before_the_cutoff_is_refused(conn):
    add_window(conn, 1)
    a = StubAnalyst()
    with pytest.raises(ValueError, match="2026-08-01"):
        go(conn, a, since=datetime(2026, 7, 31, 23, tzinfo=UTC))
    assert a.calls == [] and count(conn, "SELECT count(*) AS n FROM news_analyses") == 0


def test_the_first_allowed_day_is_allowed(conn):
    assert go(conn, StubAnalyst(), since=datetime(2026, 8, 1, tzinfo=UTC))["replayed"] == 0


def test_the_cutoff_comes_from_the_limits(conn):
    late = Limits(no_replay_before=datetime(2026, 9, 1).date())
    with pytest.raises(ValueError, match="2026-09-01"):
        go(conn, StubAnalyst(), limits=late)


def test_a_window_reaching_into_the_future_is_refused(conn):
    with pytest.raises(ValueError, match="future"):
        go(conn, StubAnalyst(), until=NOW + timedelta(minutes=1))


@pytest.mark.parametrize("since,until", [(UNTIL, SINCE), (SINCE, SINCE)])
def test_an_empty_or_backwards_window_is_refused(conn, since, until):
    with pytest.raises(ValueError, match="before its end"):
        go(conn, StubAnalyst(), since=since, until=until)


def test_a_disconnected_analyst_is_refused_before_anything_is_stored(conn):
    a = StubAnalyst(connected=False)
    a.problem = "Add ANTHROPIC_API_KEY to .env on box1"
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        go(conn, a, source=news_feed.FakeNewsSource())
    assert count(conn, "SELECT count(*) AS n FROM news") == 0


def test_no_money_to_spend_is_refused(conn):
    with pytest.raises(ValueError, match="max_usd"):
        go(conn, StubAnalyst(), max_usd=0)


# ------------------------------------------------------------------ what is replayed


def test_only_backfill_articles_of_the_window_are_replayed_oldest_first(conn):
    add(conn, 10, SINCE - timedelta(minutes=1))                       # just before the window
    add(conn, 11, SINCE + timedelta(hours=5))                         # in
    add(conn, 12, SINCE + timedelta(hours=3))                         # in, older: goes first
    add(conn, 13, SINCE + timedelta(hours=4), status="pending")       # the live loop's
    add(conn, 14, SINCE + timedelta(hours=4), status="analysed")      # seen live already
    add(conn, 15, UNTIL)                                              # the end is excluded
    a = StubAnalyst()
    out = go(conn, a)
    assert [i for i, _ in a.calls] == [12, 11]
    assert out["replayed"] == 2 and out["predictions"] == 2 and out["stopped_by"] is None
    status = {r["id"]: r["analysis_status"] for r in conn.execute("SELECT id, analysis_status FROM news")}
    assert status[10] == "backfill" and status[15] == "backfill"
    assert status[13] == "pending" and status[14] == "analysed"       # untouched
    assert status[11] == status[12] == "analysed"


def test_the_decision_is_five_minutes_after_publication_on_prices_known_then(conn):
    t = SINCE + timedelta(hours=14, minutes=20)   # 14:20 UTC, Monday 2026-08-03
    add(conn, 1, t)
    # Friday's daily bar had closed; Monday's (stamped 04:00 UTC, closing 21:00 UTC) had not.
    conn.execute("INSERT INTO bars VALUES ('AAPL', '1Day', %s, 100, 100, 100, 100, 1, 'test')",
                 (datetime(2026, 7, 31, 4, tzinfo=UTC),))
    conn.execute("INSERT INTO bars VALUES ('AAPL', '1Day', %s, 150, 150, 150, 150, 1, 'test')",
                 (datetime(2026, 8, 3, 4, tzinfo=UTC),))
    a = StubAnalyst()
    go(conn, a)
    assert news_replay.REPLAY_DELAY == timedelta(minutes=5)
    context = a.calls[0][1]
    assert datetime.fromisoformat(context["decided_at"]) == t + timedelta(minutes=5)
    assert context["minutes_since_published"] == 5.0
    assert context["symbols"]["AAPL"]["last_close"] == 100.0           # never the later bar
    assert context["symbols"]["AAPL"]["price_now"] is None
    p = conn.execute("SELECT * FROM news_predictions").fetchone()
    assert p["source"] == "replay" and p["decided_at"] == t + timedelta(minutes=5)
    an = conn.execute("SELECT * FROM news_analyses").fetchone()
    assert an["source"] == "replay" and an["created_at"] == NOW        # the money is dated by the real clock


def test_a_second_run_resumes_and_never_repeats(conn):
    add_window(conn, 3)
    a = StubAnalyst()
    first = go(conn, a)
    assert first["replayed"] == 3
    second = go(conn, a)
    assert second["replayed"] == 0 and second["predictions"] == 0 and second["cost"] == 0.0
    assert len(a.calls) == 3
    assert count(conn, "SELECT count(*) AS n FROM news_predictions") == 3


def test_an_article_with_a_replay_analysis_is_not_sent_again_even_if_still_backfill(conn):
    add_window(conn, 2)
    conn.execute("INSERT INTO news_analyses (news_id, source, requested_model, served_model, prompt_version, context)"
                 " VALUES (1, 'replay', 'm', 'm', 'v', '{}')")
    a = StubAnalyst()
    go(conn, a)
    assert [i for i, _ in a.calls] == [2]


def test_the_backfill_fetches_the_window_first(conn):
    a = research.FakeAnalyst()
    out = go(conn, a, source=news_feed.FakeNewsSource(), until=SINCE + timedelta(days=1))
    n = count(conn, "SELECT count(*) AS n FROM news")
    assert out["fetched"] == n > 0 and out["replayed"] == n
    assert count(conn, "SELECT count(*) AS n FROM news WHERE analysis_status = 'backfill'") == 0
    again = go(conn, a, source=news_feed.FakeNewsSource(), until=SINCE + timedelta(days=1))
    assert again["fetched"] == 0 and again["replayed"] == 0


def test_an_article_seen_live_keeps_its_status_through_the_backfill(conn):
    class OneArticle(EmptySource):
        def fetch(self, start, end, symbols, page_token):
            t = SINCE + timedelta(hours=9)
            return [news_feed.Article(7, "h", "s", "c", "a", "s", "u", ("AAPL",), t, t)], None
    add(conn, 7, SINCE + timedelta(hours=9), status="pending")
    a = StubAnalyst()
    out = go(conn, a, source=OneArticle())
    assert out["fetched"] == 0 and a.calls == []
    assert conn.execute("SELECT analysis_status FROM news").fetchone()["analysis_status"] == "pending"


def test_progress_is_logged_every_ten_articles(conn):
    add_window(conn, 25)
    out = go(conn, StubAnalyst(cost=0.01, worst=0.02))
    progress = [x for x in out["lines"] if x.startswith("replayed 10") or x.startswith("replayed 20")]
    assert len(progress) == 2 and not any(x.startswith("replayed 30") for x in out["lines"])


def test_failed_analyses_are_counted_and_three_in_a_row_stop_the_run(conn):
    add_window(conn, 6)
    a = StubAnalyst(error="error: invalid x-api-key")
    out = go(conn, a)
    assert len(a.calls) == 3 and out["replayed"] == 3 and out["failed"] == 3 and out["predictions"] == 0
    assert "3 analyses in a row failed" in out["stopped_by"]


# ------------------------------------------------------------------ money


def test_max_usd_stops_it_before_an_article_that_could_pass_it(conn):
    add_window(conn, 5)
    a = StubAnalyst(cost=0.10, worst=0.30)
    out = go(conn, a, max_usd=0.45)
    assert len(a.calls) == 2                       # 0.20 spent + 0.30 worst > 0.45
    assert out["cost"] == pytest.approx(0.20)
    assert "--max-usd" in out["stopped_by"] and "$0.45" in out["stopped_by"]
    again = go(conn, a, max_usd=100.0)             # a bigger allowance carries on from article 3
    assert [i for i, _ in a.calls] == [1, 2, 3, 4, 5] and again["stopped_by"] is None


def test_the_monthly_cap_stops_it_and_replay_spend_counts_toward_it(conn):
    add_window(conn, 5)
    a = StubAnalyst(cost=0.10, worst=0.30)
    out = go(conn, a, settings=AiSettings(monthly_cap_usd=0.45, news_daily_cap_usd=5.0))
    assert len(a.calls) == 2 and "monthly_cap_usd" in out["stopped_by"]
    assert haiku.month_spend(conn, NOW) == pytest.approx(0.20)         # recorded in October, not August


def test_other_ai_spend_this_month_leaves_less_room(conn):
    add_window(conn, 2)
    haiku.record_call(conn, "recipes", "m", 0, 0, 49.80, "ok", at=NOW - timedelta(days=2))
    a = StubAnalyst()
    out = go(conn, a)
    assert a.calls == [] and "monthly_cap_usd" in out["stopped_by"]


def test_a_replay_neither_uses_nor_consumes_the_live_daily_cap(conn):
    add_window(conn, 3)
    haiku.record_call(conn, "news", "m", 0, 0, 4.99, "ok", at=NOW - timedelta(hours=1))   # live day almost spent
    a = StubAnalyst(cost=0.10, worst=0.30)
    out = go(conn, a)
    assert out["replayed"] == 3 and out["stopped_by"] is None
    assert research.spent_today(conn, NOW) == pytest.approx(4.99)      # still only the live spend
    kinds = {r["feature"]: r["n"] for r in conn.execute(
        "SELECT feature, count(*) AS n FROM ai_calls GROUP BY feature")}
    assert kinds == {"news": 1, "news_replay": 3}
    assert haiku.month_spend(conn, NOW) == pytest.approx(4.99 + 0.30)  # but the month sees both


def test_a_free_analyst_is_never_stopped_by_money(conn):
    add_window(conn, 2)
    out = go(conn, research.FakeAnalyst(), max_usd=0.01)
    assert out["replayed"] == 2 and out["cost"] == 0.0


# ------------------------------------------------------------------ scoring


def test_the_predictions_are_scored_at_the_end(conn):
    add(conn, 1, datetime(2026, 8, 3, 14, 20, tzinfo=UTC))            # Monday
    for sym in ("AAPL", "SPY"):
        for day, o, c in ((3, 100, 101), (4, 102, 104), (5, 105, 106)):
            conn.execute("INSERT INTO bars VALUES (%s, '1Day', %s, %s, %s, %s, %s, 1, 'test')",
                         (sym, datetime(2026, 8, day, 4, tzinfo=UTC), o, c, o, c))
    out = go(conn, StubAnalyst())
    assert out["scored"] == 1
    p = conn.execute("SELECT * FROM news_predictions").fetchone()
    assert p["scored_at"] is not None and p["entry_price"] == 102 and p["exit_price"] == 104   # Tuesday's bar


# ------------------------------------------------------------------ the command line


@pytest.fixture
def cli_env(test_db_url, monkeypatch):
    def use(**kw):
        monkeypatch.setattr(cli.Config, "from_env",
                            classmethod(lambda cls, env=None: Config(database_url=test_db_url, **kw)))
    return use


def test_the_command_without_yes_spends_nothing_and_says_what_it_would_do(conn, cli_env, capsys):
    add_window(conn, 4)
    cli_env(fake_broker=True)
    assert cli.main(["news-replay", "--since", "2026-08-03", "--until", "2026-08-10", "--max-usd", "20"]) == 0
    out = capsys.readouterr().out
    assert "4 waiting" in out and "$0.20" in out and "Nothing was spent" in out and "--yes" in out
    assert count(conn, "SELECT count(*) AS n FROM ai_calls") == 0
    assert count(conn, "SELECT count(*) AS n FROM news_analyses") == 0
    assert count(conn, "SELECT count(*) AS n FROM news WHERE analysis_status = 'backfill'") == 4


def test_the_dry_run_does_not_even_download_news(conn, cli_env, capsys):
    cli_env()
    assert cli.main(["news-replay", "--since", "2026-08-03", "--max-usd", "20"]) == 0
    assert count(conn, "SELECT count(*) AS n FROM news") == 0
    assert "Nothing is waiting yet" in capsys.readouterr().out


def test_the_command_refuses_a_window_before_the_cutoff(conn, cli_env, capsys):
    cli_env(fake_broker=True)
    assert cli.main(["news-replay", "--since", "2026-07-01", "--max-usd", "20", "--yes"]) == 1
    assert "2026-08-01" in capsys.readouterr().err
    assert count(conn, "SELECT count(*) AS n FROM news") == 0


def test_the_command_with_yes_runs_the_replay(conn, cli_env, capsys):
    cli_env(fake_broker=True)
    assert cli.main(["news-replay", "--since", "2026-08-03", "--until", "2026-08-04", "--max-usd", "20",
                     "--yes"]) == 0
    assert count(conn, "SELECT count(*) AS n FROM news_analyses WHERE source = 'replay'") > 0
    assert "replayed" in capsys.readouterr().out


def test_the_command_with_yes_and_no_key_refuses(conn, cli_env, capsys):
    cli_env()
    assert cli.main(["news-replay", "--since", "2026-08-03", "--max-usd", "20", "--yes"]) == 1
    assert "not connected" in capsys.readouterr().err
