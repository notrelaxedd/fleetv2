"""Post-cutoff replay: the honest backtest of the news engine (docs/NEWS_PLAN.md, section 1).

Claude remembers how most past news turned out, so a replay on old news would look
brilliant and prove nothing. Only news published on or after `limits.no_replay_before`
(after the model's training cutoff plus a margin, default 2026-08-01) is replayed. Each
such article is shown to Claude as if it were live: the article, plus the prices known
when it was published (research.build_context reads only bars that had already closed
by then), at a decision time a few minutes after publication. The predictions are saved
with source 'replay' and scored by news_scoring.score_pending exactly like live ones,
and the stats keep the two sources apart.

Things this module insists on:

- The window is checked first and refused, in plain words, if it reaches before
  no_replay_before or into the future, if it is empty, or if Claude is not connected.
- A replay spends real money, so it is capped by the owner's `max_usd` for the run and,
  as for every AI feature, by the monthly cap in config/ai.toml. It does not use the
  live daily cap (news_daily_cap_usd): that one protects a normal day of live news from
  a runaway, while a replay of months of news is a deliberate spend. Before each article
  it adds the most the call could cost to the spend so far and stops, saying which limit
  it was, rather than pass either.
- It can be stopped and run again. An article is replayed once: one that already has a
  'replay' analysis (even a failed one, which was perhaps paid for) is never sent again,
  so a second run carries on where the first stopped and never repeats a payment.
- Articles the live loop has already seen keep their status; only 'backfill' articles
  (stored for replay, never queued for the live analyst) are replayed.
- An article edited after its decision time (updated_at later than created_at +
  REPLAY_DELAY) is skipped, and counted as such: Alpaca serves only an article's latest
  text, which may already tell of the price reaction the replay is meant to predict.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable

import psycopg

from coordinator import haiku, news_feed, news_scoring, research
from coordinator.haiku import AiSettings
from coordinator.limits import Limits

log = logging.getLogger(__name__)

# The decision time of a replayed article is its publication time plus this. The live
# loop polls for news once a minute and Claude then takes some seconds, so a live
# decision also comes a few minutes after publication. Deciding at the publication time
# itself would flatter the replay with a speed the live engine does not have.
REPLAY_DELAY = timedelta(minutes=5)

PROGRESS_EVERY = 10          # a progress line every this many articles
MAX_FAILURES_IN_A_ROW = 3    # stop after this many failed analyses in a row (bad key, outage)
ESTIMATE_PER_ARTICLE_USD = 0.05  # a rough typical cost of one article, for the dry-run estimate
MAX_SCORING_BATCHES = 200   # a safety bound on the scoring loop
LOCK_KEY = 7_402_009         # one replay at a time (Postgres advisory lock)
EDITED_REASON = ("edited after the decision time: Alpaca gives only an article's latest text, which may already "
                 "tell of the price reaction")


def _utc(value: datetime) -> datetime:
    """A datetime in UTC (a naive one is taken to be UTC already)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def day_start(day: date) -> datetime:
    """Midnight UTC at the start of a date."""
    return datetime.combine(day, time.min, tzinfo=timezone.utc)


def check_window(limits: Limits, since: datetime, until: datetime, now: datetime) -> None:
    """Raise ValueError (a plain sentence) unless since..until may be replayed."""
    since, until, now = _utc(since), _utc(until), _utc(now)
    first = day_start(limits.no_replay_before)
    if since < first:
        raise ValueError(
            f"A replay cannot start before {limits.no_replay_before.isoformat()} (no_replay_before in "
            "config/limits.toml [news]): Claude may already know how earlier news turned out, so the result "
            "would prove nothing.")
    if until > now:
        raise ValueError("A replay cannot reach into the future: the end of the window is after now.")
    if since >= until:
        raise ValueError("The start of the window must be before its end.")


def window_counts(conn: psycopg.Connection, since: datetime, until: datetime) -> dict[str, int]:
    """What is stored for a window, without spending anything: `stored` articles of any
    status, `waiting` (status 'backfill', no replay analysis yet) and `replayed`
    (articles with a replay analysis)."""
    row = conn.execute(
        "SELECT count(*) AS stored, "
        "count(*) FILTER (WHERE analysis_status = 'backfill' AND NOT EXISTS ("
        "  SELECT 1 FROM news_analyses a WHERE a.news_id = news.id AND a.source = 'replay')) AS waiting, "
        "count(*) FILTER (WHERE EXISTS ("
        "  SELECT 1 FROM news_analyses a WHERE a.news_id = news.id AND a.source = 'replay')) AS replayed "
        "FROM news WHERE created_at >= %s AND created_at < %s", (_utc(since), _utc(until))).fetchone()
    return {k: int(row[k]) for k in ("stored", "waiting", "replayed")}


def _waiting_ids(conn: psycopg.Connection, since: datetime, until: datetime) -> list[int]:
    """The ids to replay, oldest first: 'backfill' articles in the window with no replay analysis."""
    rows = conn.execute(
        "SELECT id FROM news WHERE analysis_status = 'backfill' AND created_at >= %s AND created_at < %s "
        "AND NOT EXISTS (SELECT 1 FROM news_analyses a WHERE a.news_id = news.id AND a.source = 'replay') "
        "ORDER BY created_at, id", (_utc(since), _utc(until))).fetchall()
    return [int(r["id"]) for r in rows]


def replay(conn: psycopg.Connection, news_source: Any, analyst: Any, settings: AiSettings, limits: Limits,
           since: datetime, until: datetime, max_usd: float, now: datetime,
           log: Callable[[str], None] = print) -> dict[str, Any]:
    """Replay the news of since..until (until excluded) and score what can be scored.

    Returns {"fetched": articles newly stored by the backfill, "waiting": articles to
    replay when it started, "replayed": articles analysed, "failed": of those, analyses
    that failed, "skipped": articles not sent because they were edited after their
    decision time, "skipped_why": why (None when none was), "predictions": predictions
    saved, "cost": dollars spent by this run, "stopped_by": None or why it stopped early,
    "scored": predictions scored}.
    Raises ValueError when the window or the analyst is refused (see the module docstring).
    Commit after each article is the caller's connection's business: use an autocommit
    connection, so that an interruption never loses a payment already made."""
    since, until, now = _utc(since), _utc(until), _utc(now)
    check_window(limits, since, until, now)
    if not analyst.connected:
        raise ValueError(f"Claude is not connected, so nothing can be replayed: {analyst.problem or 'no API key'}.")
    if not max_usd > 0:
        raise ValueError("max_usd must be more than zero.")
    got = conn.execute("SELECT pg_try_advisory_lock(%s) AS ok", (LOCK_KEY,)).fetchone()["ok"]
    if not got:
        raise ValueError("Another replay is already running.")
    try:
        return _run(conn, news_source, analyst, settings, limits, since, until, float(max_usd), now, log)
    finally:
        conn.execute("SELECT pg_advisory_unlock(%s)", (LOCK_KEY,))


def _run(conn: psycopg.Connection, news_source: Any, analyst: Any, settings: AiSettings, limits: Limits,
         since: datetime, until: datetime, max_usd: float, now: datetime,
         log: Callable[[str], None]) -> dict[str, Any]:
    fetched = news_feed.backfill(conn, news_source, since, until)
    ids = _waiting_ids(conn, since, until)
    log(f"replay {since:%Y-%m-%d %H:%M} to {until:%Y-%m-%d %H:%M} UTC: {fetched} new articles stored, "
        f"{len(ids)} waiting to be replayed, up to ${max_usd:,.2f} to spend")
    out: dict[str, Any] = {"fetched": fetched, "waiting": len(ids), "replayed": 0, "failed": 0, "skipped": 0,
                           "skipped_why": None, "predictions": 0, "cost": 0.0, "stopped_by": None, "scored": 0}
    in_a_row = 0
    for done, news_id in enumerate(ids):
        if done and done % PROGRESS_EVERY == 0:
            log(f"replayed {out['replayed']} of {len(ids)} articles, {_skipped(out)}{out['predictions']} "
                f"predictions, ${out['cost']:,.2f} spent")
        row = conn.execute("SELECT * FROM news WHERE id = %s", (news_id,)).fetchone()
        decided_at = row["created_at"] + REPLAY_DELAY
        if row["updated_at"] is not None and row["updated_at"] > decided_at:
            # Marked once, so later runs and the dry run's count leave it out.
            conn.execute("UPDATE news SET analysis_status = 'skipped', analysis_note = %s WHERE id = %s",
                         (EDITED_REASON, news_id))
            out["skipped"] += 1
            out["skipped_why"] = EDITED_REASON
            continue
        context = research.build_context(conn, row, decided_at, None)
        worst = float(analyst.worst_cost(row, context))
        if worst > 0:
            if out["cost"] + worst > max_usd:
                out["stopped_by"] = (f"the --max-usd limit of ${max_usd:,.2f} (spent ${out['cost']:,.2f}; the next "
                                     f"article could cost up to ${worst:,.2f})")
            elif haiku.month_spend(conn, now) + worst > settings.monthly_cap_usd:
                out["stopped_by"] = (f"this month's cap of ${settings.monthly_cap_usd:,.2f} (monthly_cap_usd in "
                                     "config/ai.toml)")
            if out["stopped_by"]:
                break
        saved = research.analyse_article(conn, analyst, row, context, decided_at, source="replay",
                                         settings=settings, clock=now)
        made = conn.execute("SELECT cost_usd, error FROM news_analyses WHERE news_id = %s AND source = 'replay' "
                            "ORDER BY id DESC LIMIT 1", (news_id,)).fetchone()
        if made is None:  # analyse_article held back by a cap we did not foresee: nothing was spent
            out["stopped_by"] = "a spending cap (see the log)"
            break
        out["replayed"] += 1
        out["cost"] += float(made["cost_usd"])
        out["predictions"] += len(saved)
        if made["error"]:
            out["failed"] += 1
            in_a_row += 1
            if in_a_row >= MAX_FAILURES_IN_A_ROW:
                out["stopped_by"] = f"{in_a_row} analyses in a row failed (the last: {made['error']})"
                break
        else:
            in_a_row = 0
    # score_pending does one batch of at most BATCH predictions; a replay can leave more.
    for _ in range(MAX_SCORING_BATCHES):
        n = news_scoring.score_pending(conn, limits, now)
        out["scored"] += n
        if n < news_scoring.BATCH:
            break
    log(f"replayed {out['replayed']} of {len(ids)} articles, {_skipped(out)}{out['predictions']} predictions, "
        f"${out['cost']:,.2f} spent, {out['scored']} predictions scored"
        + (f"; stopped by {out['stopped_by']}" if out["stopped_by"] else ""))
    return out


def _skipped(out: dict[str, Any]) -> str:
    """The skipped articles for a progress line ("" when none were)."""
    return f"{out['skipped']} skipped ({out['skipped_why']}), " if out["skipped"] else ""
