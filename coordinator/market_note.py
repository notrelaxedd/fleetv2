"""Claude Haiku's daily market note (docs/AI_PLAN.md, stage D, an experiment).

On each CME trading day, at note_time Chicago time (config/ai.toml, 07:45 to start),
the coordinator reads the headlines since the last session's close from Alpaca's news
feed (the Alpaca paper keys already in .env; made-up headlines in demo mode), and Claude
Haiku writes a short note: the day's scheduled events that move index futures, whether
the news is quiet, normal or heavy, and a flag for a day unusual enough that patterns
from ordinary days may not hold.

Why it is only tested going forward: Haiku learned from text that covers past market
days, so on old headlines it may "remember" what happened next. A note written before
the session opens can only use news known then. So only notes written before the open
count (before_open) when the paper results of futures models on flagged days are
compared with the other days, and the Futures view says how many days that comparison
rests on. The note is advice only: nothing trades on it, and the recipe writer never
reads it.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from coordinator import haiku
from coordinator.haiku import AiSettings
from fleet2.sim import cme_session

log = logging.getLogger(__name__)

FEATURE = "market_note"
LEVELS = ("quiet", "normal", "heavy")
ENOUGH_DAYS = 40  # notes written before the open needed before the comparison means anything

SYSTEM = (
    "You write a short note before the US stock index futures session for the owner of an automated day-trading "
    "system that trades CME micro E-mini S&P 500 (MES) and micro Nasdaq-100 (MNQ) futures within the day. From "
    "the headlines given (and only those: never add events they do not mention), list today's scheduled events "
    "that usually move index futures, with their time if a headline gives it (Fed decisions and speeches, "
    "inflation, jobs and other economic reports, large company results, Treasury auctions), say whether the "
    "news flow looks quiet, normal or heavy, and flag the day only if it looks unusual enough that intraday "
    "patterns from ordinary days may not hold. Plain English, no jargon, no trading advice."
)


def schema() -> dict[str, Any]:
    event = {"type": "object", "properties": {"time": {"type": "string"}, "what": {"type": "string"}},
             "required": ["time", "what"], "additionalProperties": False}
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "events": {"type": "array", "items": event},
            "news_level": {"type": "string", "enum": list(LEVELS)},
            "flag": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["summary", "events", "news_level", "flag", "reason"],
        "additionalProperties": False,
    }


# ------------------------------------------------------------------ headlines


class AlpacaNews:
    """Headlines from Alpaca's news feed (alpaca-py's NewsClient, the paper keys)."""

    name = "Alpaca news"

    def __init__(self, key_id: str, secret: str, client: Any = None) -> None:
        if client is None:
            from alpaca.data.historical.news import NewsClient

            client = NewsClient(key_id, secret, raw_data=True)
        self._client = client

    def headlines(self, start: datetime, end: datetime, limit: int) -> list[dict[str, Any]]:
        from alpaca.data.requests import NewsRequest

        raw = self._client.get_news(NewsRequest(start=start, end=end, limit=limit, sort="desc",
                                                exclude_contentless=False))
        items = raw.get("news", []) if isinstance(raw, dict) else []
        out = []
        for a in items[:limit]:
            out.append({"time": str(a.get("created_at") or ""), "headline": str(a.get("headline") or "").strip(),
                        "summary": str(a.get("summary") or "").strip(), "source": str(a.get("source") or ""),
                        "symbols": list(a.get("symbols") or [])})
        return [h for h in out if h["headline"]]


class FakeNews:
    """Made-up headlines for demo mode, labelled so."""

    name = "made-up demo headlines"

    def headlines(self, start: datetime, end: datetime, limit: int) -> list[dict[str, Any]]:
        t = end - timedelta(hours=2)
        return [
            {"time": t.isoformat(), "headline": "Demo: consumer prices report due at 7:30 a.m. Chicago time",
             "summary": "Made-up headline for demo mode.", "source": "demo", "symbols": []},
            {"time": (t - timedelta(hours=3)).isoformat(), "headline": "Demo: Asian shares mixed overnight",
             "summary": "Made-up headline for demo mode.", "source": "demo", "symbols": []},
        ][:limit]


def make_news(config: Any) -> Any:
    if getattr(config, "fake_broker", False):
        return FakeNews()
    if config.alpaca_paper_key_id and config.alpaca_paper_secret:
        return AlpacaNews(config.alpaca_paper_key_id, config.alpaca_paper_secret)
    return None


# ------------------------------------------------------------------ when


def chicago_now(now: datetime) -> datetime:
    return now.astimezone(cme_session.CHICAGO)


def due(conn: psycopg.Connection, settings: AiSettings, now: datetime) -> date | None:
    """Today's Chicago date when today's note is due and not written yet, else None."""
    local = chicago_now(now)
    day = local.date()
    sess = cme_session.session(day)
    if sess is None or now.timestamp() >= sess[1]:
        return None  # not a trading day, or the session is over
    hh, mm = (int(x) for x in settings.note_time.split(":"))
    if (local.hour, local.minute) < (hh, mm):
        return None
    if conn.execute("SELECT 1 FROM market_notes WHERE day = %s", (day,)).fetchone():
        return None
    return day


def last_close(day: date) -> datetime:
    """The close of the trading day before `day`."""
    d = day - timedelta(days=1)
    while not cme_session.is_trading_day(d):
        d -= timedelta(days=1)
    return datetime.fromtimestamp(cme_session.session(d)[1], timezone.utc)


# ------------------------------------------------------------------ the note


def _headline_lines(headlines: list[dict[str, Any]]) -> list[str]:
    lines = []
    for h in headlines:
        try:
            t = chicago_now(datetime.fromisoformat(h["time"].replace("Z", "+00:00"))).strftime("%a %H:%M")
        except ValueError:
            t = "?"
        text = h["headline"]
        if h.get("summary") and h["summary"] != text:
            text += " - " + h["summary"][:200]
        lines.append(f"- {t} [{h.get('source') or '?'}] {text}")
    return lines


def write_note(conn: psycopg.Connection, settings: AiSettings, key: str, news: Any, client: Any = None,
               now: datetime | None = None) -> dict[str, Any]:
    """Write today's note when it is due. Never raises for a news or API problem."""
    now = now or datetime.now(timezone.utc)
    if not key or not settings.note_enabled:
        return {"written": False, "skipped": "off"}
    if news is None:
        return {"written": False, "skipped": "no news source (Alpaca paper keys in .env)"}
    day = due(conn, settings, now)
    if day is None:
        return {"written": False, "skipped": "not due"}
    try:
        headlines = news.headlines(last_close(day), now, settings.note_headlines)
    except Exception as exc:  # noqa: BLE001 - shown on the dashboard, tried again later
        log.warning("market note: reading the news failed: %s", exc)
        return {"written": False, "error": f"reading the news failed: {str(exc)[:200]}"}
    sess_open = datetime.fromtimestamp(cme_session.session(day)[0], timezone.utc)
    local = chicago_now(now)
    user = "\n".join([f"Today is {day:%A, %B} {day.day}, {day.year}. It is {local:%H:%M} Chicago time; the session "
                      f"opens at {chicago_now(sess_open):%H:%M}.",
                      f"Headlines since the last session's close ({len(headlines)}, newest first, Chicago time):",
                      *(_headline_lines(headlines) or ["- (none)"]), "", "Write today's note."])
    answer = haiku.ask(conn, settings, key, FEATURE, SYSTEM, user, schema(), settings.note_effort,
                       settings.note_max_tokens, client=client, now=now)
    data = answer["data"]
    if data is None or data.get("news_level") not in LEVELS:
        return {"written": False, "error": answer["outcome"] if data is None else "bad: no news level"}
    note = {"summary": str(data.get("summary") or "").strip()[:800],
            "events": [{"time": str(e.get("time") or "")[:40], "what": str(e.get("what") or "")[:300]}
                       for e in (data.get("events") or []) if isinstance(e, dict)][:12],
            "news_level": data["news_level"], "flag": bool(data.get("flag")),
            "reason": str(data.get("reason") or "").strip()[:400], "source": getattr(news, "name", "news"),
            "cost": answer["cost"]}
    conn.execute("INSERT INTO market_notes (day, written_at, before_open, headlines, flagged, note) "
                 "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (day) DO NOTHING",
                 (day, now, now < sess_open, len(headlines), note["flag"], Jsonb(note)))
    return {"written": True, "day": day, "flagged": note["flag"], "cost": answer["cost"]}


# ------------------------------------------------------------------ the forward test


def comparison(conn: psycopg.Connection) -> dict[str, Any]:
    """Futures paper and Topstep results per model and day, on days whose note (written
    before the open) was flagged against the other noted days."""
    notes = conn.execute("SELECT count(*) AS n, count(*) FILTER (WHERE flagged) AS f FROM market_notes "
                         "WHERE before_open").fetchone()
    rows = conn.execute(
        """
        SELECT n.flagged, count(*) AS n, avg(d.pnl) AS avg
          FROM market_notes n JOIN futures_days d ON d.day = n.day
         WHERE n.before_open
         GROUP BY n.flagged
        """).fetchall()
    by = {bool(r["flagged"]): {"n": int(r["n"]), "avg": float(r["avg"])} for r in rows}
    return {"days": int(notes["n"]), "flagged_days": int(notes["f"]), "flagged": by.get(True),
            "other": by.get(False), "enough": int(notes["n"]) >= ENOUGH_DAYS}


def view(conn: psycopg.Connection, settings: AiSettings, key: str, news: Any,
         now: datetime | None = None) -> dict[str, Any]:
    """The Futures view's "Market note" panel."""
    now = now or datetime.now(timezone.utc)
    if not key:
        return {"on": False, "text": "Claude Haiku can write a short note before each session from the morning's "
                                     "headlines once ANTHROPIC_API_KEY is in .env on box1."}
    if not settings.note_enabled:
        return {"on": False, "text": "The daily market note is turned off in config/ai.toml."}
    if news is None:
        return {"on": False, "text": "The daily market note needs the Alpaca paper keys in .env on box1 (it reads "
                                     "Alpaca's news feed)."}
    row = conn.execute("SELECT * FROM market_notes ORDER BY day DESC LIMIT 1").fetchone()
    out: dict[str, Any] = {"on": True}
    today = chicago_now(now).date()
    if row is None:
        out["text"] = f"No note yet: the first one is written at {settings.note_time} Chicago time on a trading day."
    else:
        n = row["note"]
        when = chicago_now(row["written_at"])
        label = "Today's" if row["day"] == today else f"{row['day']:%a %b} {row['day'].day}'s"
        out.update(title=f"{label} market note", summary=n.get("summary"), events=n.get("events") or [],
                   level=n.get("news_level"), flagged=bool(row["flagged"]), reason=n.get("reason"),
                   written=(f"Written by Claude Haiku at {when:%H:%M} Chicago time from {row['headlines']} "
                            f"headlines ({n.get('source', 'news')})"
                            + ("" if row["before_open"] else ", after the open, so it does not count below")))
    c = comparison(conn)

    def avg(part: dict[str, Any] | None) -> str:
        if not part:
            return "no paper days yet"
        return f"{part['n']} model-day{'s' if part['n'] != 1 else ''}, average ${part['avg']:,.2f}"

    out["comparison"] = (f"Forward test: {c['days']} notes written before the open, {c['flagged_days']} flagged. "
                         f"Futures paper results on flagged days: {avg(c['flagged'])}; on other days: {avg(c['other'])}.")
    if not c["enough"]:
        out["comparison"] += f" Too few days to mean anything yet ({ENOUGH_DAYS} needed)."
    problem = haiku.last_problem(conn, FEATURE)
    if problem and not (row is not None and row["day"] == today):  # today's note was written after all
        out["problem"] = f"Last note call: {problem}"
    return out
