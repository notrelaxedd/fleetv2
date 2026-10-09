"""Claude Haiku reviews futures models for the owner (docs/AI_PLAN.md, stage C).

Automatic: once a minute the coordinator looks for a model that is worth a look and
whose results changed since its last review:
- worth a look: it beats its coin-flip twin on the held-out prices, has a Final check,
  or is trading (Alpaca paper or Topstep);
- changed: a new backtest, a Final check, or five more paper days (review_key).
The owner can also ask for a review of any tested model on its page. At most
reviews_per_day reviews a day (config/ai.toml), automatic and asked-for together; asked
ones go first.

Haiku is shown what the model's page shows: its description and settings, the training
line, every held-out number with its coin-flip twin's, the chance it is luck, and the
"ready for a Combine" checklist (Final check, paper days). It answers in a fixed shape:
a one-line headline, a verdict word, what looks like luck, what looks fragile, and what
to watch on paper.

Advice only: a review never changes a model, a setting or a trade, and reviews are kept
in model_reviews, which the recipe writer (ai_ideas) never reads, so held-out numbers
never reach the ideas.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from coordinator import futures_models, futures_view, haiku
from coordinator.haiku import AiSettings
from fleet2.sim import topstep

log = logging.getLogger(__name__)

FEATURE = "reviews"
PAPER_DAYS_PER_REVIEW = 5
VERDICTS = ("promising", "unclear", "weak")
VERDICT_TEXT = {"promising": "Promising", "unclear": "Unclear", "weak": "Weak"}

SYSTEM = (
    "You review backtest results of a day-trading model for its owner, who is not a professional trader. The "
    "model trades CME micro index futures (MES, MNQ) and is meant to pass a Topstep 50K Combine: reach $3,000 "
    "profit before losing $2,000 from the best end-of-day balance, within about 60 trading days. Its numbers "
    "come from a held-out period the model search never saw, each beside a coin-flip twin that trades at the "
    "same times in random directions. Be honest and skeptical: say plainly when a result could be luck (few "
    "attempts, a pass rate close to the twin's, one big day doing most of the work, a high chance of luck, a "
    "loss at double slippage, a long losing stretch), and when something looks fragile. Do not recommend "
    "trading with real money; say what would make you more or less confident instead. Plain English, short "
    "sentences, no jargon, at most four points per list."
)


def schema() -> dict[str, Any]:
    points = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "headline": {"type": "string"},
            "verdict": {"type": "string", "enum": list(VERDICTS)},
            "luck": points,
            "fragile": points,
            "watch": points,
        },
        "required": ["headline", "verdict", "luck", "fragile", "watch"],
        "additionalProperties": False,
    }


# ------------------------------------------------------------------ which models, when


def paper_days(conn: psycopg.Connection, model_id: str) -> int:
    row = conn.execute("SELECT count(DISTINCT d.day) AS n FROM futures_days d JOIN futures_books b ON b.id = d.book_id "
                       "WHERE b.model_id = %s", (model_id,)).fetchone()
    return int(row["n"])


def review_key(conn: psycopg.Connection, m: dict[str, Any]) -> str:
    """What a review is of: the backtest, the Final check and the paper days so far (in
    steps of five), so a model is reviewed again when one of them changes."""
    check = futures_models.final_check(conn, m["id"])
    tested = m.get("backtested_at")
    return (f"backtest:{tested.isoformat() if tested else '-'}|final:{check['created_at'].isoformat() if check else '-'}"
            f"|paper:{paper_days(conn, m['id']) // PAPER_DAYS_PER_REVIEW}")


def trading(conn: psycopg.Connection, model_id: str) -> bool:
    return conn.execute("SELECT 1 FROM futures_books WHERE model_id = %s AND status IN ('active', 'closing') LIMIT 1",
                        (model_id,)).fetchone() is not None


def worth_reviewing(conn: psycopg.Connection, m: dict[str, Any], rules: topstep.Rules) -> bool:
    if m["market"] != "futures" or m["status"] == "retired" or not (m.get("metrics") or {}).get("held_out"):
        return False
    st = futures_view._status(m, rules)
    return bool(st["beats"]) or futures_models.final_check(conn, m["id"]) is not None or trading(conn, m["id"])


def reviews_today(conn: psycopg.Connection, now: datetime) -> int:
    row = conn.execute("SELECT count(*) AS n FROM model_reviews WHERE status IN ('done', 'failed') AND done_at > %s",
                       (now - timedelta(days=1),)).fetchone()
    return int(row["n"])


def request(conn: psycopg.Connection, model_id: str, asked_by: str = "owner") -> dict[str, Any]:
    """Ask for a review of one model (once: a waiting request is not doubled)."""
    m = conn.execute("SELECT * FROM models WHERE id = %s", (model_id,)).fetchone()
    if m is None or m["market"] != "futures":
        raise ValueError("Reviews are for futures models")
    if not (m.get("metrics") or {}).get("held_out"):
        raise ValueError("Run a backtest first: a review is of the model's held-out results")
    waiting = conn.execute("SELECT id FROM model_reviews WHERE model_id = %s AND status = 'requested'",
                           (model_id,)).fetchone()
    if waiting:
        return {"id": int(waiting["id"]), "message": "Claude Haiku's review of this model is already on its way"}
    row = conn.execute("INSERT INTO model_reviews (model_id, key, asked_by) VALUES (%s, %s, %s) RETURNING id",
                       (model_id, review_key(conn, m), asked_by)).fetchone()
    return {"id": int(row["id"]), "message": "Asked Claude Haiku for a review: it appears here within a minute or two"}


def queue_due(conn: psycopg.Connection, settings: AiSettings, rules: topstep.Rules, now: datetime) -> int:
    """Request reviews of the models that are worth a look and changed, within today's room."""
    room = settings.reviews_per_day - reviews_today(conn, now)
    room -= int(conn.execute("SELECT count(*) AS n FROM model_reviews WHERE status = 'requested'").fetchone()["n"])
    added = 0
    if room <= 0:
        return 0
    for m in conn.execute("SELECT * FROM models WHERE market = 'futures' AND status <> 'retired' AND metrics IS NOT NULL "
                          "ORDER BY backtested_at DESC NULLS LAST").fetchall():
        if added >= room:
            break
        if not worth_reviewing(conn, m, rules):
            continue
        key = review_key(conn, m)
        if conn.execute("SELECT 1 FROM model_reviews WHERE model_id = %s AND key = %s", (m["id"], key)).fetchone():
            continue  # reviewed (or tried) for these results already
        conn.execute("INSERT INTO model_reviews (model_id, key, asked_by) VALUES (%s, %s, 'auto')", (m["id"], key))
        added += 1
    return added


# ------------------------------------------------------------------ the review


def facts(conn: psycopg.Connection, m: dict[str, Any], rules: topstep.Rules, now: datetime) -> str:
    """The model's page, in words: what Haiku is asked to review."""
    tries = futures_models.tries(conn)
    check = futures_models.final_check(conn, m["id"])
    d = futures_view.detail(m, rules, tries, check, False, None, conn, None, now)
    lines = [f"Model: {d['name']}", f"What it does: {d['description']} {d['how_it_works']}",
             f"Where it came from: {d['origin']}", f"Settings: {d['settings']}"]
    if d.get("period"):
        lines.append(d["period"])
    if d.get("training_line"):
        lines.append(d["training_line"])
    lines.append("Held-out numbers (each with its coin-flip twin's where there is one):")
    for c in d.get("metrics") or []:
        line = f"- {c['label']}: {c['value']}"
        if c.get("note"):
            line += f" ({c['note']})"
        lines.append(line)
    verdict = d.get("verdict") or {}
    if verdict.get("items"):
        lines.append(f"Checklist for a Combine: {verdict.get('text')}")
        for item in verdict["items"]:
            lines.append(f"- {item['label']}: {item['state']}" + (f" ({item['note']})" if item.get("note") else ""))
    for venue, t in (d.get("trading") or {}).items():
        if t.get("book"):
            lines.append(f"Trading now ({venue}): {t['book']['line']}")
    return "\n".join(lines)


def run_one(conn: psycopg.Connection, settings: AiSettings, key: str, rules: topstep.Rules, client: Any = None,
            now: datetime | None = None) -> dict[str, Any]:
    """Write the oldest waiting review (owner requests first), within today's limit."""
    now = now or datetime.now(timezone.utc)
    row = conn.execute("SELECT * FROM model_reviews WHERE status = 'requested' "
                       "ORDER BY (asked_by = 'owner') DESC, id LIMIT 1").fetchone()
    if row is None:
        return {"reviewed": None}
    if reviews_today(conn, now) >= settings.reviews_per_day:
        return {"reviewed": None, "skipped": f"{settings.reviews_per_day} reviews in the last day already "
                                             "(reviews per_day in config/ai.toml)"}
    m = conn.execute("SELECT * FROM models WHERE id = %s", (row["model_id"],)).fetchone()
    user = facts(conn, m, rules, now) + "\n\nReview these results."
    answer = haiku.ask(conn, settings, key, FEATURE, SYSTEM, user, schema(), settings.review_effort,
                       settings.review_max_tokens, client=client, now=now)
    if answer["outcome"].startswith("skipped"):
        return {"reviewed": None, "skipped": answer["outcome"]}
    data = answer["data"]
    if data is None or data.get("verdict") not in VERDICTS:
        error = answer["outcome"] if data is None else "bad: no verdict in the answer"
        conn.execute("UPDATE model_reviews SET status = 'failed', error = %s, done_at = %s WHERE id = %s",
                     (error, now, row["id"]))
        return {"reviewed": row["model_id"], "error": error}
    review = {"headline": str(data.get("headline") or "").strip()[:300], "verdict": data["verdict"],
              **{k: [str(x).strip()[:400] for x in (data.get(k) or [])][:4] for k in ("luck", "fragile", "watch")},
              "cost": answer["cost"]}
    conn.execute("UPDATE model_reviews SET status = 'done', review = %s, done_at = %s WHERE id = %s",
                 (Jsonb(review), now, row["id"]))
    return {"reviewed": row["model_id"], "verdict": review["verdict"], "cost": answer["cost"]}


def tick(conn: psycopg.Connection, settings: AiSettings, key: str, rules: topstep.Rules, client: Any = None,
         now: datetime | None = None) -> dict[str, Any]:
    """One minute's work: queue the reviews that are due (if automatic), write one."""
    now = now or datetime.now(timezone.utc)
    if not key or not settings.review_enabled:
        return {"reviewed": None}
    if settings.review_auto:
        queue_due(conn, settings, rules, now)
    return run_one(conn, settings, key, rules, client, now)


# ------------------------------------------------------------------ the model page


def view(conn: psycopg.Connection, m: dict[str, Any], settings: AiSettings, key: str) -> dict[str, Any]:
    """What the model page shows under "Claude Haiku's review"."""
    out: dict[str, Any] = {"on": bool(key) and settings.review_enabled, "auto": settings.review_auto}
    if not (m.get("metrics") or {}).get("held_out"):
        out["state"] = "untested"
        return out
    if not out["on"]:
        out["state"] = "off"
        out["text"] = ("Claude Haiku can review this model's results once ANTHROPIC_API_KEY is in .env on box1."
                       if not key else "Claude Haiku's reviews are turned off in config/ai.toml.")
        return out
    rows = conn.execute("SELECT * FROM model_reviews WHERE model_id = %s ORDER BY id DESC LIMIT 2",
                        (m["id"],)).fetchall()
    waiting = next((r for r in rows if r["status"] == "requested"), None)
    done = conn.execute("SELECT * FROM model_reviews WHERE model_id = %s AND status = 'done' ORDER BY id DESC LIMIT 1",
                        (m["id"],)).fetchone()
    out["waiting"] = waiting is not None
    failed = rows[0] if rows and rows[0]["status"] == "failed" else None
    if failed is not None:
        out["failed"] = f"The last review could not be written: {failed['error']}"
    if done is None:
        out["state"] = "waiting" if waiting else "none"
        return out
    r = done["review"]
    when = done["done_at"]
    out.update(state="done", headline=r.get("headline"), verdict=VERDICT_TEXT.get(r.get("verdict"), "-"),
               verdict_key=r.get("verdict"), luck=r.get("luck") or [], fragile=r.get("fragile") or [],
               watch=r.get("watch") or [],
               when=f"{when:%b} {when.day}, {when:%H:%M} UTC" if when else "-",
               stale=done["key"] != review_key(conn, m),
               asked_by="you" if done["asked_by"] == "owner" else "automatically")
    return out
