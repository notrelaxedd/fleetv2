"""Claude Haiku writes recipes for futures model search (docs/AI_PLAN.md, stage B).

What Haiku is shown: the building blocks (fleet2/models/futures/recipe.py), the goal in
words, and the recipes tried so far with their TRAINING numbers only (recipe_ideas.result
holds nothing else: record_results keeps only TRAIN_KEYS). Held-out and lockbox numbers
are never in that table and never in the prompt, so Haiku can never tune its ideas to
the tests that judge them.

What it answers: a JSON list of recipes (structured output, so the shape is fixed), each
with one sentence on the idea behind it. Each must pass recipe.validate() and not be a
recipe already tried, queued or kept; the rest wait in recipe_ideas until a search round
takes them (take), beside the round's random recipes.

Money: every call is recorded in ai_calls with its cost, worked out from the token counts
and the prices in config/ai.toml. A call is only made when this month's spend plus the
most the call could cost (its prompt, plus max_tokens of answer) stays within
monthly_cap_usd.

Only the coordinator holds ANTHROPIC_API_KEY. Nothing here places an order or changes a
model: recipes go through the same search, backtest, Final check and paper trading as
everything else.
"""
from __future__ import annotations

import json
import logging
import tomllib
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import psycopg
from psycopg.types.json import Jsonb

from fleet2.models.futures import recipe as R

log = logging.getLogger(__name__)

FEATURE = "recipes"
EFFORTS = ("low", "medium", "high")
# The training numbers kept per recipe, and so the only numbers Haiku ever sees.
TRAIN_KEYS = ("tried", "best_score", "passes", "days_traded", "pnl_double", "pnl_normal")
HISTORY_BEST, HISTORY_RECENT = 25, 15


@dataclass(frozen=True)
class AiSettings:
    monthly_cap_usd: float = 5.0
    enabled: bool = True
    model: str = "claude-haiku-5-5"
    per_call: int = 4
    queue: int = 8
    calls_per_hour: int = 6
    per_round: int = 3
    effort: str = "low"
    max_tokens: int = 8000
    input_per_million: float = 0.10
    output_per_million: float = 0.50

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_million + output_tokens * self.output_per_million) / 1e6

    def worst_cost(self, input_tokens: int) -> float:
        """The most a call with this prompt can cost: its whole answer at max_tokens."""
        return self.cost(input_tokens, self.max_tokens)


def load_settings(path: Path) -> AiSettings:
    """config/ai.toml (defaults when the file is missing)."""
    if not Path(path).is_file():
        return AiSettings()
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    flat: dict[str, Any] = {}
    for section in ("spend", "recipes", "prices"):
        flat.update(raw.get(section) or {})
    known = {f.name for f in fields(AiSettings)}
    s = AiSettings(**{k: v for k, v in flat.items() if k in known})
    if s.effort not in EFFORTS:
        raise ValueError(f"effort in {path} must be one of {', '.join(EFFORTS)}")
    if min(s.monthly_cap_usd, s.input_per_million, s.output_per_million) < 0 or min(s.per_call, s.max_tokens) < 1:
        raise ValueError(f"the numbers in {path} must be positive")
    return s


# ------------------------------------------------------------------ spend and when to call


def month_start(now: datetime) -> datetime:
    return now.astimezone(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def month_spend(conn: psycopg.Connection, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    row = conn.execute("SELECT coalesce(sum(cost_usd), 0) AS s FROM ai_calls WHERE at >= %s",
                       (month_start(now),)).fetchone()
    return float(row["s"])


def record_call(conn: psycopg.Connection, model: str, input_tokens: int, output_tokens: int, cost: float,
                outcome: str, detail: dict[str, Any] | None = None, at: datetime | None = None) -> None:
    conn.execute("INSERT INTO ai_calls (at, feature, model, input_tokens, output_tokens, cost_usd, outcome, detail) "
                 "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                 (at or datetime.now(timezone.utc), FEATURE, model, int(input_tokens), int(output_tokens), float(cost),
                  outcome[:300], Jsonb(detail or {})))


def futures_search_running(conn: psycopg.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM jobs WHERE kind = 'model_search' AND status IN ('queued', 'leased', 'cancel_requested') "
        "AND params->'markets' = '[\"futures\"]'::jsonb LIMIT 1").fetchone() is not None


def queued(conn: psycopg.Connection) -> int:
    return int(conn.execute("SELECT count(*) AS n FROM recipe_ideas WHERE by = 'haiku' AND status = 'queued'")
               .fetchone()["n"])


def why_not(conn: psycopg.Connection, settings: AiSettings, key: str, now: datetime | None = None,
            prompt_tokens: int = 4000) -> str | None:
    """Why Haiku should not write recipes now (None: it may)."""
    now = now or datetime.now(timezone.utc)
    if not key:
        return "no ANTHROPIC_API_KEY in .env on box1"
    if not settings.enabled:
        return "turned off in config/ai.toml"
    if not futures_search_running(conn):
        return "no futures model search is running"
    if queued(conn) >= settings.queue:
        return "enough recipes are waiting"
    recent = conn.execute("SELECT count(*) AS n FROM ai_calls WHERE feature = %s AND at > %s - interval '1 hour'",
                          (FEATURE, now)).fetchone()["n"]
    if recent >= settings.calls_per_hour:
        return f"{settings.calls_per_hour} calls this hour already"
    if month_spend(conn, now) + settings.worst_cost(prompt_tokens) > settings.monthly_cap_usd:
        return f"this month's cap of ${settings.monthly_cap_usd:,.2f} (monthly_cap_usd in config/ai.toml) is reached"
    return None


# ------------------------------------------------------------------ the prompt


def catalog() -> str:
    """The building blocks in words, straight from recipe.py."""
    lines = ["SIGNALS (pick one; an 'up' and a 'down' event):"]
    for name, s in sorted(R.SIGNALS.items()):
        lines.append(f"- {name}: up when {s['up']}; down when {s['down']}")
    lines.append("DIRECTION: follow (buy on up, sell short on down) or fade (sell short on up, buy on down)")
    lines.append(f"FILTERS (pick 0 to {R.MAX_FILTERS}; a trade must also meet them):")
    for name, f in sorted(R.FILTERS.items()):
        lines.append(f"- {name}: {f['words']}")
    lines.append("Never together: " + "; ".join(" and ".join(sorted(p)) for p in R.CLASHES))
    lines.append("EXITS (pick one; every trade also has a stop, a target and the close):")
    for name, e in sorted(R.EXITS.items()):
        lines.append(f"- {name}: {e['words']}")
    lines.append("ENTRIES: first (only the day's first signal) or every (any signal)")
    lines.append("SIDE: both, long (buys only) or short (sells short only)")
    lines.append("Numbers (thresholds, minutes, stop and target ticks, bar size of 1 to 15 minutes, MES or MNQ) "
                 "are tuned later by the search; you choose only the blocks.")
    return "\n".join(lines)


SYSTEM = (
    "You design day-trading ideas for the CME micro E-mini S&P 500 (MES) and micro Nasdaq-100 (MNQ) futures, "
    "built only from fixed building blocks. A program tunes each idea's numbers on the training years and keeps "
    "it only if it makes money there at double the normal costs, trades on at least 100 days, and its worst "
    "quarter of the training years is still good (the score is the worst of four parts' daily Sharpe ratio). "
    "The end goal is a Topstep 50K Combine: reach $3,000 profit before losing $2,000 from the best end-of-day "
    "balance, within about 60 trading days, with every position closed each day by 15:00 Chicago time. "
    "Prefer ideas with a plain market reason behind them (a known intraday habit of index futures), and ideas "
    "unlike those already tried. Write each idea's reason as one short plain-English sentence, no jargon."
)


def _blocks(r: dict[str, Any]) -> str:
    f = "+".join(r["filters"]) or "no filter"
    return f"{r['direction']} {r['signal']}, {f}, exit {r['exit']}, {r['entries']}, {r['side']}"


def _result_text(res: dict[str, Any]) -> str:
    score = res.get("best_score")
    parts = [f"{int(res.get('tried') or 0)} settings tried",
             "best score " + ("none (never traded enough)" if score is None else f"{float(score):.3f}"),
             "passed every check" if res.get("passes") else "did not pass"]
    if res.get("days_traded") is not None:
        parts.append(f"{int(res['days_traded'])} days traded")
    if res.get("pnl_double") is not None:
        parts.append(f"${float(res['pnl_double']):,.0f} at double costs")
    return ", ".join(parts)


def history(conn: psycopg.Connection) -> tuple[list[dict[str, Any]], list[str]]:
    """(tried recipes to show: the best and the most recent, with training numbers only;
    names of the recipes waiting or being tried, not to be written again)."""
    best = conn.execute(
        "SELECT family, recipe, by, result FROM recipe_ideas WHERE status = 'tried' "
        "ORDER BY (result->>'best_score')::float DESC NULLS LAST, id DESC LIMIT %s", (HISTORY_BEST,)).fetchall()
    recent = conn.execute(
        "SELECT family, recipe, by, result FROM recipe_ideas WHERE status = 'tried' ORDER BY tried_at DESC, id DESC "
        "LIMIT %s", (HISTORY_RECENT,)).fetchall()
    seen: dict[str, dict[str, Any]] = {}
    for row in [*best, *recent]:
        seen.setdefault(row["family"], row)
    pending = [r["family"] for r in conn.execute(
        "SELECT family FROM recipe_ideas WHERE status IN ('queued', 'taken') ORDER BY id").fetchall()]
    return list(seen.values()), pending


def prompt(conn: psycopg.Connection, settings: AiSettings) -> str:
    tried, pending = history(conn)
    lines = ["The building blocks:", catalog(), ""]
    if tried:
        lines.append("Ideas tried so far, with their results on the training years:")
        for row in tried:
            who = "yours" if row["by"] == "haiku" else "random"
            lines.append(f"- [{who}] {_blocks(row['recipe'])}: {_result_text(row['result'] or {})}")
    else:
        lines.append("No ideas have been tried yet.")
    if pending:
        lines.append(f"Already waiting to be tried (do not repeat): {len(pending)} ideas.")
    lines += ["", f"Write {settings.per_call} new ideas, each different from the ones above."]
    return "\n".join(lines)


def schema() -> dict[str, Any]:
    item = {
        "type": "object",
        "properties": {
            "signal": {"type": "string", "enum": sorted(R.SIGNALS)},
            "direction": {"type": "string", "enum": list(R.DIRECTIONS)},
            "filters": {"type": "array", "items": {"type": "string", "enum": sorted(R.FILTERS)}},
            "exit": {"type": "string", "enum": sorted(R.EXITS)},
            "entries": {"type": "string", "enum": list(R.ENTRIES)},
            "side": {"type": "string", "enum": list(R.SIDES)},
            "idea": {"type": "string"},
        },
        "required": ["signal", "direction", "filters", "exit", "entries", "side", "idea"],
        "additionalProperties": False,
    }
    return {"type": "object", "properties": {"recipes": {"type": "array", "items": item}},
            "required": ["recipes"], "additionalProperties": False}


# ------------------------------------------------------------------ the call


def make_client(key: str) -> Any:
    import anthropic

    return anthropic.Anthropic(api_key=key, max_retries=2, timeout=120.0)


def _known(conn: psycopg.Connection) -> set[str]:
    rows = conn.execute("SELECT family AS name FROM recipe_ideas UNION SELECT module FROM models "
                        "WHERE market = 'futures'").fetchall()
    return {r["name"] for r in rows}


def _error_text(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return "error: Anthropic refused ANTHROPIC_API_KEY in .env on box1"
    if status is not None:
        return f"error: Anthropic answered {status}"
    return f"error: {type(exc).__name__}: {str(exc)[:200]}"


def write_recipes(conn: psycopg.Connection, settings: AiSettings, key: str, client: Any = None,
                  now: datetime | None = None, make: Callable[[str], Any] = make_client) -> dict[str, Any]:
    """One call to Haiku, when allowed: queue the good recipes it writes. Never raises for
    an API problem (it is recorded in ai_calls and shown on the dashboard)."""
    now = now or datetime.now(timezone.utc)
    user = prompt(conn, settings)
    estimate = (len(SYSTEM) + len(user)) // 3 + 200  # generous: about 4 characters per token
    reason = why_not(conn, settings, key, now, estimate)
    if reason:
        return {"written": 0, "skipped": reason}
    client = client or make(key)
    try:
        response = client.messages.create(
            model=settings.model,
            max_tokens=settings.max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": settings.effort, "format": {"type": "json_schema", "schema": schema()}},
        )
    except Exception as exc:  # noqa: BLE001 - recorded and shown, never crashes the coordinator
        outcome = _error_text(exc)
        record_call(conn, settings.model, 0, 0, 0.0, outcome, at=now)
        log.warning("Claude Haiku recipes: %s", outcome)
        return {"written": 0, "error": outcome}
    usage = response.usage
    in_tokens = int(getattr(usage, "input_tokens", 0) or 0) + int(getattr(usage, "cache_read_input_tokens", 0) or 0) \
        + int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    out_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    cost = settings.cost(in_tokens, out_tokens)
    if response.stop_reason == "refusal":
        record_call(conn, settings.model, in_tokens, out_tokens, cost, "refused", at=now)
        return {"written": 0, "error": "refused", "cost": cost}
    if response.stop_reason == "max_tokens":
        record_call(conn, settings.model, in_tokens, out_tokens, cost, "bad: the answer was cut off (max_tokens)", at=now)
        return {"written": 0, "error": "cut off", "cost": cost}
    text = next((b.text for b in response.content if getattr(b, "type", "") == "text"), "")
    try:
        items = json.loads(text)["recipes"]
        if not isinstance(items, list):
            raise ValueError("recipes is not a list")
    except (ValueError, KeyError, TypeError) as exc:
        record_call(conn, settings.model, in_tokens, out_tokens, cost, f"bad: unreadable answer ({exc})"[:120], at=now)
        return {"written": 0, "error": "unreadable", "cost": cost}
    known = _known(conn)
    written: list[str] = []
    dropped: list[str] = []
    for item in items[: settings.per_call]:
        try:
            raw = dict(item)
            note = str(raw.pop("idea", "") or "").strip()[:300]
            r = R.validate(raw)
        except (R.BadRecipe, TypeError, ValueError) as exc:
            dropped.append(f"not a valid recipe: {exc}")
            continue
        name = R.name_of(r)
        if name in known:
            dropped.append(f"{name} was already tried, queued or kept")
            continue
        known.add(name)
        conn.execute("INSERT INTO recipe_ideas (family, recipe, by, note) VALUES (%s, %s, 'haiku', %s)",
                     (name, Jsonb(r), note))
        written.append(name)
    record_call(conn, settings.model, in_tokens, out_tokens, cost, "ok",
                {"written": written, "dropped": dropped}, at=now)
    return {"written": len(written), "names": written, "dropped": dropped, "cost": cost}


# ------------------------------------------------------------------ what searches take and report


def take(conn: psycopg.Connection, job_id: str | None, count: int) -> list[dict[str, Any]]:
    """Up to `count` waiting Haiku recipes for a search round (oldest first). Recipes
    taken by a search that has since ended go back to the queue first."""
    conn.execute(
        "UPDATE recipe_ideas SET status = 'queued', job_id = NULL, taken_at = NULL WHERE status = 'taken' AND "
        "(job_id IS NULL OR job_id NOT IN (SELECT id FROM jobs WHERE status IN ('queued', 'leased', 'cancel_requested')))")
    if count <= 0:
        return []
    rows = conn.execute(
        """
        UPDATE recipe_ideas SET status = 'taken', job_id = %s::uuid, taken_at = now()
         WHERE id IN (SELECT id FROM recipe_ideas WHERE status = 'queued' ORDER BY id LIMIT %s FOR UPDATE SKIP LOCKED)
        RETURNING id, family, recipe, note
        """, (job_id, int(count))).fetchall()
    return [{"id": int(r["id"]), "family": r["family"], "recipe": r["recipe"], "note": r["note"]} for r in rows]


def _training_only(entry: dict[str, Any]) -> dict[str, Any]:
    out = {k: entry.get(k) for k in TRAIN_KEYS}
    out["tried"] = int(out["tried"] or 0)
    out["passes"] = bool(out["passes"])
    for k in ("best_score", "pnl_double", "pnl_normal"):
        out[k] = None if out[k] is None else float(out[k])
    out["days_traded"] = None if out["days_traded"] is None else int(out["days_traded"])
    return out


def _merge(old: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    """The better of two results (by best score), with the tries added up."""
    if not old:
        return new
    better = new if (new.get("best_score") or float("-inf")) > (old.get("best_score") or float("-inf")) else old
    return {**better, "tried": int(old.get("tried") or 0) + int(new.get("tried") or 0)}


def record_results(conn: psycopg.Connection, job_id: str | None, recipes: list[dict[str, Any]]) -> int:
    """A search round's training results per recipe. A Haiku recipe (idea_id) is marked
    tried; a random one is recorded once and kept up to date, so Haiku learns from both."""
    stored = 0
    for entry in recipes:
        try:
            r = R.validate(entry.get("recipe"))
        except R.BadRecipe:
            continue
        name = R.name_of(r)
        if name != entry.get("name"):
            continue
        result = _training_only(entry)
        idea_id = entry.get("idea_id")
        if idea_id is not None:
            row = conn.execute("SELECT result FROM recipe_ideas WHERE id = %s AND family = %s FOR UPDATE",
                               (int(idea_id), name)).fetchone()
            if row is None:
                continue
            conn.execute("UPDATE recipe_ideas SET status = 'tried', tried_at = now(), result = %s, tries = tries + %s "
                         "WHERE id = %s", (Jsonb(_merge(row["result"], result)), result["tried"], int(idea_id)))
        else:
            row = conn.execute("SELECT id, result FROM recipe_ideas WHERE family = %s AND by = 'random' FOR UPDATE",
                               (name,)).fetchone()
            if row is None:
                conn.execute("INSERT INTO recipe_ideas (family, recipe, by, status, job_id, result, tries, tried_at) "
                             "VALUES (%s, %s, 'random', 'tried', %s::uuid, %s, %s, now())",
                             (name, Jsonb(r), job_id, Jsonb(result), result["tried"]))
            else:
                conn.execute("UPDATE recipe_ideas SET result = %s, tries = tries + %s, tried_at = now() WHERE id = %s",
                             (Jsonb(_merge(row["result"], result)), result["tried"], row["id"]))
        stored += 1
    return stored


# ------------------------------------------------------------------ the dashboard line


def status_line(conn: psycopg.Connection, settings: AiSettings, key: str, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    if not key:
        return {"tone": "muted", "text": "Claude Haiku is off. To let it write recipes for model search, add "
                                         "ANTHROPIC_API_KEY to .env on box1 (see docs/AI_PLAN.md)."}
    if not settings.enabled:
        return {"tone": "muted", "text": "Claude Haiku is turned off in config/ai.toml."}
    spent = month_spend(conn, now)
    counts = conn.execute(
        """
        SELECT count(*) FILTER (WHERE created_at >= %s) AS written,
               count(*) FILTER (WHERE status = 'tried') AS tried,
               count(*) FILTER (WHERE status = 'queued') AS waiting
          FROM recipe_ideas WHERE by = 'haiku'
        """, (month_start(now),)).fetchone()
    kept = conn.execute(
        "SELECT count(*) FILTER (WHERE metrics->>'recipe_by' = 'haiku') AS haiku, "
        "count(*) FILTER (WHERE metrics->>'recipe_by' = 'random') AS random FROM models "
        "WHERE market = 'futures' AND origin = 'search' AND status = 'backtested'").fetchone()
    text = (f"Claude Haiku: {counts['written']} recipes written this month, {counts['tried']} tried, "
            f"{counts['waiting']} waiting · recipe models kept now: {kept['haiku']} by Haiku, {kept['random']} random · "
            f"${spent:,.2f} of ${settings.monthly_cap_usd:,.2f} spent this month")
    tone = "plain"
    if spent + settings.worst_cost(4000) > settings.monthly_cap_usd:
        text += " · stopped until next month (monthly_cap_usd in config/ai.toml)"
        tone = "warn"
    last = conn.execute("SELECT outcome FROM ai_calls WHERE feature = %s ORDER BY id DESC LIMIT 1",
                        (FEATURE,)).fetchone()
    if last and last["outcome"].startswith(("error", "refused", "bad")):
        text += f" · last call: {last['outcome']}"
        tone = "warn"
    return {"tone": tone, "text": text}
