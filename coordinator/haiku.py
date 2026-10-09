"""Claude Haiku on the coordinator: settings, the spending cap, and one way to ask
(docs/AI_PLAN.md). Used by the recipe writer (ai_ideas), the model reviews (ai_reviews)
and the daily market note (market_note).

ask() is the only place that calls Anthropic. Before a call it adds the most the call
could cost (its prompt, plus max_tokens of answer, at the prices in config/ai.toml) to
what this calendar month has cost so far, and makes no call if that would pass
monthly_cap_usd. After a call it records the real cost from the token counts in
ai_calls, whatever happened. Answers have a fixed shape (structured output). Problems
(a refused key, a busy server, a refusal, a cut-off answer) are recorded and returned,
never raised, so the coordinator carries on.

Only the coordinator holds ANTHROPIC_API_KEY.
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

log = logging.getLogger(__name__)

EFFORTS = ("low", "medium", "high")
NO_KEY = "no ANTHROPIC_API_KEY in .env on box1"


@dataclass(frozen=True)
class AiSettings:
    # [spend]
    monthly_cap_usd: float = 5.0
    # [recipes]
    enabled: bool = True
    model: str = "claude-haiku-5-5"
    per_call: int = 4
    queue: int = 8
    calls_per_hour: int = 6
    per_round: int = 3
    effort: str = "low"
    max_tokens: int = 8000
    # [reviews]
    review_enabled: bool = True
    review_auto: bool = True
    reviews_per_day: int = 20
    review_effort: str = "low"
    review_max_tokens: int = 6000
    # [market_note]
    note_enabled: bool = True
    note_time: str = "07:45"
    note_headlines: int = 60
    note_effort: str = "low"
    note_max_tokens: int = 4000
    # [prices]
    input_per_million: float = 0.10
    output_per_million: float = 0.50

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_million + output_tokens * self.output_per_million) / 1e6

    def worst_cost(self, input_tokens: int, max_tokens: int | None = None) -> float:
        """The most a call with this prompt can cost: its whole answer at max_tokens."""
        return self.cost(input_tokens, self.max_tokens if max_tokens is None else max_tokens)


# Section and key in config/ai.toml -> setting.
_RENAMED = {
    "reviews": {"enabled": "review_enabled", "auto": "review_auto", "per_day": "reviews_per_day",
                "effort": "review_effort", "max_tokens": "review_max_tokens"},
    "market_note": {"enabled": "note_enabled", "time": "note_time", "headlines": "note_headlines",
                    "effort": "note_effort", "max_tokens": "note_max_tokens"},
}


def load_settings(path: Path) -> AiSettings:
    """config/ai.toml (defaults when the file is missing)."""
    if not Path(path).is_file():
        return AiSettings()
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    flat: dict[str, Any] = {}
    for section in ("spend", "recipes", "prices"):
        flat.update(raw.get(section) or {})
    for section, names in _RENAMED.items():
        for key, value in (raw.get(section) or {}).items():
            if key in names:
                flat[names[key]] = value
    known = {f.name for f in fields(AiSettings)}
    s = AiSettings(**{k: v for k, v in flat.items() if k in known})
    for name in ("effort", "review_effort", "note_effort"):
        if getattr(s, name) not in EFFORTS:
            raise ValueError(f"{name} in {path} must be one of {', '.join(EFFORTS)}")
    if min(s.monthly_cap_usd, s.input_per_million, s.output_per_million) < 0 or \
            min(s.per_call, s.max_tokens, s.review_max_tokens, s.note_max_tokens) < 1:
        raise ValueError(f"the numbers in {path} must be positive")
    try:
        hh, mm = (int(x) for x in s.note_time.split(":"))
        if not (0 <= hh < 24 and 0 <= mm < 60):
            raise ValueError
    except ValueError:
        raise ValueError(f'market_note time in {path} must look like "07:45" (Chicago time)') from None
    return s


# ------------------------------------------------------------------ spend


def month_start(now: datetime) -> datetime:
    return now.astimezone(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def month_spend(conn: psycopg.Connection, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    row = conn.execute("SELECT coalesce(sum(cost_usd), 0) AS s FROM ai_calls WHERE at >= %s",
                       (month_start(now),)).fetchone()
    return float(row["s"])


def record_call(conn: psycopg.Connection, feature: str, model: str, input_tokens: int, output_tokens: int,
                cost: float, outcome: str, detail: dict[str, Any] | None = None,
                at: datetime | None = None) -> int:
    row = conn.execute(
        "INSERT INTO ai_calls (at, feature, model, input_tokens, output_tokens, cost_usd, outcome, detail) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
        (at or datetime.now(timezone.utc), feature, model, int(input_tokens), int(output_tokens), float(cost),
         outcome[:300], Jsonb(detail or {}))).fetchone()
    return int(row["id"])


def over_cap(conn: psycopg.Connection, settings: AiSettings, prompt_tokens: int, max_tokens: int,
             now: datetime) -> str | None:
    """The reason no call can be made this month, or None."""
    if month_spend(conn, now) + settings.worst_cost(prompt_tokens, max_tokens) > settings.monthly_cap_usd:
        return f"this month's cap of ${settings.monthly_cap_usd:,.2f} (monthly_cap_usd in config/ai.toml) is reached"
    return None


def tokens_of(*texts: str) -> int:
    """A generous guess at a prompt's tokens (about 3 characters each, plus a margin)."""
    return sum(len(t) for t in texts) // 3 + 200


# ------------------------------------------------------------------ asking


def make_client(key: str) -> Any:
    import anthropic

    return anthropic.Anthropic(api_key=key, max_retries=2, timeout=120.0)


def error_text(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return "error: Anthropic refused ANTHROPIC_API_KEY in .env on box1"
    if status is not None:
        return f"error: Anthropic answered {status}"
    return f"error: {type(exc).__name__}: {str(exc)[:200]}"


def ask(conn: psycopg.Connection, settings: AiSettings, key: str, feature: str, system: str, user: str,
        schema: dict[str, Any], effort: str, max_tokens: int, client: Any = None, now: datetime | None = None,
        make: Callable[[str], Any] = make_client) -> dict[str, Any]:
    """One call to Haiku. Returns {"data": the answer as a dict or None, "outcome": "ok",
    "refused", "skipped: ...", "error: ..." or "bad: ...", "cost", "call_id"}."""
    now = now or datetime.now(timezone.utc)
    if not key:
        return {"data": None, "outcome": "skipped: " + NO_KEY, "cost": 0.0, "call_id": None}
    capped = over_cap(conn, settings, tokens_of(system, user), max_tokens, now)
    if capped:
        return {"data": None, "outcome": "skipped: " + capped, "cost": 0.0, "call_id": None}
    client = client or make(key)
    try:
        response = client.messages.create(
            model=settings.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        )
    except Exception as exc:  # noqa: BLE001 - recorded and shown, never crashes the coordinator
        outcome = error_text(exc)
        call_id = record_call(conn, feature, settings.model, 0, 0, 0.0, outcome, at=now)
        log.warning("Claude Haiku (%s): %s", feature, outcome)
        return {"data": None, "outcome": outcome, "cost": 0.0, "call_id": call_id}
    usage = response.usage
    in_tokens = sum(int(getattr(usage, k, 0) or 0)
                    for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    out_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    cost = settings.cost(in_tokens, out_tokens)
    data = None
    if response.stop_reason == "refusal":
        outcome = "refused"
    elif response.stop_reason == "max_tokens":
        outcome = "bad: the answer was cut off (max_tokens)"
    else:
        text = next((b.text for b in response.content if getattr(b, "type", "") == "text"), "")
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("not an object")
            outcome = "ok"
        except ValueError as exc:
            data, outcome = None, f"bad: unreadable answer ({exc})"[:120]
    call_id = record_call(conn, feature, settings.model, in_tokens, out_tokens, cost, outcome, at=now)
    return {"data": data, "outcome": outcome, "cost": cost, "call_id": call_id}


def note_detail(conn: psycopg.Connection, call_id: int | None, detail: dict[str, Any]) -> None:
    """Add what came of a call (e.g. which recipes were kept) to its record."""
    if call_id is not None:
        conn.execute("UPDATE ai_calls SET detail = %s WHERE id = %s", (Jsonb(detail), call_id))


def last_problem(conn: psycopg.Connection, feature: str) -> str | None:
    row = conn.execute("SELECT outcome FROM ai_calls WHERE feature = %s ORDER BY id DESC LIMIT 1",
                       (feature,)).fetchone()
    if row and row["outcome"].startswith(("error", "refused", "bad")):
        return row["outcome"]
    return None
