"""Writers for job_events and audit_log rows."""
from __future__ import annotations

from typing import Any

import psycopg
from psycopg.types.json import Jsonb


def add_job_event(
    conn: psycopg.Connection,
    job_id: Any,
    event: str,
    worker_id: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    """Append one job_events row."""
    conn.execute(
        "INSERT INTO job_events (job_id, worker_id, event, detail) VALUES (%s, %s, %s, %s)",
        (job_id, worker_id, event, Jsonb(detail) if detail is not None else None),
    )


def add_audit(
    conn: psycopg.Connection,
    action: str,
    entity: str | None,
    actor: str | None = None,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    ip: str | None = None,
    confirmation_text: str | None = None,
) -> None:
    """Append one audit_log row (`confirmation_text` is what the owner typed, e.g. RESUME)."""
    conn.execute(
        "INSERT INTO audit_log (actor, ip, action, entity, before, after, confirmation_text)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (
            actor,
            ip,
            action,
            entity,
            Jsonb(before) if before is not None else None,
            Jsonb(after) if after is not None else None,
            confirmation_text,
        ),
    )
