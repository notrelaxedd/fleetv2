"""Settings table access (copied from v1, trimmed).

v2 keeps only a handful of runtime values in the database: fleet timing, the pause
switch and the owner's live confirmation. Money limits live in config/limits.toml
(coordinator.limits), so there is one file to read and edit.
"""
from __future__ import annotations

from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from coordinator.events import add_audit

JOB_KINDS = ("sleep", "data_refresh", "backtest", "paper_trade", "model_search")


def get_settings(conn: psycopg.Connection) -> dict[str, Any]:
    """All settings as a plain dict."""
    rows = conn.execute("SELECT key, value FROM settings ORDER BY key").fetchall()
    return {row["key"]: row["value"] for row in rows}


def get_setting(conn: psycopg.Connection, key: str, default: Any = None) -> Any:
    """One setting value, or `default` when the key is missing."""
    row = conn.execute("SELECT value FROM settings WHERE key = %s", (key,)).fetchone()
    return default if row is None else row["value"]


def get_int_setting(conn: psycopg.Connection, key: str, default: int) -> int:
    """One numeric setting coerced to int."""
    value = get_setting(conn, key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def put_setting(conn: psycopg.Connection, key: str, value: Any, actor: str | None, action: str,
                confirmation_text: str | None = None) -> None:
    """Store one setting and write an audit row (internal callers only: pause, live)."""
    before = get_setting(conn, key)
    conn.execute(
        "INSERT INTO settings (key, value, updated_at) VALUES (%s, %s, now())"
        " ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
        (key, Jsonb(value)),
    )
    add_audit(conn, action, key, actor, {key: before}, {key: value}, confirmation_text=confirmation_text)
