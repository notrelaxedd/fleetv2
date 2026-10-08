"""Safety controls. Stage 1: the pause switch. Stage 4 adds the daily loss limit, the
per-model limits and approve_and_place(), the one path every order takes.

Pausing stops every model from placing orders at once; backtests and model searches
keep running. Only the owner resumes: the coordinator never resumes on its own.
"""
from __future__ import annotations

import psycopg

from coordinator.settings import get_setting, put_setting


def is_paused(conn: psycopg.Connection) -> bool:
    """True while trading is paused (only the JSON boolean true counts)."""
    return get_setting(conn, "trading_paused", False) is True


def pause_trading(conn: psycopg.Connection, actor: str | None, reason: str) -> bool:
    """Pause all trading; True when this call changed it (an audit row is written)."""
    conn.execute("SELECT pg_advisory_xact_lock(7402001)")
    if is_paused(conn):
        return False
    put_setting(conn, "trading_paused", True, actor, "trading_paused")
    put_setting(conn, "paused_reason", reason, actor, "trading_paused_reason")
    return True


def resume_trading(conn: psycopg.Connection, actor: str | None) -> bool:
    """Resume trading; True when this call changed it."""
    conn.execute("SELECT pg_advisory_xact_lock(7402001)")
    if not is_paused(conn):
        return False
    put_setting(conn, "trading_paused", False, actor, "trading_resumed")
    put_setting(conn, "paused_reason", None, actor, "trading_paused_reason")
    return True
