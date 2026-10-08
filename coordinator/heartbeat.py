"""Register and heartbeat transactions (the worker side of the contract).

Copied from v1 (host/heartbeat.py). Changed: no roles, epochs, reboot or trade slots;
the heartbeat stores CPU %, RAM %, temperature (null = no sensor) and each running
job's one-line detail, and claims at most one job when the worker asks for one.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import psycopg

from coordinator import auth
from coordinator.errors import Unauthorized
from coordinator.events import add_audit
from coordinator.leases import claim, job_payload, lease_seconds, release, renew
from coordinator.recovery import held_jobs, orphan_jobs
from coordinator.settings import get_int_setting, get_setting


def server_time(conn: psycopg.Connection) -> datetime:
    """Database clock, timezone aware."""
    return conn.execute("SELECT now() AS t").fetchone()["t"].astimezone(timezone.utc)


def trading_paused(conn: psycopg.Connection) -> bool:
    """The pause-all-trading switch (only the JSON boolean true counts)."""
    return get_setting(conn, "trading_paused", False) is True


def _common_reply(conn: psycopg.Connection) -> dict[str, Any]:
    """Fields shared by register and heartbeat responses."""
    return {
        "paused": trading_paused(conn),
        "server_time": server_time(conn),
        "heartbeat_seconds": get_int_setting(conn, "heartbeat_seconds", 5),
    }


def _machine_fields(body: dict[str, Any], remote_ip: str | None) -> tuple[Any, ...]:
    return (body.get("hostname"), body.get("python_version"), body.get("code_version"), body.get("boot_id"), remote_ip)


def _enroll(conn: psycopg.Connection, body: dict[str, Any], remote_ip: str | None) -> dict[str, Any]:
    """First registration via an enroll token: create the worker (name must be unique)."""
    token_row = auth.lock_enroll_token(conn, str(body.get("enroll_token", "")))
    worker_id = auth.new_worker_id()
    while conn.execute("SELECT 1 FROM workers WHERE id = %s", (worker_id,)).fetchone():
        worker_id = auth.new_worker_id()
    name = body.get("name") or body.get("hostname") or worker_id
    if conn.execute("SELECT 1 FROM workers WHERE name = %s", (name,)).fetchone():
        raise Unauthorized(f"a worker named {name!r} is already enrolled; pick another name with --name")
    plain = auth.mint_token()
    worker = conn.execute(
        """
        INSERT INTO workers (id, name, token_hash, hostname, python_version, code_version, boot_id, remote_ip)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING *
        """,
        (worker_id, name, auth.hash_token(plain)) + _machine_fields(body, remote_ip),
    ).fetchone()
    auth.mark_enroll_token_used(conn, token_row["token_hash"], worker_id)
    add_audit(conn, "worker_enrolled", worker_id, worker_id, None, {"name": name, "hostname": body.get("hostname")}, ip=remote_ip)
    worker["_plain_token"] = plain
    return worker


def _reregister(conn: psycopg.Connection, body: dict[str, Any], remote_ip: str | None) -> dict[str, Any]:
    """Re-registration: verify the current (or previous) token under the row lock, then rotate it."""
    worker_id = str(body.get("worker_id", ""))
    _before, presented = auth.verify_register_token(conn, worker_id, str(body.get("worker_token", "")))
    plain = auth.rotate_worker_token(conn, worker_id, presented)
    worker = conn.execute(
        """
        UPDATE workers SET hostname = COALESCE(%s, hostname), python_version = %s,
               code_version = %s, boot_id = %s, remote_ip = %s
         WHERE id = %s RETURNING *
        """,
        _machine_fields(body, remote_ip) + (worker_id,),
    ).fetchone()
    worker["_plain_token"] = plain
    return worker


def register(conn: psycopg.Connection, body: dict[str, Any], remote_ip: str | None = None) -> dict[str, Any]:
    """POST /api/v1/workers/register, one transaction."""
    if body.get("enroll_token"):
        worker = _enroll(conn, body, remote_ip)
    elif body.get("worker_id") and body.get("worker_token"):
        worker = _reregister(conn, body, remote_ip)
    else:
        raise Unauthorized("enroll_token or worker_id + worker_token required")
    lease = lease_seconds(conn)
    held = held_jobs(conn, worker["id"], lease)
    reply = _common_reply(conn)
    reply.update(
        {
            "worker_id": worker["id"],
            "worker_token": worker["_plain_token"],
            "held_jobs": [job_payload(row, lease) for row in held],
        }
    )
    return reply


def _update_worker(conn: psycopg.Connection, worker_id: str, body: dict[str, Any], token_hash: str | None) -> dict[str, Any] | None:
    """Record the heartbeat and lock the worker row; the token is checked inside the
    locking UPDATE (as in v1) so a heartbeat that raced a register cannot act."""
    return conn.execute(
        """
        UPDATE workers SET last_heartbeat_at = now(), cpu_pct = %(cpu)s, ram_pct = %(ram_pct)s,
               ram_used_mb = %(ram_used)s, ram_total_mb = %(ram_total)s, temp_c = %(temp)s,
               code_version = COALESCE(%(code)s, code_version), skew_ms = %(skew)s,
               prev_token_hash = NULL
         WHERE id = %(wid)s AND (%(hash)s::text IS NULL OR token_hash = %(hash)s)
         RETURNING *
        """,
        {
            "cpu": body.get("cpu_pct"),
            "ram_pct": body.get("ram_pct"),
            "ram_used": body.get("ram_used_mb"),
            "ram_total": body.get("ram_total_mb"),
            "temp": body.get("temp_c"),
            "code": body.get("code_version"),
            "skew": body.get("skew_ms"),
            "wid": worker_id,
            "hash": token_hash,
        },
    ).fetchone()


def _preempt_ids(conn: psycopg.Connection, worker_id: str) -> tuple[list[str], list[str]]:
    """Jobs this worker must stop and hand back, as (preempt, cancel)."""
    rows = conn.execute(
        """
        SELECT id, status FROM jobs
         WHERE lease_worker_id = %s AND status IN ('leased', 'cancel_requested')
           AND (preempt_requested OR status = 'cancel_requested')
         ORDER BY created_at
        """,
        (worker_id,),
    ).fetchall()
    preempt = [str(row["id"]) for row in rows]
    cancel = [str(row["id"]) for row in rows if row["status"] == "cancel_requested"]
    return preempt, cancel


def _reported_ids(body: dict[str, Any]) -> list[str]:
    """Every job id the heartbeat mentions (running or released)."""
    ids: list[str] = []
    for key in ("jobs", "released"):
        for entry in body.get(key) or []:
            if entry.get("id"):
                ids.append(str(entry["id"]))
    return ids


def process_heartbeat(conn: psycopg.Connection, worker_id: str, body: dict[str, Any], token_hash: str | None = None) -> dict[str, Any]:
    """POST /api/v1/workers/{id}/heartbeat, one transaction: record stats, renew,
    release, stop requests, then claim one job if the worker asked for one."""
    worker = _update_worker(conn, worker_id, body, token_hash)
    if worker is None:
        raise Unauthorized("invalid worker token")
    lease = lease_seconds(conn)
    lost: list[str] = []
    for entry in body.get("jobs") or []:
        live = "progress" in entry and entry.get("progress") is None and entry.get("detail") is not None
        ok = renew(conn, worker_id, entry.get("id"), entry.get("lease_token"), lease,
                   entry.get("progress"), entry.get("checkpoint"), entry.get("detail"), live)
        if not ok:
            lost.append(str(entry.get("id")))
    for entry in body.get("released") or []:
        release(conn, entry.get("id"), entry.get("lease_token"), entry.get("progress"),
                entry.get("checkpoint"), worker_id, entry.get("reason"))
    preempt, cancel = _preempt_ids(conn, worker_id)
    claimed: list[dict[str, Any]] = []
    if worker["enabled"] and body.get("want_job"):
        # A lease this worker holds but did not report (its claim reply was lost) is
        # handed back first; nothing new is claimed while one exists (v1 rule).
        orphans = orphan_jobs(conn, worker_id, _reported_ids(body), lease)
        if orphans:
            claimed = [job_payload(row, lease) for row in orphans]
        else:
            row = claim(conn, worker_id, lease)
            claimed = [job_payload(row, lease)] if row else []
    reply = _common_reply(conn)
    reply.update({"preempt": preempt, "cancel": cancel, "lost": lost, "claimed": claimed})
    return reply
