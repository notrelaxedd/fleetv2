"""Owner-driven scheduling: job creation, targeting, cancel, run again, dispatcher.

Copied from v1 (host/scheduling.py) and simplified: v2 has no worker roles, so aiming
a job at a worker no longer flips the worker's role. Targets:
- "auto": the least busy online worker (no job, lowest CPU); if none is free the job
  waits and the dispatcher places it as soon as one is.
- "all_idle": one job per idle online worker.
- a worker id: that worker runs it next.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from coordinator.errors import BadRequest, Conflict, NotFound
from coordinator.events import add_job_event
from coordinator.leases import get_job
from coordinator.settings import JOB_KINDS, get_int_setting

AUTO = "auto"
ALL_IDLE = "all_idle"


@dataclass
class CreateResult:
    """Outcome of create_job: the jobs made (one, or one per idle worker)."""

    jobs: list[dict[str, Any]] = field(default_factory=list)
    waiting: bool = False


def get_worker(conn: psycopg.Connection, worker_id: str, for_update: bool = False) -> dict[str, Any]:
    """Fetch one worker row; 404 when missing."""
    sql = "SELECT * FROM workers WHERE id = %s" + (" FOR UPDATE" if for_update else "")
    row = conn.execute(sql, (worker_id,)).fetchone()
    if row is None:
        raise NotFound("worker not found")
    return row


def online_after(conn: psycopg.Connection) -> int:
    """Seconds after which a silent worker counts as offline."""
    return get_int_setting(conn, "online_after_seconds", 20)


IDLE_SQL = """
    SELECT * FROM workers w
     WHERE enabled
       AND last_heartbeat_at > now() - make_interval(secs => %(online)s)
       AND NOT EXISTS (SELECT 1 FROM jobs j WHERE (j.target_worker_id = w.id AND j.status = 'queued')
                       OR (j.lease_worker_id = w.id AND j.status IN ('leased', 'cancel_requested')))
"""
PICK_RETRIES = 4
PICK_WAIT_S = 0.02


def idle_workers(conn: psycopg.Connection) -> list[dict[str, Any]]:
    """Every online, enabled worker with no job running or waiting for it, by name."""
    return conn.execute(IDLE_SQL + " ORDER BY name", {"online": online_after(conn)}).fetchall()


def least_busy_pick(conn: psycopg.Connection) -> dict[str, Any] | None:
    """Lock and return the idle worker with the lowest CPU (ties: most recent heartbeat).

    SKIP LOCKED keeps two pickers off the same worker; a worker's own heartbeat holds
    its row lock for a few milliseconds, so a few short retries (as in v1) let an
    "auto" job land at creation instead of waiting for the dispatcher.
    """
    for attempt in range(PICK_RETRIES + 1):
        row = conn.execute(
            IDLE_SQL + " ORDER BY cpu_pct NULLS LAST, last_heartbeat_at DESC LIMIT 1 FOR UPDATE SKIP LOCKED",
            {"online": online_after(conn)},
        ).fetchone()
        if row is not None or attempt == PICK_RETRIES:
            return row
        time.sleep(PICK_WAIT_S)
    return None


def target_job_at(conn: psycopg.Connection, job: dict[str, Any], worker: dict[str, Any], auto: bool) -> dict[str, Any]:
    """Point a queued job at a worker. `auto` marks a system pick (Auto, dispatcher)."""
    job = conn.execute(
        "UPDATE jobs SET target_worker_id = %s, target_auto = %s, updated_at = now() WHERE id = %s RETURNING *",
        (worker["id"], auto, job["id"]),
    ).fetchone()
    add_job_event(conn, job["id"], "targeted", worker["id"], {"auto": auto})
    return job


def _insert_job(conn: psycopg.Connection, kind: str, params: dict[str, Any], model_id: str | None,
                idempotency_key: str | None, retry_of: Any = None) -> dict[str, Any]:
    row = conn.execute(
        """
        INSERT INTO jobs (kind, params, model_id, idempotency_key, retry_of, progress)
        VALUES (%s, %s, %s, %s, %s, CASE WHEN %s = 'paper_trade' THEN NULL ELSE 0 END) RETURNING *
        """,
        (kind, Jsonb(params), model_id, idempotency_key, retry_of, kind),
    ).fetchone()
    add_job_event(conn, row["id"], "created", None, {"kind": kind})
    return row


def create_job(
    conn: psycopg.Connection,
    kind: str,
    params: dict[str, Any] | None = None,
    target: str | None = AUTO,
    model_id: str | None = None,
    idempotency_key: str | None = None,
    retry_of: Any = None,
) -> CreateResult:
    """Create one job (or one per idle worker for "all_idle"); caller owns the transaction."""
    if kind not in JOB_KINDS:
        raise BadRequest(f"unknown job kind: {kind!r}")
    params = params if params is not None else {}
    if not isinstance(params, dict):
        raise BadRequest("params must be a JSON object")
    if idempotency_key:
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (idempotency_key,))
        existing = conn.execute("SELECT * FROM jobs WHERE idempotency_key = %s", (idempotency_key,)).fetchone()
        if existing is not None:
            return CreateResult([existing])
    target = target or AUTO
    if target == ALL_IDLE:
        workers = idle_workers(conn)
        if not workers:
            raise Conflict("No worker is idle right now")
        jobs = []
        for worker in workers:
            job = _insert_job(conn, kind, params, model_id, None, retry_of)
            jobs.append(target_job_at(conn, job, worker, auto=False))
        return CreateResult(jobs)
    if target == AUTO:
        job = _insert_job(conn, kind, params, model_id, idempotency_key, retry_of)
        worker = least_busy_pick(conn)
        if worker is None:
            return CreateResult([job], waiting=True)
        return CreateResult([target_job_at(conn, job, worker, auto=True)])
    worker = get_worker(conn, target, for_update=True)
    job = _insert_job(conn, kind, params, model_id, idempotency_key, retry_of)
    return CreateResult([target_job_at(conn, job, worker, auto=False)])


def cancel_job(conn: psycopg.Connection, job_id: Any) -> dict[str, Any]:
    """queued -> cancelled; leased -> cancel_requested (the worker stops it within one
    heartbeat); a finished job -> 409."""
    job = get_job(conn, job_id, for_update=True)
    status = job["status"]
    if status in ("cancelled", "cancel_requested"):
        return job
    if status == "queued":
        row = conn.execute(
            "UPDATE jobs SET status = 'cancelled', finished_at = now(), updated_at = now() WHERE id = %s RETURNING *",
            (job["id"],),
        ).fetchone()
        add_job_event(conn, job["id"], "cancelled")
        return row
    if status == "leased":
        row = conn.execute(
            "UPDATE jobs SET status = 'cancel_requested', updated_at = now() WHERE id = %s RETURNING *",
            (job["id"],),
        ).fetchone()
        add_job_event(conn, job["id"], "cancel_requested", job["lease_worker_id"])
        return row
    raise Conflict(f"job is {status}")


def run_again(conn: psycopg.Connection, job_id: Any) -> CreateResult:
    """A failed or cancelled job as a new queued job with the same kind, params and
    model, sent to the least busy worker (its old worker may still be offline)."""
    job = get_job(conn, job_id)
    if job["status"] not in ("failed", "cancelled"):
        raise Conflict("only a failed or cancelled job can be run again")
    return create_job(conn, job["kind"], dict(job["params"] or {}), AUTO, job["model_id"], retry_of=job["id"])


def dispatch(conn: psycopg.Connection) -> list[dict[str, Any]]:
    """Aim waiting unaimed jobs, oldest first, at the least busy idle workers."""
    jobs = conn.execute(
        """
        SELECT * FROM jobs WHERE status = 'queued' AND target_worker_id IS NULL AND run_after <= now()
         ORDER BY created_at FOR UPDATE SKIP LOCKED
        """
    ).fetchall()
    assigned: list[dict[str, Any]] = []
    for job in jobs:
        worker = least_busy_pick(conn)
        if worker is None:
            break
        assigned.append(target_job_at(conn, job, worker, auto=True))
    return assigned


def release_dead_targets(conn: psycopg.Connection) -> list[dict[str, Any]]:
    """A queued job aimed by Auto at a worker that has since gone offline is unaimed
    again so the dispatcher can place it elsewhere (a job aimed by the owner waits)."""
    return conn.execute(
        """
        UPDATE jobs j SET target_worker_id = NULL, target_auto = false, updated_at = now()
          FROM workers w
         WHERE j.status = 'queued' AND j.target_auto AND w.id = j.target_worker_id
           AND (w.last_heartbeat_at IS NULL OR w.last_heartbeat_at < now() - make_interval(secs => %s))
        RETURNING j.id
        """,
        (online_after(conn),),
    ).fetchall()
