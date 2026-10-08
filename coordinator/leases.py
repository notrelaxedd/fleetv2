"""Lease lifecycle: claim, renew, release, checkpoint, complete, fail (copied from v1).

Recovery paths (orphaned leases, held jobs on register, the reaper) are in
coordinator.recovery.

Changed from v1: no roles, so a worker claims any queued job aimed at it, or an
unaimed one. A release is final: `cancel` ends the job as cancelled, every other
reason (the worker shut down, its runner was stopped, it ran out of memory) fails it
with a plain reason, and the owner can run it again from the dashboard. v1 quietly
requeued such jobs instead.
"""
from __future__ import annotations

import uuid
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from coordinator.errors import Conflict, NotFound
from coordinator.events import add_job_event
from coordinator.settings import get_int_setting

ACTIVE = ("leased", "cancel_requested")
RELEASE_REASONS = ("preempt", "cancel", "oom", "shutdown", "stopped", "update")
RELEASE_ERRORS = {
    "oom": "Stopped: the job used too much memory on {worker}",
    "shutdown": "Stopped: the agent on {worker} was shut down",
    "stopped": "Stopped: the job on {worker} was stopped from outside",
    "preempt": "Stopped on {worker}",
    None: "Stopped on {worker}",
}


def as_uuid(value: Any) -> uuid.UUID | None:
    """Parse a uuid-ish value; None when it is not a valid uuid."""
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return None


def lease_seconds(conn: psycopg.Connection) -> int:
    """Current lease length from settings."""
    return get_int_setting(conn, "lease_seconds", 30)


def job_payload(row: dict[str, Any], lease: int) -> dict[str, Any]:
    """What a worker needs to run a job it was just handed."""
    return {
        "id": str(row["id"]),
        "kind": row["kind"],
        "params": row["params"],
        "checkpoint": row["checkpoint"],
        "progress": row.get("progress"),
        "model_id": row.get("model_id"),
        "lease_token": str(row["lease_token"]),
        "lease_seconds": lease,
    }


def _fence(job: dict[str, Any], lease_token: Any, worker_id: str | None) -> uuid.UUID:
    """The parsed token when it matches the job's active lease held by `worker_id`, else 409."""
    tok = as_uuid(lease_token)
    if job["status"] not in ACTIVE or tok is None or job["lease_token"] != tok:
        raise Conflict("lease token mismatch or job not leased")
    if worker_id is not None and job["lease_worker_id"] != worker_id:
        raise Conflict("job is leased to another worker")
    return tok


def get_job(conn: psycopg.Connection, job_id: Any, for_update: bool = False) -> dict[str, Any]:
    """Fetch one job row; 404 when missing or the id is not a uuid."""
    jid = as_uuid(job_id)
    if jid is None:
        raise NotFound("job not found")
    sql = "SELECT * FROM jobs WHERE id = %s" + (" FOR UPDATE" if for_update else "")
    row = conn.execute(sql, (jid,)).fetchone()
    if row is None:
        raise NotFound("job not found")
    return row


def claim(conn: psycopg.Connection, worker_id: str, lease: int) -> dict[str, Any] | None:
    """Claim at most one queued job for the worker: jobs aimed at it first, then
    unaimed ones, oldest first. SKIP LOCKED keeps two workers off the same job."""
    row = conn.execute(
        """
        WITH c AS (
          SELECT id FROM jobs
           WHERE status = 'queued' AND run_after <= now()
             AND (target_worker_id IS NULL OR target_worker_id = %(wid)s)
           ORDER BY (target_worker_id IS NOT DISTINCT FROM %(wid)s) DESC, created_at
           LIMIT 1 FOR UPDATE SKIP LOCKED)
        UPDATE jobs j SET status = 'leased', lease_worker_id = %(wid)s, last_worker_id = %(wid)s,
               lease_token = gen_random_uuid(),
               lease_expires_at = now() + make_interval(secs => %(lease)s),
               started_at = COALESCE(started_at, now()),
               preempt_requested = false, updated_at = now()
          FROM c WHERE j.id = c.id RETURNING j.*
        """,
        {"wid": worker_id, "lease": lease},
    ).fetchone()
    if row is not None:
        add_job_event(conn, row["id"], "claimed", worker_id)
    return row


def renew(
    conn: psycopg.Connection,
    worker_id: str,
    job_id: Any,
    lease_token: Any,
    lease: int,
    progress: float | None = None,
    checkpoint: dict[str, Any] | None = None,
    detail: str | None = None,
    live: bool = False,
) -> bool:
    """Extend one lease; False when the (id, token, worker, status) do not match.
    `live` stores a null progress (a paper trade job has no end: the card says Live)."""
    jid, tok = as_uuid(job_id), as_uuid(lease_token)
    if jid is None or tok is None:
        return False
    row = conn.execute(
        """
        UPDATE jobs SET lease_expires_at = now() + make_interval(secs => %(lease)s),
               progress = CASE WHEN %(live)s THEN NULL ELSE COALESCE(%(progress)s, progress) END,
               checkpoint = COALESCE(%(checkpoint)s, checkpoint),
               detail = COALESCE(%(detail)s, detail),
               updated_at = now()
         WHERE id = %(id)s AND lease_token = %(tok)s AND lease_worker_id = %(wid)s
           AND status IN ('leased', 'cancel_requested')
         RETURNING id
        """,
        {
            "lease": lease,
            "progress": progress,
            "live": live,
            "detail": detail,
            "checkpoint": Jsonb(checkpoint) if checkpoint is not None else None,
            "id": jid,
            "tok": tok,
            "wid": worker_id,
        },
    ).fetchone()
    return row is not None


def _worker_name(conn: psycopg.Connection, worker_id: str | None) -> str:
    row = conn.execute("SELECT name FROM workers WHERE id = %s", (worker_id,)).fetchone() if worker_id else None
    return row["name"] if row else "the worker"


def release(
    conn: psycopg.Connection,
    job_id: Any,
    lease_token: Any,
    progress: float | None = None,
    checkpoint: dict[str, Any] | None = None,
    worker_id: str | None = None,
    reason: str | None = None,
) -> str | None:
    """A worker hands a leased job back. Cancelled when cancel was requested (or the
    reason is `cancel`), else failed with a plain reason the dashboard shows.

    Returns the new status, or None when the token (or, when given, the worker) did
    not match.
    """
    jid, tok = as_uuid(job_id), as_uuid(lease_token)
    if jid is None or tok is None:
        return None
    reason = reason if reason in RELEASE_REASONS else None
    job = conn.execute(
        """
        SELECT id, status, lease_worker_id FROM jobs
         WHERE id = %(id)s AND lease_token = %(tok)s
           AND (%(wid)s::text IS NULL OR lease_worker_id = %(wid)s)
           AND status IN ('leased', 'cancel_requested')
         FOR UPDATE
        """,
        {"id": jid, "tok": tok, "wid": worker_id},
    ).fetchone()
    if job is None:
        return None
    cancelled = job["status"] == "cancel_requested" or reason == "cancel"
    if reason == "update" and not cancelled:
        # An endless job (paper trading, model search) handed back so its worker can
        # update itself: queued again for the same worker, which takes it back from
        # its checkpoint a few seconds later.
        conn.execute(
            """
            UPDATE jobs SET status = 'queued', target_worker_id = lease_worker_id, target_auto = false,
                   progress = COALESCE(%s, progress), checkpoint = COALESCE(%s, checkpoint),
                   lease_worker_id = NULL, lease_token = NULL, lease_expires_at = NULL,
                   preempt_requested = false, updated_at = now()
             WHERE id = %s
            """,
            (progress, Jsonb(checkpoint) if checkpoint is not None else None, jid),
        )
        add_job_event(conn, jid, "released", worker_id, {"status": "queued", "reason": "update"})
        return "queued"
    status = "cancelled" if cancelled else "failed"
    error = None if cancelled else RELEASE_ERRORS[reason].format(worker=_worker_name(conn, job["lease_worker_id"]))
    conn.execute(
        """
        UPDATE jobs SET status = %(status)s, error = COALESCE(%(error)s, error), finished_at = now(),
               progress = COALESCE(%(progress)s, progress),
               checkpoint = COALESCE(%(checkpoint)s, checkpoint),
               lease_worker_id = NULL, lease_token = NULL, lease_expires_at = NULL,
               preempt_requested = false, updated_at = now()
         WHERE id = %(id)s
        """,
        {
            "status": status,
            "error": error,
            "progress": progress,
            "checkpoint": Jsonb(checkpoint) if checkpoint is not None else None,
            "id": jid,
        },
    )
    detail: dict[str, Any] = {"status": status}
    if reason is not None:
        detail["reason"] = reason
    add_job_event(conn, jid, "released", worker_id, detail)
    return status


def checkpoint(
    conn: psycopg.Connection,
    job_id: Any,
    lease_token: Any,
    checkpoint_data: dict[str, Any] | None,
    progress: float | None,
    do_release: bool = False,
    worker_id: str | None = None,
    reason: str | None = None,
) -> str:
    """POST /checkpoint: store progress under the fence; optionally release.

    `worker_id` (the caller's identity) must own the lease when given; `reason`
    is passed to release() when do_release is set.
    """
    job = get_job(conn, job_id, for_update=True)
    tok = _fence(job, lease_token, worker_id)
    if do_release:
        status = release(conn, job_id, tok, progress, checkpoint_data, job["lease_worker_id"], reason)
        return status or job["status"]
    conn.execute(
        """
        UPDATE jobs SET progress = COALESCE(%s, progress),
               checkpoint = COALESCE(%s, checkpoint), updated_at = now()
         WHERE id = %s
        """,
        (progress, Jsonb(checkpoint_data) if checkpoint_data is not None else None, job["id"]),
    )
    return job["status"]


def complete(
    conn: psycopg.Connection, job_id: Any, lease_token: Any, result: Any, worker_id: str | None = None
) -> dict[str, Any]:
    """POST /complete: succeed the job. Idempotent for a repeat with the same token."""
    job = get_job(conn, job_id, for_update=True)
    tok = as_uuid(lease_token)
    if tok is None or job["lease_token"] != tok:
        raise Conflict("lease token mismatch")
    if job["status"] == "succeeded":
        return job
    if job["status"] not in ACTIVE:
        raise Conflict(f"job is {job['status']}")
    _fence(job, tok, worker_id)
    row = conn.execute(
        """
        UPDATE jobs SET status = 'succeeded', result = %s, progress = 1, finished_at = now(),
               lease_worker_id = NULL, lease_expires_at = NULL,
               preempt_requested = false, updated_at = now()
         WHERE id = %s RETURNING *
        """,
        (Jsonb(result) if result is not None else None, job["id"]),
    ).fetchone()
    add_job_event(conn, job["id"], "succeeded", job["lease_worker_id"])
    return row


def fail(
    conn: psycopg.Connection, job_id: Any, lease_token: Any, error: str, worker_id: str | None = None
) -> dict[str, Any]:
    """POST /fail: terminal failure, never retried."""
    job = get_job(conn, job_id, for_update=True)
    tok = as_uuid(lease_token)
    if tok is None or job["lease_token"] != tok:
        raise Conflict("lease token mismatch")
    if job["status"] == "failed":
        return job
    if job["status"] not in ACTIVE:
        raise Conflict(f"job is {job['status']}")
    _fence(job, tok, worker_id)
    row = conn.execute(
        """
        UPDATE jobs SET status = 'failed', error = %s, finished_at = now(),
               lease_worker_id = NULL, lease_expires_at = NULL,
               preempt_requested = false, updated_at = now()
         WHERE id = %s RETURNING *
        """,
        (error, job["id"]),
    ).fetchone()
    add_job_event(conn, job["id"], "failed", job["lease_worker_id"], {"error": error})
    return row
