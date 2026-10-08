"""Lease recovery: orphaned leases (re-offer), held jobs on register (re-lease), the reaper.

Copied from v1. Changed: the reaper fails a job whose worker went silent ("Worker w3
went offline") instead of requeueing it; the owner runs it again from the dashboard.
A worker that crashes and comes back before its lease runs out still gets its job
back through held_jobs, exactly as in v1."""
from __future__ import annotations

from typing import Any

import psycopg

from coordinator.events import add_job_event
from coordinator.leases import as_uuid


def orphan_jobs(conn: psycopg.Connection, worker_id: str, reported_ids: list[str], lease: int) -> list[dict[str, Any]]:
    """Leases this worker holds but did not mention in its heartbeat: the claim reply
    was lost (or a stale heartbeat claimed on its behalf).

    `leased` orphans are renewed and returned so the worker is handed the same lease
    again instead of a second job. A `cancel_requested` orphan is cancelled outright:
    the worker is not running it, so there is nothing to stop.
    """
    known = [jid for jid in (as_uuid(v) for v in reported_ids) if jid is not None]
    cancelled = conn.execute(
        """
        UPDATE jobs SET status = 'cancelled', finished_at = now(),
               lease_worker_id = NULL, lease_token = NULL, lease_expires_at = NULL,
               preempt_requested = false, updated_at = now()
         WHERE lease_worker_id = %(wid)s AND status = 'cancel_requested'
           AND NOT (id = ANY(%(known)s::uuid[]))
         RETURNING id
        """,
        {"wid": worker_id, "known": known},
    ).fetchall()
    for row in cancelled:
        add_job_event(conn, row["id"], "cancelled", worker_id, {"reason": "orphaned lease"})
    rows = conn.execute(
        """
        UPDATE jobs SET lease_expires_at = now() + make_interval(secs => %(lease)s), updated_at = now()
         WHERE lease_worker_id = %(wid)s AND status = 'leased'
           AND NOT (id = ANY(%(known)s::uuid[]))
         RETURNING *
        """,
        {"lease": lease, "wid": worker_id, "known": known},
    ).fetchall()
    for row in rows:
        add_job_event(conn, row["id"], "re-offered", worker_id)
    return rows


def _last_event(conn: psycopg.Connection, job_id: Any) -> dict[str, Any] | None:
    """The newest job_events row for a job."""
    return conn.execute(
        "SELECT event, worker_id FROM job_events WHERE job_id = %s ORDER BY id DESC LIMIT 1", (job_id,)
    ).fetchone()


def held_jobs(conn: psycopg.Connection, worker_id: str, lease: int) -> list[dict[str, Any]]:
    """Re-lease the worker's live leases with fresh tokens (register).

    A `re-leased` event is written once per run of registers: when the job's newest
    event is already `re-leased` by this worker (a register retry loop) no row is added,
    so a looping agent cannot grow job_events without bound.
    """
    rows = conn.execute(
        """
        UPDATE jobs SET lease_token = gen_random_uuid(),
               lease_expires_at = now() + make_interval(secs => %s), updated_at = now()
         WHERE lease_worker_id = %s AND status IN ('leased', 'cancel_requested')
           AND lease_expires_at > now()
         RETURNING *
        """,
        (lease, worker_id),
    ).fetchall()
    for row in rows:
        last = _last_event(conn, row["id"])
        if last is None or last["event"] != "re-leased" or last["worker_id"] != worker_id:
            add_job_event(conn, row["id"], "re-leased", worker_id)
    return rows


def reap(conn: psycopg.Connection) -> list[dict[str, Any]]:
    """Expire overdue leases: a cancel_requested job is cancelled, every other job is
    failed with "Worker <name> went offline" (it can be run again from the dashboard)."""
    rows = conn.execute(
        """
        WITH e AS (
          SELECT j.id, j.lease_worker_id AS old_worker, COALESCE(w.name, 'the worker') AS name
            FROM jobs j LEFT JOIN workers w ON w.id = j.lease_worker_id
           WHERE j.status IN ('leased', 'cancel_requested') AND j.lease_expires_at < now()
           FOR UPDATE OF j SKIP LOCKED)
        UPDATE jobs j SET
               status = CASE WHEN j.status = 'cancel_requested' THEN 'cancelled' ELSE 'failed' END,
               error = CASE WHEN j.status = 'cancel_requested' THEN j.error
                            ELSE 'Worker ' || e.name || ' went offline' END,
               finished_at = now(),
               lease_worker_id = NULL, lease_token = NULL, lease_expires_at = NULL,
               preempt_requested = false, updated_at = now()
          FROM e WHERE j.id = e.id
          RETURNING j.id, j.status, e.old_worker
        """
    ).fetchall()
    for row in rows:
        add_job_event(conn, row["id"], "lease_expired", row["old_worker"], {"status": row["status"]})
    return rows
