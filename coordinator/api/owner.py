"""Owner JSON API (the dashboard's buttons and the CLI call these): fleet data, jobs,
the pause switch, enroll tokens. Every route needs the owner login (coordinator.auth)."""
from __future__ import annotations

from typing import Any

import psycopg
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from coordinator import auth, fleet_view, models, queue
from coordinator.api.deps import DB, get_config, require_owner
from coordinator.api.serialize import jsonable
from coordinator.config import Config
from coordinator.errors import BadRequest
from coordinator.safety import pause_trading, resume_trading

router = APIRouter(prefix="/api", tags=["owner"], dependencies=[Depends(require_owner)])
health_router = APIRouter(tags=["health"])


class JobBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    kind: str = Field(max_length=32)
    target: str = Field(default="auto", max_length=64)
    model_id: str | None = Field(default=None, max_length=128)
    params: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = Field(default=None, max_length=128)


class EnabledBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool


def _job_line(job: dict[str, Any], worker_names: dict[str, str]) -> str:
    label = fleet_view.JOB_LABELS.get(job["kind"], job["kind"])
    target = job.get("target_worker_id")
    if target:
        return f"{label} sent to {worker_names.get(target, target)}"
    return f"{label} queued: it starts when a worker is free"


@router.get("/fleet")
def fleet(request: Request, conn: psycopg.Connection = DB) -> dict[str, Any]:
    """Everything on the Fleet screen."""
    return jsonable(fleet_view.fleet_page(conn, request.app.state.broker_status, request.app.state.limits))


@router.post("/jobs", status_code=201)
def create_job(body: JobBody, request: Request, conn: psycopg.Connection = DB) -> dict[str, Any]:
    """Assign a job. target: "auto", "all_idle" or a worker id. The confirmation line
    the dashboard shows comes back in `message`."""
    if body.kind != "sleep" and body.kind not in fleet_view.AVAILABLE_KINDS:
        raise BadRequest(f"{fleet_view.JOB_LABELS.get(body.kind, body.kind)} jobs are not available in this build yet")
    if body.kind in fleet_view.NEEDS_MODEL and not body.model_id:
        raise BadRequest("Pick a model for this job")
    if body.kind not in fleet_view.NEEDS_MODEL:
        body.model_id = None
    params = models.job_params(conn, body.kind, body.model_id, body.params, request.app.state.limits)
    result = queue.create_job(conn, body.kind, params, body.target, body.model_id, body.idempotency_key)
    names = {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM workers").fetchall()}
    lines = [_job_line(j, names) for j in result.jobs]
    message = lines[0] if len(lines) == 1 else f"{len(lines)} jobs sent, one to each idle worker"
    return jsonable({"jobs": result.jobs, "waiting": result.waiting, "message": message})


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, conn: psycopg.Connection = DB) -> dict[str, Any]:
    return jsonable(queue.cancel_job(conn, job_id))


@router.post("/jobs/{job_id}/run-again", status_code=201)
def run_again(job_id: str, conn: psycopg.Connection = DB) -> dict[str, Any]:
    """A failed or cancelled job, queued again for the least busy worker."""
    result = queue.run_again(conn, job_id)
    names = {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM workers").fetchall()}
    return jsonable({"jobs": result.jobs, "message": _job_line(result.jobs[0], names)})


@router.get("/jobs/{job_id}")
def get_job(job_id: str, conn: psycopg.Connection = DB) -> dict[str, Any]:
    job = queue.get_job(conn, job_id)
    events = conn.execute("SELECT * FROM job_events WHERE job_id = %s ORDER BY id DESC LIMIT 50", (job["id"],)).fetchall()
    return jsonable({"job": job, "events": events})


@router.post("/trading/pause")
def pause(conn: psycopg.Connection = DB, actor: str = Depends(require_owner)) -> dict[str, Any]:
    """Pause all trading at once. Backtests keep running."""
    pause_trading(conn, actor, "Paused by you")
    return {"paused": True}


@router.post("/trading/resume")
def resume(conn: psycopg.Connection = DB, actor: str = Depends(require_owner)) -> dict[str, Any]:
    resume_trading(conn, actor)
    return {"paused": False}


@router.post("/workers/{worker_id}/enabled")
def set_enabled(worker_id: str, body: EnabledBody, conn: psycopg.Connection = DB) -> dict[str, Any]:
    """A disabled worker takes no new jobs (use it before switching a box back to v1)."""
    queue.get_worker(conn, worker_id)
    row = conn.execute("UPDATE workers SET enabled = %s WHERE id = %s RETURNING id, name, enabled",
                       (body.enabled, worker_id)).fetchone()
    return jsonable(row)


@router.post("/enroll-token")
def enroll_token(conn: psycopg.Connection = DB, config: Config = Depends(get_config)) -> dict[str, Any]:
    """A single-use token (1 hour) and the install command for a new worker."""
    token, expires_at = auth.create_enroll_token(conn)
    url = config.public_url
    return jsonable({
        "token": token,
        "expires_at": expires_at,
        "command": f"curl -fsSL {url}/install.sh | sudo bash -s -- {url} {token} --name w<N>",
    })


@health_router.get("/healthz")
def healthz(request: Request) -> dict[str, Any]:
    try:
        with request.app.state.pool.connection() as conn:
            conn.execute("SELECT 1")
        db_ok = True
    except Exception:  # noqa: BLE001
        db_ok = False
    return {"ok": db_ok, "db": db_ok}
