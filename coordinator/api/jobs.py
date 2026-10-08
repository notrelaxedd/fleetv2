"""Worker job routes: checkpoint, complete, fail (fenced by lease_token and the caller's worker id)."""
from __future__ import annotations

from typing import Any

import psycopg
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field, field_validator

from coordinator import auth, queue
from coordinator.api.deps import DB, bearer
from coordinator.api.limits import small_payload
from coordinator.api.serialize import jsonable

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


class CheckpointBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    lease_token: str = Field(max_length=64)
    checkpoint: dict[str, Any] | None = None
    progress: float | None = Field(default=None, allow_inf_nan=False)
    release: bool = False
    reason: str | None = Field(default=None, max_length=32)

    @field_validator("checkpoint")
    @classmethod
    def _small_checkpoint(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        return small_payload(value, "checkpoint")


class CompleteBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    lease_token: str = Field(max_length=64)
    result: Any = None

    @field_validator("result")
    @classmethod
    def _small_result(cls, value: Any) -> Any:
        return small_payload(value, "result")


class FailBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    lease_token: str = Field(max_length=64)
    error: str = Field(default="", max_length=16 * 1024)


@router.post("/{job_id}/checkpoint")
def checkpoint(
    job_id: str,
    body: CheckpointBody,
    token: str = Depends(bearer),
    conn: psycopg.Connection = DB,
) -> dict[str, Any]:
    """Store checkpoint/progress; release=true hands the job back."""
    worker = auth.worker_for_token(conn, token)
    status = queue.checkpoint(
        conn, job_id, body.lease_token, body.checkpoint, body.progress, body.release, worker["id"],
        body.reason,
    )
    return {"status": status}


@router.post("/{job_id}/complete")
def complete(
    job_id: str,
    body: CompleteBody,
    token: str = Depends(bearer),
    conn: psycopg.Connection = DB,
) -> dict[str, Any]:
    """Mark the job succeeded."""
    worker = auth.worker_for_token(conn, token)
    job = queue.complete(conn, job_id, body.lease_token, body.result, worker["id"])
    return {"status": job["status"], "id": str(job["id"])}


@router.post("/{job_id}/fail")
def fail(
    job_id: str,
    body: FailBody,
    token: str = Depends(bearer),
    conn: psycopg.Connection = DB,
) -> dict[str, Any]:
    """Mark the job failed (terminal)."""
    worker = auth.worker_for_token(conn, token)
    job = queue.fail(conn, job_id, body.lease_token, body.error, worker["id"])
    return jsonable({"status": job["status"], "id": str(job["id"])})
