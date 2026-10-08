"""Worker routes: register and heartbeat."""
from __future__ import annotations

from typing import Any

import psycopg
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from coordinator import auth, queue
from coordinator.api.deps import DB, bearer, code_version, remote_ip, with_code_version
from coordinator.api.limits import MAX_JOB_ENTRIES, small_payload
from coordinator.api.serialize import jsonable

router = APIRouter(prefix="/api/v1/workers", tags=["workers"])

NAME_RE = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
SHORT_RE = r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$"


class RegisterBody(BaseModel):
    """Either enroll_token or worker_id + worker_token."""

    model_config = ConfigDict(extra="ignore")
    enroll_token: str | None = Field(default=None, max_length=256)
    worker_id: str | None = Field(default=None, max_length=64)
    worker_token: str | None = Field(default=None, max_length=256)
    hostname: str | None = Field(default=None, pattern=NAME_RE)
    name: str | None = Field(default=None, pattern=NAME_RE)
    python_version: str | None = Field(default=None, pattern=SHORT_RE)
    code_version: str | None = Field(default=None, pattern=SHORT_RE)
    boot_id: str | None = Field(default=None, pattern=SHORT_RE)


class JobEntry(BaseModel):
    """A running or released job as the worker reports it."""

    model_config = ConfigDict(extra="ignore")
    id: str = Field(max_length=64)
    lease_token: str = Field(max_length=64)
    progress: float | None = Field(default=None, allow_inf_nan=False)
    checkpoint: dict[str, Any] | None = None
    reason: str | None = Field(default=None, max_length=32)
    detail: str | None = Field(default=None, max_length=200)

    @field_validator("checkpoint")
    @classmethod
    def _small_checkpoint(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        return small_payload(value, "checkpoint")


class HeartbeatBody(BaseModel):
    """Heartbeat payload."""

    model_config = ConfigDict(extra="ignore")
    cpu_pct: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    ram_pct: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    ram_used_mb: int | None = None
    ram_total_mb: int | None = None
    jobs: list[JobEntry] = Field(default_factory=list, max_length=MAX_JOB_ENTRIES)
    released: list[JobEntry] = Field(default_factory=list, max_length=MAX_JOB_ENTRIES)
    want_job: bool = False
    code_version: str | None = Field(default=None, pattern=SHORT_RE)
    skew_ms: int | None = None
    temp_c: float | None = Field(default=None, ge=-50, le=150, allow_inf_nan=False)


@router.post("/register")
def register(
    body: RegisterBody,
    request: Request,
    conn: psycopg.Connection = DB,
    version: str = Depends(code_version),
) -> dict[str, Any]:
    """First registration (enroll token) or re-registration (token rotation)."""
    reply = queue.register(conn, body.model_dump(), remote_ip(request))
    return jsonable(with_code_version(reply, version))


@router.post("/{worker_id}/heartbeat")
def heartbeat(
    worker_id: str,
    body: HeartbeatBody,
    token: str = Depends(bearer),
    conn: psycopg.Connection = DB,
    version: str = Depends(code_version),
) -> dict[str, Any]:
    """Heartbeat: stats, renew, release, stop requests, claim.

    The token is checked again inside process_heartbeat under the worker row lock.
    """
    auth.verify_worker_token(conn, worker_id, token)
    reply = queue.process_heartbeat(conn, worker_id, body.model_dump(), auth.hash_token(token))
    return jsonable(with_code_version(reply, version))
