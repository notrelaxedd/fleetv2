"""Model routes: workers report backtest results; the owner lists models."""
from __future__ import annotations

from typing import Any

import psycopg
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field, field_validator

from coordinator import auth, models
from coordinator.api.deps import DB, bearer, require_owner
from coordinator.api.limits import small_payload
from coordinator.api.serialize import jsonable

worker_router = APIRouter(prefix="/api/v1/models", tags=["models"])
owner_router = APIRouter(prefix="/api/models", tags=["models"], dependencies=[Depends(require_owner)])


class BacktestBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    job_id: str | None = Field(default=None, max_length=64)
    backtest_metrics: dict[str, Any]

    @field_validator("backtest_metrics")
    @classmethod
    def _small(cls, value: dict[str, Any]) -> dict[str, Any]:
        return small_payload(value, "backtest_metrics")


@worker_router.post("/{model_id}/backtest")
def report_backtest(model_id: str, body: BacktestBody, token: str = Depends(bearer),
                    conn: psycopg.Connection = DB) -> dict[str, Any]:
    """A worker's finished backtest of one model (sent before it completes the job)."""
    auth.worker_for_token(conn, token)
    return jsonable(models.store_backtest(conn, model_id, body.backtest_metrics, body.job_id))


@owner_router.get("")
def list_models(conn: psycopg.Connection = DB) -> list[dict[str, Any]]:
    return jsonable(models.list_models(conn, include_retired=True))


@owner_router.get("/{model_id}")
def get_model(model_id: str, conn: psycopg.Connection = DB) -> dict[str, Any]:
    return jsonable(models.get_model(conn, model_id))
