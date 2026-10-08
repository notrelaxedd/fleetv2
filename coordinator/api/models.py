"""Model routes: workers report backtest results; the owner lists models."""
from __future__ import annotations

from typing import Any

import psycopg
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from coordinator import auth, models, search
from coordinator.errors import BadRequest
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


class FoundBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    job_id: str | None = Field(default=None, max_length=64)
    module: str = Field(max_length=64)
    params: dict[str, Any]
    train_score: float = Field(allow_inf_nan=False)
    seed: str | None = Field(default=None, max_length=64)
    metrics: dict[str, Any]

    @field_validator("metrics")
    @classmethod
    def _small(cls, value: dict[str, Any]) -> dict[str, Any]:
        return small_payload(value, "metrics")


@worker_router.post("")
def report_found(body: FoundBody, token: str = Depends(bearer), conn: psycopg.Connection = DB) -> dict[str, Any]:
    """A model found by a model search job."""
    worker = auth.worker_for_token(conn, token)
    return search.store_found(conn, worker["id"], body.model_dump())


class SearchBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    markets: list[str] | None = None
    target: str = Field(default="all_idle", max_length=64)


search_router = APIRouter(prefix="/api/search", tags=["search"], dependencies=[Depends(require_owner)])


@search_router.post("/start", status_code=201)
def start_search(request: Request, body: SearchBody | None = None, conn: psycopg.Connection = DB) -> dict[str, Any]:
    body = body or SearchBody()
    if body.markets and not set(body.markets) <= {"stocks", "crypto"}:
        raise BadRequest("markets must be stocks and/or crypto")
    return jsonable(search.start(conn, request.app.state.limits, body.markets, body.target))


@search_router.post("/stop")
def stop_search(conn: psycopg.Connection = DB) -> dict[str, Any]:
    return search.stop(conn)
