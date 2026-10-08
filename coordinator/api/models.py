"""Model routes: workers report backtest results; the owner lists models."""
from __future__ import annotations

from typing import Any

import psycopg
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from coordinator import auth, futures_models, models, search
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


class FuturesFoundBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    job_id: str | None = Field(default=None, max_length=64)
    module: str = Field(max_length=64)
    params: dict[str, Any]
    train_score: float = Field(allow_inf_nan=False)
    seed: str | None = Field(default=None, max_length=200)
    parent: str | None = Field(default=None, max_length=128)
    metrics: dict[str, Any]

    @field_validator("metrics")
    @classmethod
    def _small(cls, value: dict[str, Any]) -> dict[str, Any]:
        return small_payload(value, "metrics")


@worker_router.post("/futures")
def report_found_futures(body: FuturesFoundBody, token: str = Depends(bearer), conn: psycopg.Connection = DB) -> dict[str, Any]:
    """A futures model a model search job found (kept, replacing one, or dropped)."""
    worker = auth.worker_for_token(conn, token)
    return futures_models.store_found(conn, worker["id"], body.model_dump())


@worker_router.get("/kept")
def kept_models(market: str = "futures", token: str = Depends(bearer), conn: psycopg.Connection = DB) -> list[dict[str, Any]]:
    """The futures models search has kept: next round's small changes start from them."""
    auth.worker_for_token(conn, token)
    if market != "futures":
        raise BadRequest("only futures searches ask for their kept models")
    return jsonable(futures_models.kept_models(conn))


class FinalCheckBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    job_id: str = Field(max_length=64)
    feed: str = Field(default="", max_length=32)
    result: dict[str, Any]

    @field_validator("result")
    @classmethod
    def _small(cls, value: dict[str, Any]) -> dict[str, Any]:
        return small_payload(value, "result")


@worker_router.post("/{model_id}/final-check")
def report_final_check(model_id: str, body: FinalCheckBody, token: str = Depends(bearer),
                       conn: psycopg.Connection = DB) -> dict[str, Any]:
    """A Final check's lockbox result, kept forever."""
    worker = auth.worker_for_token(conn, token)
    return futures_models.store_final_check(conn, model_id, worker["id"], body.model_dump())


@owner_router.post("/{model_id}/final-check", status_code=201)
def start_final_check(model_id: str, request: Request, conn: psycopg.Connection = DB) -> dict[str, Any]:
    """Open the lockbox for one futures model, once."""
    return jsonable(futures_models.start_final_check(conn, model_id, request.app.state.topstep))


class TriesBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    job_id: str | None = Field(default=None, max_length=64)
    tries: dict[str, dict[str, float]]


worker_search_router = APIRouter(prefix="/api/v1/search", tags=["search"])


@worker_search_router.post("/tries")
def report_tries(body: TriesBody, token: str = Depends(bearer), conn: psycopg.Connection = DB) -> dict[str, Any]:
    """How many futures settings a search round tried, per model file."""
    auth.worker_for_token(conn, token)
    futures_models.add_tries(conn, body.tries)
    return {"ok": True}


class SearchBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    markets: list[str] | None = None
    target: str = Field(default="all_idle", max_length=64)


search_router = APIRouter(prefix="/api/search", tags=["search"], dependencies=[Depends(require_owner)])


@search_router.post("/start", status_code=201)
def start_search(request: Request, body: SearchBody | None = None, conn: psycopg.Connection = DB) -> dict[str, Any]:
    body = body or SearchBody()
    if body.markets and body.markets != ["futures"] and not set(body.markets) <= {"stocks", "crypto"}:
        raise BadRequest("markets must be stocks and/or crypto, or futures on its own")
    return jsonable(search.start(conn, request.app.state.limits, body.markets, body.target, request.app.state.topstep))


@search_router.post("/stop")
def stop_search(conn: psycopg.Connection = DB) -> dict[str, Any]:
    return search.stop(conn)
