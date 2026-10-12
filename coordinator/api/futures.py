"""Futures trading routes: workers send a futures model's decisions; the owner starts and
stops a model on Alpaca paper or Topstep, confirms Topstep, and resumes it after its
loss-limit stop."""
from __future__ import annotations

from typing import Any, Literal

import psycopg
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from coordinator import auth, futures_trading, futures_view, safety
from coordinator.api.deps import DB, bearer, require_owner
from coordinator.api.serialize import jsonable
from coordinator.errors import BadRequest, Conflict
from coordinator.settings import put_setting
from coordinator.topstep_broker import CONFIRM_PHRASE, fingerprint

worker_router = APIRouter(prefix="/api/v1/futures", tags=["futures"])
owner_router = APIRouter(prefix="/api", tags=["futures"], dependencies=[Depends(require_owner)])


class SignalBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    job_id: str = Field(max_length=64)
    minute_t: int
    contracts: dict[str, int]
    reason: str | None = Field(default=None, max_length=500)


@worker_router.post("/signal")
def signal(body: SignalBody, token: str = Depends(bearer), conn: psycopg.Connection = DB) -> dict[str, Any]:
    """The contracts a futures model wants from the next minute on."""
    worker = auth.worker_for_token(conn, token)
    return futures_trading.receive_signal(conn, worker["id"], body.model_dump())


class VenueBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    venue: Literal["alpaca_paper", "topstep"] = "alpaca_paper"
    target: str = Field(default="auto", max_length=64)


@owner_router.post("/models/{model_id}/futures/start", status_code=201)
def start(model_id: str, request: Request, body: VenueBody | None = None, conn: psycopg.Connection = DB) -> dict[str, Any]:
    body = body or VenueBody()
    if body.target == "all_idle":
        raise BadRequest("A model trades on one worker: pick Auto or one worker")
    rules = request.app.state.topstep
    ready = False
    if body.venue == "topstep":
        ready = futures_view.model_verdict(conn, model_id, rules)["ready"]
    return jsonable(futures_trading.start(conn, model_id, body.venue, request.app.state.venues, rules,
                                         request.app.state.limits, body.target, ready))


@owner_router.post("/models/{model_id}/futures/stop")
def stop(model_id: str, body: VenueBody | None = None, conn: psycopg.Connection = DB) -> dict[str, Any]:
    body = body or VenueBody()
    return futures_trading.stop(conn, model_id, body.venue)


@owner_router.get("/models/{model_id}/futures/orders")
def orders(model_id: str, conn: psycopg.Connection = DB) -> list[dict[str, Any]]:
    """The model's last 100 futures orders, newest first."""
    return jsonable(conn.execute("SELECT * FROM futures_orders WHERE model_id = %s ORDER BY created_at DESC LIMIT 100",
                                 (model_id,)).fetchall())


class ConfirmBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    confirm: str = Field(default="", max_length=64)


@owner_router.post("/topstep/confirm")
def confirm(body: ConfirmBody, request: Request, conn: psycopg.Connection = DB,
            actor: str = Depends(require_owner)) -> dict[str, Any]:
    """The owner's confirmation for trading on Topstep, for exactly the keys in .env;
    it takes effect when the coordinator restarts."""
    if body.confirm != CONFIRM_PHRASE:
        raise BadRequest(f"Type {CONFIRM_PHRASE} exactly to confirm")
    config = request.app.state.config
    if not (config.topstepx_username and config.topstepx_api_key and config.topstepx_account):
        raise Conflict("Put TOPSTEPX_USERNAME, TOPSTEPX_API_KEY and TOPSTEPX_ACCOUNT in .env first: "
                       "the confirmation is for those")
    put_setting(conn, "topstep_confirmed", {"key": fingerprint(config)}, actor, "topstep_confirmed",
                confirmation_text=body.confirm)
    return {"message": "Confirmed. Restart the coordinator to connect to Topstep."}


@owner_router.post("/topstep/withdraw")
def withdraw(conn: psycopg.Connection = DB, actor: str = Depends(require_owner)) -> dict[str, Any]:
    put_setting(conn, "topstep_confirmed", False, actor, "topstep_withdrawn")
    return {"message": "Topstep confirmation withdrawn. It takes effect when the coordinator restarts; stop the "
                       "models on Topstep now if you want them out at once."}


@owner_router.post("/topstep/resume")
def resume(conn: psycopg.Connection = DB, actor: str = Depends(require_owner)) -> dict[str, Any]:
    """Resume Topstep trading after its loss-limit stop (only the owner can)."""
    if not safety.resume_topstep(conn, actor):
        raise Conflict("Topstep trading is not paused")
    return {"message": "Topstep trading resumed"}
