"""Trading routes: workers send model decisions; the owner starts and stops paper
trading and confirms (or withdraws) real-money trading."""
from __future__ import annotations

from typing import Any

import psycopg
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from coordinator import auth, queue, trading
from coordinator.api.deps import DB, bearer, require_owner
from coordinator.api.serialize import jsonable
from coordinator.broker import LIVE
from coordinator.errors import BadRequest, Conflict
from coordinator.settings import put_setting

worker_router = APIRouter(prefix="/api/v1/paper", tags=["trading"])
owner_router = APIRouter(prefix="/api", tags=["trading"], dependencies=[Depends(require_owner)])
LIVE_PHRASE = "TRADE REAL MONEY"


class SignalBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    job_id: str = Field(max_length=64)
    bar_t: int
    targets: dict[str, float]
    reason: str | None = Field(default=None, max_length=500)


class StartBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    target: str = Field(default="auto", max_length=64)
    confirm: str | None = Field(default=None, max_length=64)


class LiveBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    confirm: str = Field(default="", max_length=64)


@worker_router.post("/signal")
def signal(body: SignalBody, token: str = Depends(bearer), conn: psycopg.Connection = DB) -> dict[str, Any]:
    """A worker's latest decision for the model it paper trades."""
    worker = auth.worker_for_token(conn, token)
    return trading.receive_signal(conn, worker["id"], body.model_dump())


@owner_router.post("/models/{model_id}/paper/start", status_code=201)
def start_paper(model_id: str, request: Request, body: StartBody | None = None,
                conn: psycopg.Connection = DB) -> dict[str, Any]:
    """Open the model's book and give a worker the job of computing its decisions."""
    return start_paper_trading(conn, request, model_id, body or StartBody())


def start_paper_trading(conn: psycopg.Connection, request: Request, model_id: str, body: StartBody) -> dict[str, Any]:
    """Shared by the Models screen button and the Assign a job panel."""
    if body.target == "all_idle":
        raise BadRequest("A model paper trades on one worker: pick Auto or one worker")
    status = request.app.state.broker_status
    mode = status.broker.mode
    if mode == LIVE and body.confirm != LIVE_PHRASE:
        raise BadRequest(f"The coordinator is in live mode: type {LIVE_PHRASE} to trade this model with real money")
    book = trading.start(conn, model_id, mode, request.app.state.limits)
    model = conn.execute("SELECT * FROM models WHERE id = %s", (model_id,)).fetchone()
    result = queue.create_job(conn, "paper_trade", {"model_id": model_id, "module": model["module"],
                                                    "params": model["params"], "market": model["market"]},
                              body.target, model_id)
    trading.attach_job(conn, book["id"], result.jobs[0]["id"])
    money = f"${book['starting_balance']:,.0f}"
    where = "a worker" if result.waiting else "its worker"
    return {"jobs": result.jobs, "waiting": result.waiting,
            "message": f"{model['name']} is {'paper' if mode != LIVE else 'live'} trading with {money}; "
                       f"{where} sends its decisions to the coordinator, which places the orders"}


@owner_router.post("/models/{model_id}/paper/stop")
def stop_paper(model_id: str, conn: psycopg.Connection = DB) -> dict[str, Any]:
    book = trading.stop(conn, model_id)
    held = conn.execute("SELECT count(*) AS n FROM positions WHERE book_id = %s", (book["id"],)).fetchone()["n"]
    tail = f"; its {held} position(s) will be sold" if held else ""
    return {"message": f"Paper trading stopped{tail}"}


@owner_router.get("/models/{model_id}/orders")
def model_orders(model_id: str, conn: psycopg.Connection = DB) -> list[dict[str, Any]]:
    """The model's last 100 orders, newest first (every order with worker, time and reason)."""
    rows = conn.execute(
        "SELECT o.*, w.name AS worker_name FROM orders o LEFT JOIN workers w ON w.id = o.worker_id"
        " WHERE o.model_id = %s ORDER BY o.created_at DESC LIMIT 100",
        (model_id,),
    ).fetchall()
    return jsonable(rows)


@owner_router.get("/live")
def live_state(request: Request, conn: psycopg.Connection = DB) -> dict[str, Any]:
    config = request.app.state.config
    confirmed = conn.execute("SELECT value FROM settings WHERE key = 'live_confirmed'").fetchone()["value"] is True
    return {
        "mode": request.app.state.broker_status.broker.mode,
        "env_allows_live": config.alpaca_live_allowed,
        "live_keys_present": bool(config.alpaca_live_key_id and config.alpaca_live_secret),
        "confirmed": confirmed,
        "phrase": LIVE_PHRASE,
        "note": "The mode only changes when the coordinator restarts, never on its own.",
    }


@owner_router.post("/live/confirm")
def live_confirm(body: LiveBody, request: Request, conn: psycopg.Connection = DB,
                 actor: str = Depends(require_owner)) -> dict[str, Any]:
    """The owner's confirmation for real money. Takes effect only if ALPACA_LIVE=true and
    live keys are in .env, and only after the coordinator restarts."""
    if body.confirm != LIVE_PHRASE:
        raise BadRequest(f"Type {LIVE_PHRASE} exactly to confirm")
    if not request.app.state.config.alpaca_live_allowed:
        raise Conflict("ALPACA_LIVE=true is not set in .env on box1, so live trading stays off")
    put_setting(conn, "live_confirmed", True, actor, "live_confirmed", confirmation_text=body.confirm)
    return {"message": "Confirmed. Restart the coordinator to switch to live trading."}


@owner_router.post("/live/withdraw")
def live_withdraw(conn: psycopg.Connection = DB, actor: str = Depends(require_owner)) -> dict[str, Any]:
    put_setting(conn, "live_confirmed", False, actor, "live_withdrawn")
    return {"message": "Live trading confirmation withdrawn. It takes effect when the coordinator restarts; "
                       "pause all trading now if you want it to stop at once."}
