"""Price data routes. Workers drive the refresh one symbol at a time and download the
cached bars; only the coordinator ever talks to Alpaca (coordinator.data)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import psycopg
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from coordinator import auth, data, futures_data
from coordinator.api.deps import DB, bearer, require_owner
from coordinator.api.serialize import jsonable
from coordinator.errors import BadRequest, Forbidden, NotFound
from fleet2 import universe

router = APIRouter(prefix="/api/v1/data", tags=["data"])
owner_router = APIRouter(prefix="/api/data", tags=["data"], dependencies=[Depends(require_owner)])

NO_DATA = "No price data yet: run a Data refresh job"


class RefreshStepBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    market: Literal["stocks", "crypto", "futures"]
    symbol: str = Field(max_length=32)
    after: datetime | None = None  # crypto: the previous step's cursor (skips windows with no bars)


@router.post("/refresh-step")
def refresh_step(
    body: RefreshStepBody,
    request: Request,
    token: str = Depends(bearer),
    conn: psycopg.Connection = DB,
) -> dict[str, Any]:
    """Download one symbol (stocks) or one window of it (crypto) into the cache."""
    auth.worker_for_token(conn, token)
    if body.market == "futures":
        if body.symbol not in universe.FUTURES["symbols"]:
            raise BadRequest(f"{body.symbol} is not a futures symbol")
        cap = futures_data.max_download_usd(request.app.state.config.topstep_path)
        return futures_data.refresh_step(conn, request.app.state.futures_source, body.symbol, cap, after=body.after)
    if body.symbol not in universe.MARKETS[body.market]["symbols"]:
        raise BadRequest(f"{body.symbol} is not in the {body.market} universe")
    return data.refresh_step(conn, request.app.state.bar_source, body.market, body.symbol, after=body.after)


def _etag_matches(header: str | None, etag: str) -> bool:
    if not header:
        return False
    if header.strip() == "*":
        return True
    return any(part.strip().removeprefix("W/") == etag for part in header.split(","))


@router.get("/bars")
def bars(
    request: Request,
    market: Literal["stocks", "crypto"] = Query(...),
    token: str = Depends(bearer),
    conn: psycopg.Connection = DB,
) -> Response:
    """One market's cached bars, gzip-compressed JSON (fleet2.sim.marketdata.from_payload)."""
    auth.worker_for_token(conn, token)
    try:
        body, etag = data.bars_payload(conn, market)
    except LookupError:
        raise NotFound(NO_DATA) from None
    headers = {"ETag": etag, "Cache-Control": "no-cache"}
    if _etag_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=304, headers=headers)
    return Response(content=body, media_type="application/gzip", headers=headers)


@router.get("/futures-bars")
def futures_bars(
    request: Request,
    through: Literal["train", "held_out", "lockbox"] = Query(...),
    job_id: str | None = Query(default=None, max_length=64),
    token: str = Depends(bearer),
    conn: psycopg.Connection = DB,
) -> Response:
    """Futures bars up to the end of one period, as a compressed numpy file
    (fleet2.sim.futures_data.from_npz). Training and held-out go to any worker; the
    lockbox only to the worker running that model's Final check (job_id)."""
    worker = auth.worker_for_token(conn, token)
    if through == "lockbox" and not futures_data.lockbox_allowed(conn, worker["id"], job_id):
        raise Forbidden("Lockbox prices are only for a running Final check")
    try:
        body, etag = futures_data.payload(conn, through)
    except LookupError:
        raise NotFound("No futures prices yet: run a Futures prices job") from None
    headers = {"ETag": etag, "Cache-Control": "no-cache"}
    if _etag_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=304, headers=headers)
    return Response(content=body, media_type="application/octet-stream", headers=headers)


@owner_router.get("/status")
def status(conn: psycopg.Connection = DB) -> list[dict[str, Any]]:
    """Per symbol: how many bars are cached, how far they reach, when and with what error."""
    return jsonable(data.data_status(conn))
