"""Dashboard pages: the shared header, the Fleet screen and its refreshable fragment."""
from __future__ import annotations

from datetime import datetime, timezone

import psycopg
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from coordinator import fleet_view, web
from coordinator.api.deps import DB, require_owner

router = APIRouter(dependencies=[Depends(require_owner)])


@router.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    return RedirectResponse("/fleet")


@router.get("/fleet", response_class=HTMLResponse)
def fleet(request: Request, conn: psycopg.Connection = DB) -> HTMLResponse:
    page = fleet_view.fleet_page(conn, request.app.state.broker_status, request.app.state.limits)
    return web.render("fleet.html", {"current": "fleet", **page})


@router.get("/fragments/fleet", response_class=HTMLResponse)
def fleet_fragment(request: Request, conn: psycopg.Connection = DB) -> HTMLResponse:
    """The parts of the Fleet screen (and the header status) that app.js swaps every 5 seconds."""
    page = fleet_view.fleet_page(conn, request.app.state.broker_status, request.app.state.limits)
    return web.render("fleet_fragment.html", page)


@router.get("/models", response_class=HTMLResponse)
def models(request: Request, conn: psycopg.Connection = DB) -> HTMLResponse:
    header = fleet_view.header(conn, request.app.state.broker_status, datetime.now(timezone.utc))
    return web.render("models.html", {"current": "models", "header": header})
