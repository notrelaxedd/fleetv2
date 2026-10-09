"""Dashboard pages: the shared header, the Fleet screen and its refreshable fragment."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import psycopg
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from markupsafe import Markup

from coordinator import ai_ideas, charts, fleet_view, futures_view, models_view, search, trading_view, web
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


STOCK_PARTS = {"search": "_models_search.html", "list": "_models_list.html", "head": "_model_head.html",
               "actions": "_model_actions.html", "results": "_model_results.html"}
FUTURES_PARTS = {"search": "_futures_search.html", "list": "_futures_list.html", "head": "_futures_head.html",
                 "actions": "_futures_actions.html", "results": "_futures_results.html"}


def _models_context(request: Request, conn: psycopg.Connection) -> dict[str, Any]:
    """The Models screen's data (selected model from ?id=, the Futures view from
    ?market=futures), with the SVGs drawn."""
    now = datetime.now(timezone.utc)
    if request.query_params.get("market") == "futures":
        page = futures_view.futures_page(conn, request.app.state.topstep, request.query_params.get("id"),
                                         search=search.search_status(conn), venues=request.app.state.venues)
        page["haiku"] = ai_ideas.status_line(conn, request.app.state.ai, request.app.state.ai_key)
        parts = FUTURES_PARTS
    else:
        page = models_view.models_page(conn, request.query_params.get("id"), paper=trading_view.paper_summaries(conn),
                                       search=search.search_status(conn))
        parts = STOCK_PARTS
    for row in page["models"]:
        row["spark_svg"] = Markup(charts.sparkline(row["spark"]))
    selected = page["selected"]
    if selected and selected.get("chart"):
        selected["chart_svg"] = Markup(charts.chart_for(selected["chart"]))
    return {**page, "parts": parts, "header": fleet_view.header(conn, request.app.state.broker_status, now)}


@router.get("/models", response_class=HTMLResponse)
def models(request: Request, conn: psycopg.Connection = DB) -> HTMLResponse:
    return web.render("models.html", {"current": "models", **_models_context(request, conn)})


@router.get("/fragments/models", response_class=HTMLResponse)
def models_fragment(request: Request, conn: psycopg.Connection = DB) -> HTMLResponse:
    """The parts of the Models screen (and the header status) that app.js swaps every 5 seconds."""
    return web.render("models_fragment.html", _models_context(request, conn))
