"""FastAPI application factory."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from coordinator import db, web
from coordinator.api import dashboard, data, dl, jobs, models, owner, trading, workers
from coordinator.broker import BrokerStatus, make_broker
from coordinator.bundle import build_bundle
from coordinator.config import Config
from coordinator.data import make_bar_source
from coordinator.futures_data import make_futures_source
from coordinator.errors import QueueError
from coordinator.limits import load_limits
from fleet2.sim.topstep import load_rules
from coordinator import models as starter_models

log = logging.getLogger(__name__)

MAX_BODY_BYTES = 1024 * 1024  # above every payload cap (coordinator.api.limits) with room for a heartbeat's job list

# The dashboard is authenticated by the network (tailscale serve adds the owner login),
# so any page the owner visits could frame it and click-jack a form: refuse framing on
# every non-API response. Static files and error pages get the headers too.
DASHBOARD_HEADERS = (
    (b"x-frame-options", b"DENY"),
    (b"content-security-policy", b"frame-ancestors 'none'"),
    (b"referrer-policy", b"same-origin"),
    (b"x-content-type-options", b"nosniff"),
)


class BodyTooLarge(Exception):
    """Raised from the wrapped receive once a body passes the limit."""


class BodySizeLimit:
    """Reject request bodies above MAX_BODY_BYTES with 413 (worker writes are small).

    A declared Content-Length above the limit is refused before the body is read; a
    chunked body is counted as it arrives and refused once it passes the limit, so a
    missing Content-Length is not a way around it.
    """

    def __init__(self, app: Any, limit: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.limit = limit

    async def _reject(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        response = web.error_response(Request(scope), 413, "request body too large")
        await response(scope, receive, send)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers", []):
            if name == b"content-length" and value.isdigit() and int(value) > self.limit:
                await self._reject(scope, receive, send)
                return
        received = 0
        started = False

        async def counting_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.limit:
                    raise BodyTooLarge()
            return message

        async def tracking_send(message: dict[str, Any]) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except BodyTooLarge:
            if started:
                raise
            await self._reject(scope, receive, send)


class DashboardHeaders:
    """Add the anti-framing and referrer headers to every response outside /api/."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope.get("path", "").startswith("/api/"):
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {name for name, _ in headers}
                headers.extend((name, value) for name, value in DASHBOARD_HEADERS if name not in present)
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


def _validation_detail(exc: RequestValidationError) -> str:
    """A one-line description of the first validation problem."""
    errors = exc.errors()
    if not errors:
        return "invalid request"
    first = errors[0]
    where = ".".join(str(p) for p in first.get("loc", ()))
    return f"{where}: {first.get('msg', 'invalid')}"


def create_app(config: Config, broker_status: BrokerStatus | None = None) -> FastAPI:
    """Build the app; the pool opens in the lifespan so tests can control it.

    The broker is chosen once here (paper unless every live condition holds) and
    never switched while the coordinator runs."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pool = db.make_pool(config.database_url)
        with app.state.pool.connection() as conn:
            starter_models.sync_starters(conn)
        try:
            yield
        finally:
            app.state.pool.close()

    app = FastAPI(title="fleet-v2 coordinator", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.config = config
    app.state.limits = load_limits(config.limits_path)
    app.state.topstep = load_rules(config.topstep_path)
    if broker_status is None:
        broker_status = BrokerStatus(make_broker(config, _live_confirmed(config)))
    app.state.broker_status = broker_status
    app.state.bar_source = make_bar_source(config)
    app.state.futures_source = make_futures_source(config)
    app.state.bundle = build_bundle()
    log.info("worker bundle code_version=%s", app.state.bundle.code_version)

    @app.exception_handler(QueueError)
    async def queue_error(request: Request, exc: QueueError) -> Response:
        """JSON under /api, a small HTML page on the dashboard (401, 403, 404, ...)."""
        return web.error_response(request, exc.status, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> Response:
        return web.error_response(request, 400, _validation_detail(exc))

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        """Starlette's own 404 (unknown path, missing static file) and 405 as HTML on the dashboard."""
        return web.error_response(request, exc.status_code, str(exc.detail))

    app.include_router(workers.router)
    app.include_router(jobs.router)
    app.include_router(data.router)
    app.include_router(data.owner_router)
    app.include_router(owner.router)
    app.include_router(models.worker_router)
    app.include_router(models.owner_router)
    app.include_router(models.search_router)
    app.include_router(trading.worker_router)
    app.include_router(trading.owner_router)
    app.include_router(owner.health_router)
    app.include_router(dl.router)
    app.include_router(dashboard.router)
    # The stylesheet and script need no owner login; every other dashboard path does.
    app.mount("/static", StaticFiles(directory=str(web.STATIC_DIR)), name="static")
    app.add_middleware(BodySizeLimit)
    app.add_middleware(DashboardHeaders)
    return app


def _live_confirmed(config: Config) -> Any:
    """The owner's stored live confirmation (read once at start; None when unreadable)."""
    try:
        with db.connect(config.database_url) as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = 'live_confirmed'").fetchone()
    except Exception:  # noqa: BLE001 - no database yet means not confirmed
        return None
    return row["value"] if row else None
