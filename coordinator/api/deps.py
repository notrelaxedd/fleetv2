"""FastAPI dependencies shared by the routers."""
from __future__ import annotations

from typing import Any, Iterator

import psycopg
from fastapi import Depends, Request
from psycopg_pool import ConnectionPool

from coordinator import auth
from coordinator.config import Config


def get_config(request: Request) -> Config:
    """The Config the app was created with."""
    return request.app.state.config


def get_pool(request: Request) -> ConnectionPool:
    """The open connection pool."""
    return request.app.state.pool


def db_conn(pool: ConnectionPool = Depends(get_pool)) -> Iterator[psycopg.Connection]:
    """One pooled connection per request; commits unless the handler raised.

    Always used through DB below (scope="function") so the commit runs before the
    response is sent: a worker or owner never gets an acknowledgement for a write
    that is not durable yet.
    """
    with pool.connection() as conn:
        yield conn


DB = Depends(db_conn, scope="function")


def remote_ip(request: Request) -> str | None:
    """Peer address as recorded on workers and used for the owner worker-IP check.

    With config.trust_proxy (the default: the app binds loopback behind tailscale
    serve) the last X-Forwarded-For hop is the tailnet peer; earlier hops are client
    supplied and ignored. Without it the socket peer is used and the header is ignored.
    """
    config: Config = request.app.state.config
    forwarded = request.headers.get("x-forwarded-for") if config.trust_proxy else None
    if forwarded:
        last = forwarded.rsplit(",", 1)[-1].strip()
        if last:
            return last
    return request.client.host if request.client else None


def bearer(request: Request) -> str:
    """The worker bearer token from the Authorization header."""
    return auth.bearer_token(request.headers.get("authorization"))


def require_owner(
    request: Request, config: Config = Depends(get_config), conn: psycopg.Connection = DB
) -> str:
    """Owner login check, worker-IP refusal and the Origin CSRF guard; returns the actor name."""
    actor = auth.check_owner(config, request.headers.get("tailscale-user-login"))
    auth.owner_from_worker_ip(conn, config, remote_ip(request))
    auth.check_origin(config, request.method, request.headers.get("origin"))
    return actor


def code_version(request: Request) -> str:
    """code_version of the bundle built at app start."""
    return request.app.state.bundle.code_version


def with_code_version(reply: dict[str, Any], version: str) -> dict[str, Any]:
    """Add the host code_version to a worker-facing reply."""
    reply["code_version"] = version
    return reply
