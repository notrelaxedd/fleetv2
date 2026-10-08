"""Tokens, enrollment, worker bearer verification and owner checks."""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg

from coordinator.config import Config, normalise_origin
from coordinator.errors import Forbidden, Unauthorized

ENROLL_TTL_SECONDS = 3600


def mint_token() -> str:
    """A fresh random bearer/enroll token."""
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """sha256 hex digest; only hashes are stored."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_worker_id() -> str:
    """Internal worker ids look like w_3f9a1c (the dashboard shows the name, e.g. w3)."""
    return "w_" + secrets.token_hex(3)


def create_enroll_token(
    conn: psycopg.Connection, ttl_seconds: int = ENROLL_TTL_SECONDS
) -> tuple[str, datetime]:
    """Mint a single-use enroll token; returns (token, expires_at)."""
    token = mint_token()
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
    conn.execute(
        "INSERT INTO enroll_tokens (token_hash, expires_at) VALUES (%s, %s)",
        (hash_token(token), expires_at),
    )
    return token, expires_at


def lock_enroll_token(conn: psycopg.Connection, token: str) -> dict[str, Any]:
    """Lock an unused, unexpired enroll token row; 401 otherwise."""
    row = conn.execute(
        "SELECT token_hash FROM enroll_tokens"
        " WHERE token_hash = %s AND used_at IS NULL AND expires_at > now() FOR UPDATE",
        (hash_token(token),),
    ).fetchone()
    if row is None:
        raise Unauthorized("invalid, used or expired enroll token")
    return row


def mark_enroll_token_used(conn: psycopg.Connection, token_hash: str, worker_id: str) -> None:
    """Record which worker consumed the token."""
    conn.execute(
        "UPDATE enroll_tokens SET used_at = now(), used_by_worker_id = %s WHERE token_hash = %s",
        (worker_id, token_hash),
    )


def _same_hash(stored: str | None, presented: str) -> bool:
    """Constant-time comparison that treats a missing stored hash as a mismatch."""
    return stored is not None and secrets.compare_digest(stored, presented)


def rotate_worker_token(conn: psycopg.Connection, worker_id: str, presented_hash: str) -> str:
    """Store a new token hash and remember the hash the worker just presented.

    The presented hash stays valid for one more register (not for heartbeats) so a
    worker that never saw the reply can retry. Returns the plain new token.
    """
    token = mint_token()
    conn.execute(
        "UPDATE workers SET token_hash = %s, prev_token_hash = %s WHERE id = %s",
        (hash_token(token), presented_hash, worker_id),
    )
    return token


def verify_worker_token(conn: psycopg.Connection, worker_id: str, token: str) -> dict[str, Any]:
    """The worker row when `token` is its current token, else 401 (heartbeats, job routes)."""
    row = conn.execute("SELECT * FROM workers WHERE id = %s", (worker_id,)).fetchone()
    if row is None or not _same_hash(row["token_hash"], hash_token(token)):
        raise Unauthorized("invalid worker token")
    return row


def verify_register_token(conn: psycopg.Connection, worker_id: str, token: str) -> tuple[dict[str, Any], str]:
    """The locked worker row plus the presented hash when `token` is the current or the
    previous token (a retried register after a lost reply), else 401."""
    row = conn.execute("SELECT * FROM workers WHERE id = %s FOR UPDATE", (worker_id,)).fetchone()
    presented = hash_token(token)
    if row is None:
        raise Unauthorized("invalid worker token")
    if _same_hash(row["token_hash"], presented) or _same_hash(row["prev_token_hash"], presented):
        return row, presented
    raise Unauthorized("invalid worker token")


def worker_for_token(conn: psycopg.Connection, token: str) -> dict[str, Any]:
    """The worker owning `token` (routes without a worker id in the path), else 401."""
    row = conn.execute("SELECT * FROM workers WHERE token_hash = %s", (hash_token(token),)).fetchone()
    if row is None:
        raise Unauthorized("invalid worker token")
    return row


def bearer_token(authorization: str | None) -> str:
    """Extract the token from an Authorization header; 401 if absent."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise Unauthorized("missing bearer token")
    token = authorization[7:].strip()
    if not token:
        raise Unauthorized("missing bearer token")
    return token


def check_owner(config: Config, login_header: str | None) -> str:
    """Owner identity for a request; raises Unauthorized unless allowed.

    Returns the actor name recorded in audit rows.
    """
    login = (login_header or "").strip()
    if config.dev:
        return login or "dev"
    # FLEET_OWNER_LOGIN holds one login or a comma separated list; every listed login is an owner.
    owners = [o.strip().lower() for o in config.owner_login.split(",") if o.strip()]
    if not owners:
        raise Unauthorized("FLEET_OWNER_LOGIN is not configured")
    if not login:
        raise Unauthorized(
            "owner login required: the request carried no Tailscale-User-Login header, so it did"
            " not come through tailscale serve from a logged-in tailnet device"
        )
    # Logins are email addresses; compare them case-insensitively, in constant time.
    presented = login.lower().encode("utf-8", "surrogateescape")
    matches = [secrets.compare_digest(presented, owner.encode("utf-8")) for owner in owners]
    if not any(matches):
        raise Unauthorized(
            f"owner login required: this device is signed in to Tailscale as {login!r},"
            f" but FLEET_OWNER_LOGIN is set to {config.owner_login!r}"
        )
    return login


def owner_from_worker_ip(conn: psycopg.Connection, config: Config, peer_ip: str | None) -> None:
    """Refuse owner requests that come from a registered worker machine.

    A compromised job on a worker must not be able to reach owner routes through
    tailscale serve (worker nodes logged in as the owner carry the owner's login).
    Disabled with FLEET_OWNER_ALLOW_WORKER_IPS=1 or in dev mode.
    """
    if config.dev or config.allow_worker_ips or not peer_ip:
        return
    row = conn.execute("SELECT id FROM workers WHERE remote_ip = %s LIMIT 1", (peer_ip,)).fetchone()
    if row is not None:
        raise Forbidden(
            f"owner requests from a worker machine are refused ({peer_ip} is worker {row['id']});"
            " set FLEET_OWNER_ALLOW_WORKER_IPS=1 to allow it"
        )


def check_origin(config: Config, method: str, origin_header: str | None) -> None:
    """CSRF guard: a present Origin on a state-changing request must be allowed."""
    if method.upper() in {"GET", "HEAD", "OPTIONS"} or origin_header is None:
        return
    if normalise_origin(origin_header) not in config.allowed_origins:
        raise Forbidden("origin not allowed")
