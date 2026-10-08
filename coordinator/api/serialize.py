"""Turn database rows into JSON-safe values."""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any


def iso_utc(value: datetime) -> str:
    """ISO-8601 UTC with a Z suffix."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def jsonable(value: Any) -> Any:
    """Recursively convert datetimes, uuids and Decimals."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, datetime):
        return iso_utc(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)
    return value


def public_worker(row: dict[str, Any]) -> dict[str, Any]:
    """A worker row without its token hashes."""
    return jsonable({k: v for k, v in row.items() if k not in ("token_hash", "prev_token_hash")})
