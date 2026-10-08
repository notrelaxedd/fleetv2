"""Model search control (filled in by stage 5)."""
from __future__ import annotations

from typing import Any

import psycopg


def search_status(conn: psycopg.Connection) -> dict[str, Any]:
    """{"running": bool, "button": "Start model search" | "Stop model search", "detail": str | None}."""
    return {"running": False, "button": "Start model search", "detail": None}
