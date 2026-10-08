"""Paper-trading lines for the Models screen: one plain sentence per model with a book."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import psycopg

from coordinator.fleet_view import TZ, money, pct
from coordinator.trading import latest_prices


def _since(when: datetime) -> str:
    local = when.astimezone(TZ)
    return local.strftime("%b ") + str(local.day)


def paper_summaries(conn: psycopg.Connection) -> dict[str, dict[str, Any]]:
    """{model_id: {"line", "tone", "value", "change_pct", "positions", "orders"}} for open books."""
    out: dict[str, dict[str, Any]] = {}
    books = conn.execute(
        """
        SELECT b.*, j.status AS job_status,
               (SELECT count(*) FROM orders o WHERE o.book_id = b.id AND o.status <> 'blocked') AS order_count
          FROM books b LEFT JOIN jobs j ON j.id = b.job_id
         WHERE b.status IN ('active', 'closing')
        """
    ).fetchall()
    for book in books:
        held = conn.execute("SELECT symbol, qty FROM positions WHERE book_id = %s", (book["id"],)).fetchall()
        prices = latest_prices(conn, [h["symbol"] for h in held])
        value = book["cash"] + sum(h["qty"] * prices.get(h["symbol"], 0.0) for h in held)
        change = 100.0 * (value / book["starting_balance"] - 1.0)
        mode = "Live" if book["mode"] == "live" else "Paper"
        parts = [f"{mode} trading since {_since(book['started_at'])}: {money(value)} ({pct(change, signed=True, digits=2)})",
                 f"{len(held)} position{'s' if len(held) != 1 else ''}", f"{book['order_count']} orders"]
        if book["status"] == "closing":
            parts.append("stopping: selling its positions")
        elif book["job_status"] not in ("leased", "queued", "cancel_requested"):
            parts.append("no worker is sending its decisions: press Stop, then Start again")
        if book["waiting"]:
            parts.append(book["waiting"])
        out[book["model_id"]] = {"line": " · ".join(parts), "tone": "gain" if change > 0 else "loss" if change < 0 else "plain",
                                 "value": value, "change_pct": change, "positions": len(held), "orders": book["order_count"]}
    return out
