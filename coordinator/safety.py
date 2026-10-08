"""Safety controls: the pause switch, the daily loss limit, the money limits, and
approve_and_place(), the one path every order takes to Alpaca.

- Pause all trading: one switch. While it is on, no order is placed for any model;
  backtests and model searches keep running. Only the owner resumes it.
- Daily loss limit: when the account is down more than daily_loss_limit_pct today
  (config/limits.toml, default 2%), trading pauses itself and says why.
- Per-model limits: a buy never takes a model past max_per_position in one symbol or
  max_per_model in total, and never spends more cash than the model's book holds.
- Every order is written to the orders table (model, worker, time, reason) before it
  is sent, and also when it is blocked, with the reason it was blocked.
"""
from __future__ import annotations

import logging
from typing import Any

import psycopg

from coordinator.broker import BrokerStatus, BrokerUnavailable, OrderState
from coordinator.limits import Limits
from coordinator.settings import get_setting, put_setting

log = logging.getLogger(__name__)
PAUSE_LOCK = 7402001
SYSTEM = "system"
# A position may sit up to this many dollars over its cap: smaller changes are not
# worth an order (the same no-trade band the backtester uses).
CAP_TOLERANCE = 25.0
STATUS_MAX_AGE_S = 10.0
# Reasons an order waits rather than fails; the executor retries the decision later.
PAUSED = "trading is paused"
MARKET_CLOSED = "the stock market is closed"
NO_ACCOUNT = "the account value is unavailable, so the daily loss limit cannot be checked"
NO_CLOCK = "the market clock is unavailable"
HALTS = (PAUSED, MARKET_CLOSED, NO_ACCOUNT, NO_CLOCK)


def is_paused(conn: psycopg.Connection) -> bool:
    """True while trading is paused (only the JSON boolean true counts)."""
    return get_setting(conn, "trading_paused", False) is True


def paused_reason(conn: psycopg.Connection) -> str | None:
    return get_setting(conn, "paused_reason", None) if is_paused(conn) else None


def pause_trading(conn: psycopg.Connection, actor: str | None, reason: str) -> bool:
    """Pause all trading; True when this call changed it (an audit row is written)."""
    conn.execute("SELECT pg_advisory_xact_lock(%s)", (PAUSE_LOCK,))
    if is_paused(conn):
        return False
    put_setting(conn, "trading_paused", True, actor, "trading_paused")
    put_setting(conn, "paused_reason", reason, actor, "trading_paused_reason")
    log.warning("trading paused by %s: %s", actor, reason)
    return True


def resume_trading(conn: psycopg.Connection, actor: str | None) -> bool:
    """Resume trading; True when this call changed it. Never called by the system."""
    conn.execute("SELECT pg_advisory_xact_lock(%s)", (PAUSE_LOCK,))
    if not is_paused(conn):
        return False
    put_setting(conn, "trading_paused", False, actor, "trading_resumed")
    put_setting(conn, "paused_reason", None, actor, "trading_paused_reason")
    return True


def check_daily_loss(conn: psycopg.Connection, status: BrokerStatus, limits: Limits) -> str | None:
    """Pause trading when the account is down more than the limit today. Returns the
    reason when the limit is breached (whether or not this call paused), else None."""
    account = status.account_info
    if account is None or account.last_equity <= 0:
        return None
    down = -account.day_change_pct
    if down <= limits.daily_loss_limit_pct:
        return None
    reason = (f"Daily loss limit reached: the account is down {down:.2f}% today "
              f"(limit {limits.daily_loss_limit_pct:g}%)")
    pause_trading(conn, SYSTEM, reason)
    return reason


def book_value(conn: psycopg.Connection, book_id: int, prices: dict[str, float]) -> tuple[float, dict[str, float]]:
    """(invested dollars, {symbol: position value}) of a book at the given prices."""
    rows = conn.execute("SELECT symbol, qty FROM positions WHERE book_id = %s", (book_id,)).fetchall()
    values = {r["symbol"]: r["qty"] * prices.get(r["symbol"], 0.0) for r in rows}
    return sum(values.values()), values


def pending_buys(conn: psycopg.Connection, book_id: int) -> float:
    """Dollars of buy orders sent but not filled yet (they will spend the book's cash)."""
    row = conn.execute(
        "SELECT COALESCE(sum(notional), 0) AS d FROM orders WHERE book_id = %s AND side = 'buy'"
        " AND status IN ('submitting', 'submitted', 'partially_filled')",
        (book_id,),
    ).fetchone()
    return float(row["d"])


def limit_problem(conn: psycopg.Connection, book: dict[str, Any], order: dict[str, Any], prices: dict[str, float],
                  limits: Limits) -> str | None:
    """Why the order breaks a money limit, or None. Sells only need something to sell."""
    held = conn.execute("SELECT qty FROM positions WHERE book_id = %s AND symbol = %s",
                        (book["id"], order["symbol"])).fetchone()
    if order["side"] == "sell":
        if held is None or held["qty"] <= 0:
            return f"nothing to sell: the model holds no {order['symbol']}"
        if order["qty"] > held["qty"] * (1 + 1e-9):
            return f"sell of {order['qty']:.6g} is more than the {held['qty']:.6g} held"
        return None
    dollars = float(order["notional"])
    if dollars <= 0:
        return "a buy must be for more than $0"
    free_cash = book["cash"] - pending_buys(conn, book["id"])
    if dollars > free_cash + 0.01:
        return f"not enough cash in the model's book (${free_cash:,.2f} free)"
    invested, values = book_value(conn, book["id"], prices)
    if values.get(order["symbol"], 0.0) + dollars > limits.max_per_position + CAP_TOLERANCE:
        return f"over the ${limits.max_per_position:,.0f} limit per position"
    if invested + pending_buys(conn, book["id"]) + dollars > limits.max_per_model + CAP_TOLERANCE:
        return f"over the ${limits.max_per_model:,.0f} limit per model"
    return None


def _insert(conn: psycopg.Connection, book: dict[str, Any], order: dict[str, Any], status: str,
            error: str | None) -> dict[str, Any]:
    return conn.execute(
        """
        INSERT INTO orders (book_id, model_id, worker_id, mode, symbol, side, notional, qty, reason, status, error,
                            finished_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CASE WHEN %s = 'blocked' THEN now() END)
        RETURNING *
        """,
        (book["id"], book["model_id"], order.get("worker_id"), book["mode"], order["symbol"], order["side"],
         order.get("notional"), order.get("qty"), order["reason"], status, error, status),
    ).fetchone()


def approve_and_place(conn: psycopg.Connection, status: BrokerStatus, limits: Limits, book: dict[str, Any],
                      order: dict[str, Any], prices: dict[str, float]) -> dict[str, Any]:
    """Check one order and, when it passes, send it to Alpaca. Always logs a row.

    order: {"symbol", "side": "buy"|"sell", "notional" (buys) | "qty" (sells), "reason",
    "worker_id"}. Checks, in order: the pause switch, the broker's mode against the
    book's, a fresh account reading and the daily loss limit (no reading means no
    order: it fails closed), the stock market being open (no clock means closed), then
    the money limits. The row is committed as "submitting" before Alpaca is called, so
    an order can never reach Alpaca without a record of it. The pause lock is held
    until Alpaca has answered, so once pause_trading() returns no order is in flight.
    """
    broker = status.broker
    conn.execute("SELECT pg_advisory_lock(%s)", (PAUSE_LOCK,))
    try:
        problem = None
        if is_paused(conn):
            problem = PAUSED
        elif broker.mode != book["mode"]:
            problem = f"this model trades {book['mode']} money but the coordinator is in {broker.mode} mode"
        if problem is None:
            status.refresh(force=status.age() > STATUS_MAX_AGE_S)
            account = status.account_info
            if account is None or account.last_equity <= 0:
                problem = NO_ACCOUNT
            else:
                problem = check_daily_loss(conn, status, limits)
        if problem is None and "/" not in order["symbol"]:
            if status.clock_info is None:
                problem = NO_CLOCK
            elif not status.clock_info.is_open:
                problem = MARKET_CLOSED
        if problem is None:
            problem = limit_problem(conn, book, order, prices, limits)
        if problem is not None:
            row = _insert(conn, book, order, "blocked", problem)
            conn.commit()
            return row
        row = _insert(conn, book, order, "submitting", None)
        conn.commit()
        try:
            state = broker.place_order(str(row["id"]), order["symbol"], order["side"],
                                       notional=order.get("notional"), qty=order.get("qty"))
        except BrokerUnavailable as exc:
            state = OrderState("submitted", error=f"no clear answer from Alpaca ({exc}); checking")
        if state.status == "rejected":
            row = conn.execute(
                "UPDATE orders SET status = 'rejected', error = %s, finished_at = now() WHERE id = %s RETURNING *",
                (state.error, row["id"]),
            ).fetchone()
        else:
            row = conn.execute(
                "UPDATE orders SET status = 'submitted', broker_order_id = %s, error = %s, submitted_at = now()"
                " WHERE id = %s RETURNING *",
                (state.broker_order_id, state.error, row["id"]),
            ).fetchone()
        conn.commit()
        return row
    finally:
        conn.rollback()
        conn.execute("SELECT pg_advisory_unlock(%s)", (PAUSE_LOCK,))
        conn.commit()
