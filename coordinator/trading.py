"""Paper trading on the coordinator: books, signals from workers, turning a model's
decision into orders, and booking fills.

How a model trades:
1. The owner starts paper trading a model: a book with the starting balance
   (config/limits.toml) and a paper_trade job on a worker.
2. The worker computes the model's target weights whenever a new bar closes and posts
   them here as a signal. Workers never see keys and never call Alpaca.
3. The executor (coordinator background loop) turns the latest decision into orders,
   sized exactly as the backtester sizes them (fleet2.sim.backtest.rebalance_trades:
   same caps, same $25 minimum order, same 20% rebalance band), and sends each
   through coordinator.safety.approve_and_place. Stock orders wait for the market to
   open; crypto trades around the clock; nothing trades while paused.
4. The fill poller books what Alpaca filled into the model's book.
Stopping paper trading sells the model's positions (when trading is allowed) and then
closes the book.

The news strategies (models with origin 'news') also trade from books, but
coordinator.news_trading places their orders: execute_pass leaves their books alone,
and start/stop here refuse them. Only their books can hold a short (a negative qty);
see _book_fill for how a short is booked.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from coordinator.broker import BrokerStatus, BrokerUnavailable
from coordinator.errors import BadRequest, Conflict, NotFound
from coordinator.limits import Limits
from coordinator.models import get_model
from coordinator.safety import HALTS, _origin, approve_and_place, is_paused
from coordinator.broker import NEVER_REACHED
from fleet2.models import get_module
from fleet2.models.base import BadTargets, clean_targets
from fleet2.sim.backtest import COSTS, Limits as SimLimits, rebalance_trades

log = logging.getLogger(__name__)
OPEN_ORDER = ("submitting", "submitted", "partially_filled")
# A "submitting" row older than this was left by a crash between the record and Alpaca's
# answer: the fill poller asks Alpaca about it by its client order id.
STALE_SUBMITTING_S = 15
# After Alpaca refused a sale of a stopping model, wait this long before trying again.
CLOSE_RETRY_S = 300
NEWS_ONLY = "News strategies are started and stopped from the News screen, never as a paper-trading model"


def _sim_limits(limits: Limits) -> SimLimits:
    return SimLimits(limits.starting_balance_per_model, limits.max_per_position, limits.max_per_model)


def open_book(conn: psycopg.Connection, model_id: str) -> dict[str, Any] | None:
    return conn.execute("SELECT * FROM books WHERE model_id = %s AND status IN ('active', 'closing')",
                        (model_id,)).fetchone()


def latest_prices(conn: psycopg.Connection, symbols: list[str]) -> dict[str, float]:
    """The last cached close per symbol (used to size orders and value books)."""
    if not symbols:
        return {}
    rows = conn.execute(
        """
        SELECT DISTINCT ON (symbol) symbol, close FROM bars WHERE symbol = ANY(%s)
         ORDER BY symbol, ts DESC
        """,
        (symbols,),
    ).fetchall()
    return {r["symbol"]: float(r["close"]) for r in rows}


def latest_prices_from_pool(pool: Any, symbols: list[str]) -> dict[str, float]:
    with pool.connection() as conn:
        return latest_prices(conn, symbols)


# ------------------------------------------------------------------ start and stop


def start(conn: psycopg.Connection, model_id: str, mode: str, limits: Limits) -> dict[str, Any]:
    """Open a book for the model and set it to Paper trading. The caller creates the
    paper_trade job and stores its id with attach_job()."""
    model = get_model(conn, model_id)
    if model["market"] == "futures":
        raise Conflict("Futures models paper trade from the Models screen's Futures view (Start paper trading on "
                       "Alpaca), never through the stock and crypto books")
    if model["origin"] == "news":
        raise Conflict(NEWS_ONLY)
    if model["status"] == "retired":
        raise Conflict("This model is retired")
    if model["metrics"] is None:
        raise Conflict("Run a backtest first, so you know what you are trading")
    if open_book(conn, model_id) is not None:
        raise Conflict(f"{model['name']} is already paper trading")
    money = limits.starting_balance_per_model
    book = conn.execute(
        "INSERT INTO books (model_id, mode, starting_balance, cash) VALUES (%s, %s, %s, %s) RETURNING *",
        (model_id, mode, money, money),
    ).fetchone()
    conn.execute("UPDATE models SET status = 'paper_trading' WHERE id = %s", (model_id,))
    return book


def attach_job(conn: psycopg.Connection, book_id: int, job_id: Any) -> None:
    conn.execute("UPDATE books SET job_id = %s WHERE id = %s", (job_id, book_id))


def stop(conn: psycopg.Connection, model_id: str) -> dict[str, Any]:
    """Stop the model's job; its positions are sold by the executor, then the book closes."""
    if get_model(conn, model_id)["origin"] == "news":
        raise Conflict(NEWS_ONLY)
    book = open_book(conn, model_id)
    if book is None:
        raise Conflict("This model is not paper trading")
    conn.execute("UPDATE books SET status = 'closing', targets = '{}'::jsonb, waiting = NULL WHERE id = %s", (book["id"],))
    conn.execute("UPDATE models SET status = 'backtested' WHERE id = %s AND status = 'paper_trading'", (model_id,))
    if book["job_id"] is not None:
        conn.execute(
            "UPDATE jobs SET status = CASE WHEN status = 'queued' THEN 'cancelled' ELSE 'cancel_requested' END,"
            " finished_at = CASE WHEN status = 'queued' THEN now() ELSE finished_at END, updated_at = now()"
            " WHERE id = %s AND status IN ('queued', 'leased')",
            (book["job_id"],),
        )
    return book


# ------------------------------------------------------------------ signals


def receive_signal(conn: psycopg.Connection, worker_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """A worker's decision for a model it is paper trading. Kept as the book's latest
    decision; the executor trades it. Older or repeated decisions are ignored."""
    job = conn.execute("SELECT * FROM jobs WHERE id = %s", (body.get("job_id"),)).fetchone()
    if job is None or job["kind"] != "paper_trade" or job["lease_worker_id"] != worker_id or job["status"] != "leased":
        raise Conflict("this worker is not running that paper trade job")
    model_id = job["model_id"]
    book = conn.execute("SELECT * FROM books WHERE model_id = %s AND status = 'active' FOR UPDATE", (model_id,)).fetchone()
    if book is None:
        raise Conflict("the model is not paper trading")
    module = get_module(get_model(conn, model_id)["module"])
    try:
        targets = clean_targets(body.get("targets") or {}, module.SYMBOLS)
    except BadTargets as exc:
        raise BadRequest(f"bad targets: {exc}") from None
    bar_t = int(body.get("bar_t") or 0)
    reason = str(body.get("reason") or "")[:500]
    if book["targets_bar_t"] is not None and bar_t <= book["targets_bar_t"]:
        outcome = "already have this decision"
    else:
        conn.execute(
            """
            UPDATE books SET targets = %s, targets_bar_t = %s, targets_reason = %s, targets_worker = %s,
                   targets_at = now(), job_id = %s WHERE id = %s
            """,
            (Jsonb(targets), bar_t, reason, worker_id, job["id"], book["id"]),
        )
        outcome = "kept"
    conn.execute(
        "INSERT INTO signals (book_id, model_id, worker_id, bar_t, targets, reason, outcome) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (book["id"], model_id, worker_id, bar_t, Jsonb(targets), reason, outcome),
    )
    return {"outcome": outcome}


# ------------------------------------------------------------------ planning


def plan_orders(book: dict[str, Any], held: dict[str, float], prices: dict[str, float], weights: dict[str, float],
                limits: Limits) -> list[dict[str, Any]]:
    """The orders that move the book from what it holds to the model's weights, sized
    by the backtester's own rebalance_trades: weight x book value, capped per position
    and per model, no order under $25 (except selling a position out entirely), and a
    held position resized only when 20% of its target away. Sells come first."""
    values = {s: q * prices[s] for s, q in held.items() if s in prices and q > 0}
    equity = book["cash"] + sum(values.values())
    tradable = {s: w for s, w in weights.items() if s in prices}
    orders = []
    for symbol, delta in sorted(rebalance_trades(values, tradable, equity, _sim_limits(limits)).items()):
        have = values.get(symbol, 0.0)
        if delta > 0:
            orders.append({"symbol": symbol, "side": "buy", "notional": round(delta, 2), "delta": delta})
        elif delta <= -have:  # sell out entirely
            orders.append({"symbol": symbol, "side": "sell", "qty": held[symbol], "delta": -have})
        else:
            orders.append({"symbol": symbol, "side": "sell", "qty": held[symbol] * -delta / have, "delta": delta})
    return sorted(orders, key=lambda o: o["delta"])


def _reason(order: dict[str, Any], model_name: str, weights: dict[str, float], decided: str) -> str:
    symbol = order["symbol"]
    if order["side"] == "buy":
        return f"{model_name} wants {weights.get(symbol, 0) * 100:.0f}% of its money in {symbol} ({decided})"
    if symbol not in weights:
        return f"{model_name} no longer wants {symbol} ({decided})"
    return f"{model_name} wants less {symbol}: {weights[symbol] * 100:.0f}% of its money ({decided})"


def _decided(book: dict[str, Any]) -> str:
    if book["status"] == "closing":
        return "paper trading stopped"
    if book["targets_bar_t"] is None:
        return "no decision yet"
    when = datetime.fromtimestamp(book["targets_bar_t"], timezone.utc)
    return "decision after the bar of " + when.strftime("%b %d %H:%M UTC")


# ------------------------------------------------------------------ executor


def _set_waiting(conn: psycopg.Connection, book_id: int, text: str | None) -> None:
    conn.execute("UPDATE books SET waiting = %s WHERE id = %s", (text, book_id))
    conn.commit()


def execute_pass(conn: psycopg.Connection, status: BrokerStatus, limits: Limits) -> int:
    """Trade every book whose latest decision is not traded yet. Returns orders sent."""
    sent = 0
    books = conn.execute(
        """
        SELECT b.*, m.name AS model_name, m.market FROM books b JOIN models m ON m.id = b.model_id
         WHERE m.origin <> 'news'  -- coordinator.news_trading trades the news books
           AND (b.status = 'closing'
                OR (b.status = 'active' AND b.targets_bar_t IS NOT NULL
                    AND b.targets_bar_t IS DISTINCT FROM b.executed_bar_t))
         ORDER BY b.id
        """
    ).fetchall()
    conn.commit()
    for book in books:
        if conn.execute("SELECT 1 FROM orders WHERE book_id = %s AND status = ANY(%s)", (book["id"], list(OPEN_ORDER))).fetchone():
            continue  # wait for the last orders to settle before deciding again
        if is_paused(conn):
            _set_waiting(conn, book["id"], "Waiting: trading is paused")
            continue
        if book["mode"] != status.broker.mode:
            _set_waiting(conn, book["id"], f"Waiting: the coordinator is in {status.broker.mode} mode")
            continue
        if book["market"] == "stocks" and not (status.clock_info and status.clock_info.is_open):
            _set_waiting(conn, book["id"], "Waiting for the stock market to open")
            continue
        if book["status"] == "closing" and conn.execute(
                "SELECT 1 FROM orders WHERE book_id = %s AND status = 'rejected'"
                " AND finished_at > now() - make_interval(secs => %s)", (book["id"], CLOSE_RETRY_S)).fetchone():
            _set_waiting(conn, book["id"], "Alpaca refused a sale; trying again in a few minutes")
            continue
        held = {r["symbol"]: r["qty"] for r in conn.execute("SELECT symbol, qty FROM positions WHERE book_id = %s", (book["id"],))}
        weights = dict(book["targets"] or {})
        prices = latest_prices(conn, sorted(set(held) | set(weights)))
        decided = _decided(book)
        halted = None
        for order in plan_orders(book, held, prices, weights, limits):
            order["reason"] = _reason(order, book["model_name"], weights, decided)
            order["worker_id"] = book["targets_worker"]
            fresh = conn.execute("SELECT * FROM books WHERE id = %s", (book["id"],)).fetchone()
            row = approve_and_place(conn, status, limits, fresh, order, prices)
            sent += row["status"] != "blocked"
            if row["status"] == "blocked" and (row["error"] in HALTS or (row["error"] or "").startswith("Daily loss")):
                halted = row["error"]
                break  # paused, market closed, loss limit: try the whole decision again later
            poll_fills(conn, status, book_id=book["id"])
            if order["side"] == "sell" and conn.execute(
                    "SELECT 1 FROM orders WHERE book_id = %s AND status = ANY(%s)", (book["id"], list(OPEN_ORDER))).fetchone():
                halted = "a sale is still filling"  # buys are planned again once the cash is in
                break
        if halted:
            _set_waiting(conn, book["id"], "Waiting: " + halted)
            continue
        conn.execute("UPDATE books SET executed_bar_t = targets_bar_t, waiting = NULL WHERE id = %s", (book["id"],))
        if book["status"] == "closing":
            close_if_flat(conn, book["id"])
        conn.commit()
    return sent


def close_if_flat(conn: psycopg.Connection, book_id: int) -> bool:
    """A closing book with nothing held (long or short) and nothing in flight becomes closed."""
    if conn.execute("SELECT 1 FROM positions WHERE book_id = %s AND qty <> 0", (book_id,)).fetchone():
        return False
    if conn.execute("SELECT 1 FROM orders WHERE book_id = %s AND status = ANY(%s)", (book_id, list(OPEN_ORDER))).fetchone():
        return False
    conn.execute("UPDATE books SET status = 'closed', closed_at = now(), waiting = NULL WHERE id = %s", (book_id,))
    return True


# ------------------------------------------------------------------ fills


def _book_fill(conn: psycopg.Connection, order: dict[str, Any], new_qty: float, price: float) -> None:
    """Move a newly filled quantity into the model's book (cash and position).

    Buys by dollar amount (notional) charge the book exactly the dollars filled. Alpaca
    takes its crypto fee out of what you receive: fewer coins on a buy, fewer dollars on
    a sale. Stocks pay no fee (slippage is already in the fill price).

    Shorts (news stock books only): a sale with nothing held, or while short, makes qty
    negative and adds the proceeds to cash; a buy while short (buy to cover) moves qty
    back toward 0 and takes the dollars from cash. `cost` keeps the sign of qty: for a
    short it is minus the dollars received for the shares still short. So, long or
    short, cost / qty is the average entry price and qty x price - cost is the open
    profit, and the book's value is always cash + sum(qty x price). A book whose model
    is not a news strategy never goes short: a sale filled bigger than what it held is
    booked only up to what it held (and a warning is logged).

    `price` is the average price of this increment (poll_fills works it out from
    Alpaca's average over the whole order)."""
    fee = COSTS["crypto"].fee_bps / 10_000.0 if "/" in order["symbol"] else 0.0
    key = (order["book_id"], order["symbol"])
    pos = conn.execute("SELECT qty, cost FROM positions WHERE book_id = %s AND symbol = %s FOR UPDATE", key).fetchone()
    held = float(pos["qty"]) if pos else 0.0
    if order["side"] == "buy":
        covered = min(new_qty, -held) if held < 0 else 0.0
        if covered > 0:  # buy to cover: pay for the shares handed back
            _reduce(conn, key, pos, covered, price)
            conn.execute("UPDATE books SET cash = cash - %s WHERE id = %s", (covered * price, order["book_id"]))
        bought = new_qty - covered
        if bought <= 0:
            return
        spent = bought * price
        conn.execute(
            """
            INSERT INTO positions (book_id, symbol, qty, cost) VALUES (%s, %s, %s, %s)
            ON CONFLICT (book_id, symbol) DO UPDATE SET qty = positions.qty + EXCLUDED.qty,
                   cost = positions.cost + EXCLUDED.cost
            """,
            (*key, bought * (1.0 - fee), spent),
        )
        conn.execute("UPDATE books SET cash = cash - %s WHERE id = %s", (spent, order["book_id"]))
        return
    sold = min(new_qty, held) if held > 0 else 0.0
    if sold > 0:  # sell what is held
        _reduce(conn, key, pos, sold, price)
        conn.execute("UPDATE books SET cash = cash + %s WHERE id = %s", (sold * price * (1.0 - fee), order["book_id"]))
    shorted = new_qty - sold
    # safety.limit_problem never lets a sale cross from long to short, so after a long is
    # sold out anything left over is rounding, not a short.
    if shorted <= 0 or (held > 0 and shorted * price < 0.01):
        return
    if _origin(conn, order["model_id"]) != "news":
        # Only news books may be short. A bigger fill than was held (say, a sale planned
        # again after a lost answer from Alpaca) is booked only up to what was held.
        log.warning("order %s of %s filled %.6g %s but the book held %.6g: the extra %.6g is not booked as a short "
                    "(only news strategies short); check the Alpaca account", order["id"], order["model_id"],
                    new_qty, order["symbol"], max(held, 0.0), shorted)
        return
    got = shorted * price * (1.0 - fee)
    conn.execute(
        """
        INSERT INTO positions (book_id, symbol, qty, cost) VALUES (%s, %s, %s, %s)
        ON CONFLICT (book_id, symbol) DO UPDATE SET qty = positions.qty + EXCLUDED.qty,
               cost = positions.cost + EXCLUDED.cost
        """,
        (*key, -shorted, -got),
    )
    conn.execute("UPDATE books SET cash = cash + %s WHERE id = %s", (got, order["book_id"]))


def _increment_price(order: dict[str, Any], filled_qty: float, avg: float, new_qty: float) -> float:
    """The average price of the part of a fill not booked yet. Alpaca's filled_avg_price
    is over the whole order, so the increment is (filled_qty x avg - dollars already
    booked) / new_qty. What was booked is booked_qty x the filled_avg_price stored with
    it: poll_fills stores the order's average each time it books, and never clears it."""
    booked_qty = float(order["booked_qty"] or 0.0)
    if booked_qty <= 0 or not order["filled_avg_price"]:
        return avg
    price = (filled_qty * avg - booked_qty * float(order["filled_avg_price"])) / new_qty
    if not math.isfinite(price) or price <= 0:
        log.warning("order %s: increment price %.6g makes no sense; booking at the average %.6g",
                    order["id"], price, avg)
        return avg
    return price


def _reduce(conn: psycopg.Connection, key: tuple[int, str], pos: dict[str, Any], amount: float, price: float) -> None:
    """Take `amount` (> 0) off a long or a short position, keeping cost in proportion;
    a position worth under a cent afterwards is removed."""
    held = float(pos["qty"])
    left = held - amount if held > 0 else held + amount
    if abs(left) * price < 0.01:
        conn.execute("DELETE FROM positions WHERE book_id = %s AND symbol = %s", key)
    else:
        conn.execute("UPDATE positions SET qty = %s, cost = cost * %s WHERE book_id = %s AND symbol = %s",
                     (left, left / held, *key))


def poll_fills(conn: psycopg.Connection, status: BrokerStatus, book_id: int | None = None) -> int:
    """Ask Alpaca about every open order and book what filled. Returns orders updated.

    Includes "submitting" rows left by a crash (older than STALE_SUBMITTING_S): Alpaca
    either has them (their state is adopted) or never got them (rejected, and the
    model's decision is planned again from what the book really holds)."""
    rows = conn.execute(
        """
        SELECT * FROM orders
         WHERE (status IN ('submitted', 'partially_filled')
                OR (status = 'submitting' AND created_at < now() - make_interval(secs => %s)))
           AND (%s::bigint IS NULL OR book_id = %s)
         ORDER BY created_at
        """,
        (STALE_SUBMITTING_S, book_id, book_id),
    ).fetchall()
    changed = 0
    for order in rows:
        try:
            state = status.broker.get_order(str(order["id"]))
        except BrokerUnavailable as exc:
            log.warning("fill check for %s failed: %s", order["id"], exc)
            continue
        new_qty = max(0.0, state.filled_qty - order["booked_qty"])
        booked = 0.0
        if new_qty > 0 and state.filled_avg_price:
            _book_fill(conn, order, new_qty, _increment_price(order, state.filled_qty, state.filled_avg_price, new_qty))
            booked = new_qty
        done = state.status in ("filled", "cancelled", "rejected")
        conn.execute(
            """
            UPDATE orders SET status = %s, filled_qty = %s, filled_avg_price = COALESCE(%s, filled_avg_price),
                   booked_qty = booked_qty + %s,
                   error = CASE WHEN %s IN ('filled', 'partially_filled') THEN NULL ELSE COALESCE(%s, error) END,
                   finished_at = CASE WHEN %s THEN now() ELSE finished_at END
             WHERE id = %s
            """,
            (state.status, state.filled_qty, state.filled_avg_price, booked, state.status, state.error, done, order["id"]),
        )
        if state.status == "rejected" and state.error == NEVER_REACHED:
            # The decision this order belonged to was not carried out: plan it again.
            conn.execute("UPDATE books SET executed_bar_t = NULL WHERE id = %s AND status = 'active'", (order["book_id"],))
        conn.commit()
        changed += 1
    return changed
