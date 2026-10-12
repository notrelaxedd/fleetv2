"""The news strategies' executor (docs/NEWS_PLAN.md, sections 2 to 4): turns Claude's
live predictions into paper positions and closes them again.

There are two strategies, `news-stocks` and `news-crypto`, each a models row with origin
'news' and its own book (coordinator.trading's books, positions and orders tables). The
owner starts one only once its saved predictions beat the coin-flip twin
(coordinator.news_scoring.gate). Every order goes through
coordinator.safety.approve_and_place, which also holds the short and cross-book rules,
and fills are booked by coordinator.trading.poll_fills, like any model's.

news_pass() runs on the coordinator's main loop right after the models' own orders, so
fills are never booked by two threads at once. For each news book, every pass:
1. Book fills and move each news position along: opening -> open (with its entry
   price, stop and exit time) or failed (with the reason), closing -> closed (with its
   profit) or back to open when the close did not go through.
2. Wait, as the models' executor does, while the coordinator is in another mode, the
   stock market is closed (stock book) or an order is still filling.
3. Exits first: the latest trade price is through the stop, the horizon has passed, or
   a newer prediction expects the opposite move; a stopping book closes everything.
   While trading is paused only the protective exits go out (a stop, or closing a book
   the owner stopped: safety.approve_and_place lets those through the pause); the
   others wait for the owner to resume.
4. Entries (active books only, never while paused): fresh live predictions (decided
   in the last 30 minutes), confident enough, not already priced in, and expecting a
   move worth cost_multiple round trips of costs. Up -> buy; down -> short (stocks,
   whole shares) or nothing (crypto: Alpaca cannot short coins, and step 3 already
   sells a coin the book holds). One position per symbol; at most max_trades_per_day
   new ones a day.

Stops are watched here against Alpaca's latest trade price (no resting stop order at
Alpaca): if the coordinator is down, stops are not watched.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import psycopg

from coordinator import safety, trading
from coordinator.broker import BrokerStatus
from coordinator.errors import BadRequest, Conflict
from coordinator.limits import Limits
from fleet2.sim.backtest import COSTS, MIN_TRADE_DOLLARS

log = logging.getLogger(__name__)

NEWS_MODELS = {"stocks": "news-stocks", "crypto": "news-crypto"}
# The futures paper books trade SPY and QQQ long and short on the same Alpaca account,
# so the news strategies never trade them (predictions about them are still scored).
NEWS_EXCLUDED = safety.NEWS_EXCLUDED
# A prediction older than this is not traded: the news is likely in the price by then.
ENTRY_WINDOW = timedelta(minutes=30)
ACTIVE = ("opening", "open", "closing")
HEADLINE_CHARS = 160

PriceNow = Callable[[list[str]], dict[str, float]]

MODEL_TEXT = {
    "stocks": {
        "name": "News strategy: stocks",
        "description": "Trades Claude's predictions from the market news on the large US stocks we follow: "
                       "buys before a rise it expects, shorts before a fall.",
        "how_it_works": "Claude reads each news article about a stock we follow and predicts whether it will rise "
                        "or fall, by how much, and within how many days. When a fresh prediction is confident "
                        "enough and the move is big enough to beat the trading costs, the strategy buys the stock, "
                        "or shorts it in whole shares if Claude expects a fall. Each position is closed when its "
                        "time is up, when the price moves too far against it (its stop), or when a newer "
                        "prediction expects the opposite.",
    },
    "crypto": {
        "name": "News strategy: crypto",
        "description": "Trades Claude's predictions from the market news on the coins we follow: buys before a "
                       "rise it expects, and sells a coin it holds when a fall is expected.",
        "how_it_works": "Claude reads each news article about a coin we follow and predicts whether it will rise "
                        "or fall, by how much, and within how many days. When a fresh prediction of a rise is "
                        "confident enough and big enough to beat the trading costs, the strategy buys the coin. "
                        "Alpaca cannot short coins, so a predicted fall only sells a coin the strategy holds; each "
                        "position is also closed when its time is up or the price moves too far against it.",
    },
}


def round_trip_cost_pct(market: str) -> float:
    """Costs of buying and selling once, in percent (stocks 0.10, crypto 0.70), from the
    backtester's own cost model."""
    return 2.0 * COSTS[market].rate * 100.0


def horizon_end(market: str, start: datetime, days: int) -> datetime:
    """When a position opened at `start` with a horizon of `days` is due to close:
    stocks count weekdays (holidays are not known here), crypto counts 24-hour days."""
    if market == "crypto":
        return start + timedelta(hours=24 * days)
    end, left = start, int(days)
    while left > 0:
        end += timedelta(days=1)
        if end.weekday() < 5:
            left -= 1
    return end


def _day_start(now: datetime) -> datetime:
    return now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def _is_halt(error: str | None) -> bool:
    """A block that says "not now" (paused, market closed, no reading, loss limit), as
    opposed to "not this order"."""
    return bool(error) and (error in safety.HALTS or error.startswith("Daily loss"))


def _safe_prices(price_now: PriceNow, symbols: list[str]) -> dict[str, float]:
    """price_now() without ever raising: a symbol with no good price is just missing."""
    if not symbols:
        return {}
    try:
        got = price_now(sorted(set(symbols))) or {}
    except Exception as exc:  # noqa: BLE001 - no price means no entry and no stop check, never a crash
        log.warning("news trading: latest prices unavailable: %s", exc)
        return {}
    return {s: float(p) for s, p in got.items() if p is not None and math.isfinite(float(p)) and float(p) > 0}


# ------------------------------------------------------------------ models, start, stop


def ensure_models(conn: psycopg.Connection) -> None:
    """Insert the two news strategies' models rows (or refresh their wording); their
    status is kept. Called at start-up next to models.sync_starters."""
    for market, model_id in NEWS_MODELS.items():
        text = MODEL_TEXT[market]
        conn.execute(
            """
            INSERT INTO models (id, name, module, market, description, how_it_works, params, origin)
            VALUES (%s, %s, 'news', %s, %s, %s, '{}'::jsonb, 'news')
            ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, description = EXCLUDED.description,
                   how_it_works = EXCLUDED.how_it_works
            """,
            (model_id, text["name"], market, text["description"], text["how_it_works"]),
        )


def _model_id(market: str) -> str:
    if market not in NEWS_MODELS:
        raise BadRequest("market must be stocks or crypto")
    return NEWS_MODELS[market]


def start(conn: psycopg.Connection, market: str, mode: str, limits: Limits) -> dict[str, Any]:
    """Open the news strategy's book with the starting balance. Refused (Conflict, with
    the scoring's own reason) until its saved predictions pass news_scoring.gate. No
    worker job: news_pass places its orders."""
    from coordinator import news_scoring

    model_id = _model_id(market)
    ensure_models(conn)
    ok, reason = news_scoring.gate(conn, market, limits)
    if not ok:
        raise Conflict(reason)
    if trading.open_book(conn, model_id) is not None:
        raise Conflict(f"The {market} news strategy is already trading")
    money = limits.starting_balance_per_model
    book = conn.execute(
        "INSERT INTO books (model_id, mode, starting_balance, cash) VALUES (%s, %s, %s, %s) RETURNING *",
        (model_id, mode, money, money),
    ).fetchone()
    conn.execute("UPDATE models SET status = 'paper_trading' WHERE id = %s", (model_id,))
    return book


def stop(conn: psycopg.Connection, market: str) -> dict[str, Any]:
    """Set the book to closing: news_pass closes every position (sells longs, buys back
    shorts), then closes the book once nothing is held or in flight."""
    model_id = _model_id(market)
    book = trading.open_book(conn, model_id)
    if book is None:
        raise Conflict(f"The {market} news strategy is not trading")
    book = conn.execute("UPDATE books SET status = 'closing', waiting = NULL WHERE id = %s RETURNING *",
                        (book["id"],)).fetchone()
    conn.execute("UPDATE models SET status = NULL WHERE id = %s AND status = 'paper_trading'", (model_id,))
    return book


# ------------------------------------------------------------------ position lifecycle


def _held(conn: psycopg.Connection, book_id: int, symbol: str) -> float:
    row = conn.execute("SELECT qty FROM positions WHERE book_id = %s AND symbol = %s", (book_id, symbol)).fetchone()
    return float(row["qty"]) if row else 0.0


def _pnl(conn: psycopg.Connection, position: dict[str, Any]) -> float | None:
    """Profit of a closed position from its filled orders: what the sales brought in
    (after the crypto fee) minus what the buys cost. Long or short, the same sum: a
    long buys first, a short sells first. Every order of the book in this symbol since
    the opening order belongs to this position (one position per symbol at a time)."""
    if position["open_order_id"] is None:
        return None
    fee = COSTS["crypto"].fee_bps / 10_000.0 if "/" in position["symbol"] else 0.0
    rows = conn.execute(
        """
        SELECT side, filled_qty, filled_avg_price FROM orders
         WHERE book_id = %s AND symbol = %s AND filled_qty > 0 AND filled_avg_price IS NOT NULL
           AND created_at >= (SELECT created_at FROM orders WHERE id = %s)
        """,
        (position["book_id"], position["symbol"], position["open_order_id"]),
    ).fetchall()
    total = 0.0
    for r in rows:
        dollars = r["filled_qty"] * r["filled_avg_price"]
        total += dollars * (1.0 - fee) if r["side"] == "sell" else -dollars
    return total


def _fail(conn: psycopg.Connection, position_id: int, reason: str, now: datetime) -> None:
    conn.execute("UPDATE news_positions SET status = 'failed', close_reason = %s, closed_at = %s WHERE id = %s",
                 (reason[:500], now, position_id))


def _sync(conn: psycopg.Connection, book: dict[str, Any], limits: Limits, now: datetime) -> None:
    """Move every opening or closing news position of the book along, from its order.
    A position never stays 'opening' once its order is settled: it is open (something
    was filled) or failed (with the reason)."""
    rows = conn.execute(
        """
        SELECT np.*, p.horizon_days FROM news_positions np JOIN news_predictions p ON p.id = np.prediction_id
         WHERE np.book_id = %s AND np.status IN ('opening', 'closing') ORDER BY np.id
        """,
        (book["id"],),
    ).fetchall()
    for np in rows:
        if np["status"] == "opening":
            order_id = np["open_order_id"]
            if order_id is None:
                # A crash between the position row and the order: adopt the order if it
                # was written (it belongs to no position yet), else the order was never sent.
                # Only an order written after the position was can be its order.
                orphan = conn.execute(
                    """
                    SELECT id FROM orders o WHERE o.book_id = %s AND o.symbol = %s AND o.created_at >= %s
                       AND NOT EXISTS (SELECT 1 FROM news_positions x WHERE x.open_order_id = o.id
                                                                         OR x.close_order_id = o.id)
                     ORDER BY o.created_at DESC LIMIT 1
                    """,
                    (book["id"], np["symbol"], np["opened_at"]),
                ).fetchone()
                if orphan is None:
                    _fail(conn, np["id"], "the coordinator stopped before the order was sent", now)
                    continue
                order_id = orphan["id"]
                conn.execute("UPDATE news_positions SET open_order_id = %s WHERE id = %s", (order_id, np["id"]))
            order = conn.execute("SELECT * FROM orders WHERE id = %s", (order_id,)).fetchone()
            if order["status"] in trading.OPEN_ORDER:
                continue
            if order["status"] == "blocked":
                _fail(conn, np["id"], f"Blocked: {order['error']}", now)
                continue
            held = abs(_held(conn, book["id"], np["symbol"]))
            if order["booked_qty"] <= 0 or held <= 0 or not order["filled_avg_price"]:
                _fail(conn, np["id"], order["error"] or f"Alpaca {order['status']} the order without filling it", now)
                continue
            entry = float(order["filled_avg_price"])
            move = limits.stop_loss_pct / 100.0
            stop_price = entry * (1.0 - move) if np["side"] == "long" else entry * (1.0 + move)
            conn.execute(
                "UPDATE news_positions SET status = 'open', qty = %s, entry_price = %s, stop_price = %s,"
                " exit_due_at = %s WHERE id = %s",
                (held, entry, stop_price, horizon_end(book["market"], now, np["horizon_days"]), np["id"]),
            )
        else:
            order = (conn.execute("SELECT * FROM orders WHERE id = %s", (np["close_order_id"],)).fetchone()
                     if np["close_order_id"] else None)
            if order is not None and order["status"] in trading.OPEN_ORDER:
                continue
            held = _held(conn, book["id"], np["symbol"])
            if held == 0:
                conn.execute("UPDATE news_positions SET status = 'closed', qty = 0, pnl = %s, closed_at = %s"
                             " WHERE id = %s", (_pnl(conn, np), now, np["id"]))
            else:  # the close was blocked, refused or only partly filled: try again later
                conn.execute("UPDATE news_positions SET status = 'open', qty = %s, close_reason = NULL WHERE id = %s",
                             (abs(held), np["id"]))
    conn.commit()


# ------------------------------------------------------------------ the pass


def news_pass(conn: psycopg.Connection, status: BrokerStatus, limits: Limits, price_now: PriceNow,
              now: datetime) -> int:
    """Close and open the news strategies' positions. Returns orders sent (not blocked).
    One book failing is logged and never stops the other."""
    books = conn.execute(
        """
        SELECT b.*, m.market, m.name AS model_name FROM books b JOIN models m ON m.id = b.model_id
         WHERE m.origin = 'news' AND b.status IN ('active', 'closing') ORDER BY b.id
        """
    ).fetchall()
    conn.commit()
    sent = 0
    for book in books:
        try:
            sent += _book_pass(conn, status, limits, price_now, now, book)
        except Exception:  # noqa: BLE001 - reported; the next pass tries again
            conn.rollback()
            log.exception("news trading: book %s", book["id"])
    return sent


def _book_pass(conn: psycopg.Connection, status: BrokerStatus, limits: Limits, price_now: PriceNow,
               now: datetime, book: dict[str, Any]) -> int:
    trading.poll_fills(conn, status, book_id=book["id"])
    _sync(conn, book, limits, now)
    if _in_flight(conn, book["id"]):
        return 0  # wait for the last order to settle before deciding again
    # While paused the stops and a stopped book's closing still go out (protective exits);
    # nothing else does.
    paused = safety.is_paused(conn)
    wait = None
    if book["mode"] != status.broker.mode:
        wait = f"Waiting: the coordinator is in {status.broker.mode} mode"
    elif book["market"] == "stocks" and not (status.clock_info and status.clock_info.is_open):
        wait = "Waiting for the stock market to open"
    if wait is not None:
        trading._set_waiting(conn, book["id"], "Waiting: trading is paused" if paused else wait)
        return 0

    open_rows = conn.execute(
        """
        SELECT np.*, p.decided_at, p.horizon_days FROM news_positions np
          JOIN news_predictions p ON p.id = np.prediction_id
         WHERE np.book_id = %s AND np.status = 'open' ORDER BY np.id
        """,
        (book["id"],),
    ).fetchall()
    candidates = _candidates(conn, book, limits, now) if book["status"] == "active" and not paused else []
    held_symbols = [r["symbol"] for r in conn.execute("SELECT symbol FROM positions WHERE book_id = %s",
                                                       (book["id"],)).fetchall()]
    live = _safe_prices(price_now, [r["symbol"] for r in open_rows] + [c["symbol"] for c in candidates] + held_symbols)
    # Values for the limits: the latest trade where known, else the last bar's close.
    prices = trading.latest_prices(conn, sorted(set(held_symbols) | {c["symbol"] for c in candidates}))
    prices.update(live)

    sent, note = 0, None
    result = _exits(conn, status, limits, book, open_rows, live, prices, now, protective_only=paused)
    sent += result[0]
    if result[1] is not None:
        trading._set_waiting(conn, book["id"], "Waiting: " + result[1])
        return sent
    note = result[2]
    if book["status"] == "closing":
        if not conn.execute("SELECT 1 FROM news_positions WHERE book_id = %s AND status = ANY(%s)",
                            (book["id"], list(ACTIVE))).fetchone() and trading.close_if_flat(conn, book["id"]):
            conn.commit()
            return sent
        trading._set_waiting(conn, book["id"], note or "Stopping: closing its positions")
        return sent
    if paused:
        trading._set_waiting(conn, book["id"], "Waiting: trading is paused")
        return sent
    entered, halted, entry_note = _entries(conn, status, limits, book, candidates, live, prices, now)
    sent += entered
    if halted is not None:
        trading._set_waiting(conn, book["id"], "Waiting: " + halted)
        return sent
    trading._set_waiting(conn, book["id"], note or entry_note)
    return sent


def _in_flight(conn: psycopg.Connection, book_id: int) -> bool:
    return conn.execute("SELECT 1 FROM orders WHERE book_id = %s AND status = ANY(%s)",
                        (book_id, list(trading.OPEN_ORDER))).fetchone() is not None


def _place(conn: psycopg.Connection, status: BrokerStatus, limits: Limits, book: dict[str, Any],
           order: dict[str, Any], prices: dict[str, float]) -> dict[str, Any]:
    """One order through the safety path, against the book as it is now."""
    conn.commit()
    fresh = conn.execute("SELECT * FROM books WHERE id = %s", (book["id"],)).fetchone()
    return safety.approve_and_place(conn, status, limits, fresh, order, prices)


# ------------------------------------------------------------------ exits


def _exit_reason(conn: psycopg.Connection, limits: Limits, book: dict[str, Any], np: dict[str, Any],
                 live: dict[str, float], now: datetime) -> tuple[str, bool] | None:
    """(why an open position should be closed now, in plain words, whether the exit is
    protective: a stop, or the book was stopped), or None. Protective exits go out even
    while trading is paused, so the stop is checked before the horizon."""
    if book["status"] == "closing":
        return "the news strategy was stopped", True
    price = live.get(np["symbol"])
    if price is not None and np["stop_price"] is not None:
        if np["side"] == "long" and price <= np["stop_price"]:
            return f"the price fell to {price:,.2f}, through its stop at {np['stop_price']:,.2f}", True
        if np["side"] == "short" and price >= np["stop_price"]:
            return f"the price rose to {price:,.2f}, through its stop at {np['stop_price']:,.2f}", True
    days = np["horizon_days"]
    if np["exit_due_at"] is not None and np["exit_due_at"] <= now:
        return f"its {days}-day horizon has passed", False
    against = "down" if np["side"] == "long" else "up"
    newer = conn.execute(
        """
        SELECT expected_move_pct, confidence FROM news_predictions
         WHERE source = 'live' AND market = %s AND symbol = %s AND direction = %s
           AND decided_at > %s AND decided_at <= %s AND confidence >= %s AND NOT already_priced_in
         ORDER BY decided_at DESC LIMIT 1
        """,
        (book["market"], np["symbol"], against, np["decided_at"], now, limits.min_confidence),
    ).fetchone()
    if newer is not None:
        verb = "fall" if against == "down" else "rise"
        return (f"a newer prediction expects it to {verb} {newer['expected_move_pct']:.1f}% "
                f"(confidence {newer['confidence']:.2f})"), False
    return None


def _exits(conn: psycopg.Connection, status: BrokerStatus, limits: Limits, book: dict[str, Any],
           open_rows: list[dict[str, Any]], live: dict[str, float], prices: dict[str, float],
           now: datetime, protective_only: bool = False) -> tuple[int, str | None, str | None]:
    """Close what is due (only the protective exits when `protective_only`, i.e. while
    trading is paused). Returns (orders sent, why the book must wait, a note)."""
    sent, note = 0, None
    todo: list[tuple[dict[str, Any] | None, str, str, bool]] = []  # (position, symbol, reason, protective)
    for np in open_rows:
        why = _exit_reason(conn, limits, book, np, live, now)
        if why is not None and (why[1] or not protective_only):
            todo.append((np, np["symbol"], *why))
    if book["status"] == "closing":
        # Anything the book holds that no news position accounts for is closed too, so a
        # stopped book always ends flat.
        tracked = {r["symbol"] for r in conn.execute(
            "SELECT symbol FROM news_positions WHERE book_id = %s AND status = ANY(%s)",
            (book["id"], list(ACTIVE))).fetchall()}
        for r in conn.execute("SELECT symbol FROM positions WHERE book_id = %s AND qty <> 0 ORDER BY symbol",
                              (book["id"],)).fetchall():
            if r["symbol"] not in tracked:
                todo.append((None, r["symbol"], "the news strategy was stopped", True))
    for np, symbol, reason, protective in todo:
        if np is not None and np["close_order_id"] is not None and conn.execute(
                "SELECT 1 FROM orders WHERE id = %s AND status = 'rejected'"
                " AND finished_at > now() - make_interval(secs => %s)",
                (np["close_order_id"], trading.CLOSE_RETRY_S)).fetchone():
            note = f"Alpaca refused to close {symbol}; trying again in a few minutes"
            continue
        held = _held(conn, book["id"], symbol)
        if held == 0:  # nothing left to close (sold elsewhere): the position is over
            if np is not None:
                conn.execute("UPDATE news_positions SET status = 'closed', qty = 0, pnl = %s, close_reason = %s,"
                             " closed_at = %s WHERE id = %s", (_pnl(conn, np), reason, now, np["id"]))
                conn.commit()
            continue
        side_word = "long" if held > 0 else "short"
        order = {"symbol": symbol, "side": "sell" if held > 0 else "buy", "qty": abs(held),
                 "reason": f"News: closing the {symbol} {side_word} because {reason}"}
        if protective:
            order["protective"] = True  # may go out while paused (safety.approve_and_place)
        row = _place(conn, status, limits, book, order, prices)
        sent += row["status"] != "blocked"
        if np is not None:
            conn.execute("UPDATE news_positions SET status = 'closing', close_order_id = %s, close_reason = %s"
                         " WHERE id = %s", (row["id"], reason[:500], np["id"]))
            conn.commit()
        if row["status"] == "blocked":
            _sync(conn, book, limits, now)
            if _is_halt(row["error"]):
                return sent, row["error"], note
            note = f"Could not close {symbol}: {row['error']}"
            continue
        trading.poll_fills(conn, status, book_id=book["id"])
        _sync(conn, book, limits, now)
        if _in_flight(conn, book["id"]):
            return sent, "an order is still filling", note
    return sent, None, note


# ------------------------------------------------------------------ entries


def _candidates(conn: psycopg.Connection, book: dict[str, Any], limits: Limits, now: datetime) -> list[dict[str, Any]]:
    """Live predictions of the book's market decided in the last ENTRY_WINDOW that this
    book has not acted on yet, oldest first. A prediction whose order was blocked only
    for "not now" (paused, market closed, no account reading, loss limit) may be tried
    again while it is fresh; any other outcome is final, so a pass is idempotent."""
    return conn.execute(
        """
        SELECT p.*, n.headline FROM news_predictions p JOIN news n ON n.id = p.news_id
         WHERE p.source = 'live' AND p.market = %s AND p.decided_at > %s AND p.decided_at <= %s
           AND NOT EXISTS (
               SELECT 1 FROM news_positions x WHERE x.prediction_id = p.id AND x.book_id = %s
                  AND NOT (x.status = 'failed' AND x.close_reason IS NOT NULL
                           AND (substr(x.close_reason, 10) = ANY(%s) OR x.close_reason LIKE 'Blocked: Daily loss%%')))
         ORDER BY p.decided_at, p.id
        """,
        (book["market"], now - ENTRY_WINDOW, now, book["id"], list(safety.HALTS)),
    ).fetchall()


def _skip_reason(p: dict[str, Any], market: str, limits: Limits) -> str | None:
    """Why a prediction is not traded at all, or None. The very predicate the gate counts
    by (news_scoring.skip_reason), so the gate measures exactly what is traded."""
    from coordinator import news_scoring

    return news_scoring.skip_reason(p, market, limits)


def _entry_reason(p: dict[str, Any]) -> str:
    verb = "rise" if p["direction"] == "up" else "fall"
    days = p["horizon_days"]
    headline = " ".join(str(p["headline"]).split())
    if len(headline) > HEADLINE_CHARS:
        headline = headline[:HEADLINE_CHARS - 1].rstrip() + "…"
    return (f"News: Claude expects {p['symbol']} to {verb} {p['expected_move_pct']:.1f}% within {days} "
            f"day{'s' if days != 1 else ''} (confidence {p['confidence']:.2f}): {headline}")


def _size(conn: psycopg.Connection, limits: Limits, book_id: int, side: str, price: float,
          prices: dict[str, float]) -> dict[str, float] | None:
    """{"notional": dollars} for a long or {"qty": whole shares} for a short, within the
    book's free cash and every cap; None when there is no room."""
    fresh = conn.execute("SELECT cash FROM books WHERE id = %s", (book_id,)).fetchone()
    ex = safety.exposure(conn, book_id, prices)
    gross_room = limits.max_per_model - ex["gross"]
    if side == "long":
        free_cash = fresh["cash"] - safety.pending_buys(conn, book_id)
        dollars = math.floor(min(limits.max_per_position, free_cash, gross_room) * 100.0) / 100.0
        return {"notional": dollars} if dollars >= MIN_TRADE_DOLLARS else None
    short_room = limits.max_short_per_model - ex["short"]
    shares = math.floor(min(limits.max_per_short_position, short_room, gross_room) / price)
    return {"qty": float(shares)} if shares >= 1 else None


def _entries(conn: psycopg.Connection, status: BrokerStatus, limits: Limits, book: dict[str, Any],
             candidates: list[dict[str, Any]], live: dict[str, float], prices: dict[str, float],
             now: datetime) -> tuple[int, str | None, str | None]:
    """Open positions for fresh predictions. Returns (orders sent, why the book must
    wait, a note for the dashboard)."""
    sent, note = 0, None
    market = book["market"]
    trades_today = _trades_today(conn, book["id"], now)
    for p in candidates:
        if _skip_reason(p, market, limits):
            continue
        side = "long" if p["direction"] == "up" else "short"
        current = conn.execute("SELECT * FROM news_positions WHERE book_id = %s AND symbol = %s AND status = ANY(%s)",
                               (book["id"], p["symbol"], list(ACTIVE))).fetchone()
        if current is not None:
            if current["side"] == side and current["status"] == "open" and current["exit_due_at"] is not None:
                due = horizon_end(market, p["decided_at"], p["horizon_days"])
                if due > current["exit_due_at"]:  # the same view again: hold it longer
                    conn.execute("UPDATE news_positions SET exit_due_at = %s WHERE id = %s", (due, current["id"]))
                    conn.commit()
            continue
        if side == "short" and market == "crypto":
            continue  # Alpaca cannot short coins; a held coin was already sold by the exits
        if trades_today >= limits.max_trades_per_day:
            note = f"Done for today: {trades_today} new news positions, the most allowed per day"
            break
        price = live.get(p["symbol"])
        if price is None:
            note = f"No current price for {p['symbol']}; its prediction waits"
            continue
        size = _size(conn, limits, book["id"], side, price, prices)
        if size is None:
            note = f"No room left within the limits for {p['symbol']}"
            continue
        np = conn.execute(
            "INSERT INTO news_positions (book_id, prediction_id, symbol, side, opened_at) VALUES (%s, %s, %s, %s, %s)"
            " RETURNING id",
            (book["id"], p["id"], p["symbol"], side, now),
        ).fetchone()
        conn.commit()
        order = {"symbol": p["symbol"], "side": "buy" if side == "long" else "sell", **size,
                 "reason": _entry_reason(p)}
        row = _place(conn, status, limits, book, order, prices)
        conn.execute("UPDATE news_positions SET open_order_id = %s WHERE id = %s", (row["id"], np["id"]))
        conn.commit()
        sent += row["status"] != "blocked"
        if row["status"] == "blocked":
            _sync(conn, book, limits, now)
            if _is_halt(row["error"]):
                return sent, row["error"], note
            continue
        trading.poll_fills(conn, status, book_id=book["id"])
        _sync(conn, book, limits, now)
        if conn.execute("SELECT 1 FROM news_positions WHERE id = %s AND status <> 'failed'", (np["id"],)).fetchone():
            trades_today += 1
        if _in_flight(conn, book["id"]):
            return sent, "an order is still filling", note
    return sent, None, note


def _trades_today(conn: psycopg.Connection, book_id: int, now: datetime) -> int:
    """New news positions of the book this UTC day (failed attempts do not count)."""
    return int(conn.execute(
        "SELECT count(*) AS n FROM news_positions WHERE book_id = %s AND opened_at >= %s AND status <> 'failed'",
        (book_id, _day_start(now))).fetchone()["n"])


# ------------------------------------------------------------------ dashboard


def open_positions(conn: psycopg.Connection, market: str | None = None,
                   prices: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """The news positions still opening, open or closing, newest first, with the market,
    the headline and the prediction behind each, and (when a price is known: `prices`,
    else the last bar's close) "price", "value" (signed: a short is negative) and
    "open_pnl"."""
    rows = conn.execute(
        """
        SELECT np.*, m.market, b.model_id, n.headline, n.url, p.direction, p.expected_move_pct, p.confidence,
               p.horizon_days, p.decided_at
          FROM news_positions np JOIN books b ON b.id = np.book_id JOIN models m ON m.id = b.model_id
          JOIN news_predictions p ON p.id = np.prediction_id JOIN news n ON n.id = p.news_id
         WHERE np.status = ANY(%s) AND (%s::text IS NULL OR m.market = %s)
         ORDER BY np.opened_at DESC, np.id DESC
        """,
        (list(ACTIVE), market, market),
    ).fetchall()
    known = trading.latest_prices(conn, sorted({r["symbol"] for r in rows}))
    known.update(prices or {})
    out = []
    for r in rows:
        r = dict(r)
        price = known.get(r["symbol"])
        sign = 1.0 if r["side"] == "long" else -1.0
        r["price"] = price
        r["value"] = sign * r["qty"] * price if price is not None else None
        r["open_pnl"] = (sign * r["qty"] * (price - r["entry_price"])
                         if price is not None and r["entry_price"] is not None else None)
        out.append(r)
    return out


def book_summary(conn: psycopg.Connection, prices: dict[str, float] | None = None,
                 now: datetime | None = None) -> dict[str, dict[str, Any]]:
    """{"stocks": {...}, "crypto": {...}} for the dashboard: model_id, status
    ("not_started" | "active" | "closing" | "closed"), book_id, cash, equity,
    starting_balance, gross_exposure, long_exposure, short_exposure (positive),
    open_positions, trades_today, waiting. Values use `prices` where given, else the
    last bar's close."""
    now = now or datetime.now(timezone.utc)
    out: dict[str, dict[str, Any]] = {}
    for market, model_id in NEWS_MODELS.items():
        summary: dict[str, Any] = {
            "model_id": model_id, "market": market, "status": "not_started", "book_id": None, "cash": None,
            "equity": None, "starting_balance": None, "gross_exposure": 0.0, "long_exposure": 0.0,
            "short_exposure": 0.0, "open_positions": 0, "trades_today": 0, "waiting": None,
        }
        book = conn.execute(
            "SELECT * FROM books WHERE model_id = %s ORDER BY (status IN ('active', 'closing')) DESC, id DESC LIMIT 1",
            (model_id,),
        ).fetchone()
        if book is not None:
            held = conn.execute("SELECT symbol, qty FROM positions WHERE book_id = %s", (book["id"],)).fetchall()
            known = trading.latest_prices(conn, [h["symbol"] for h in held])
            known.update(prices or {})
            values = [h["qty"] * known.get(h["symbol"], 0.0) for h in held]
            summary.update(
                status=book["status"], book_id=book["id"], cash=book["cash"], waiting=book["waiting"],
                starting_balance=book["starting_balance"], equity=book["cash"] + sum(values),
                long_exposure=sum(v for v in values if v > 0), short_exposure=-sum(v for v in values if v < 0),
                gross_exposure=sum(abs(v) for v in values),
                open_positions=int(conn.execute(
                    "SELECT count(*) AS n FROM news_positions WHERE book_id = %s AND status = ANY(%s)",
                    (book["id"], list(ACTIVE))).fetchone()["n"]),
                trades_today=_trades_today(conn, book["id"], now),
            )
        out[market] = summary
    return out
