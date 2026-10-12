"""Safety controls: the pause switch, the daily loss limit, the money limits, and
approve_and_place(), the one path every order takes to Alpaca.

- Pause all trading: one switch. While it is on, no order is placed for any model
  (except a news book's protective reducing orders, below); backtests and model
  searches keep running. Only the owner resumes it.
- Daily loss limit: when the account is down more than daily_loss_limit_pct today
  (config/limits.toml, default 2%), trading pauses itself and says why.
- Reducing orders (the futures precedent): an order that only reduces a position the
  book holds (a sell of at most the long held, a buy by qty of at most the shares
  short) skips the daily loss check, for any book: it can only lower risk. While
  paused, a reducing order of a news book still goes out when it is protective
  (order["protective"]: a stop-loss exit, or closing a book the owner stopped);
  every other order waits for the owner to resume. The mode, the market clock and
  the account reading still apply to every order, and the caps to opening orders.
- Per-model limits: a buy never takes a model past max_per_position in one symbol or
  max_per_model in total, and never spends more cash than the model's book holds.
- Every order is written to the orders table (model, worker, time, reason) before it
  is sent, and also when it is blocked, with the reason it was blocked.

Shorts (docs/NEWS_PLAN.md, option B: every book shares one Alpaca account, which holds
one net position per symbol). For stock and crypto books an order means:
- sell by qty: reduces a long the book holds (never by more than it holds: a sale may
  not cross from long to short in one order); with nothing held, or while short, it
  opens or extends a short of qty.
- buy by notional (dollars): opens or extends a long; refused while the book is short
  the symbol (cover the short first).
- buy by qty: only covers a short, by at most the shares short.
A short is allowed only when, after the checks above and the existing money limits:
the book's model has origin 'news'; the symbol is not crypto (Alpaca cannot short
coins) and not SPY or QQQ; qty is whole shares (1 or more); Alpaca says the asset is
shortable and easy to borrow (no answer means no short); the position stays within
max_per_short_position, all of the book's shorts within max_short_per_model, and the
book's gross exposure (longs + |shorts| + orders in flight) within max_per_model, each
with the CAP_TOLERANCE band.
Across books: a short of S is refused while any other open book holds S long or has a
buy of S in flight; a buy of S that opens or extends a long (any model) is refused
while another open book is short S or has a short sale of S in flight. So whenever a
news book is short S, the account's position in S is exactly that short. The news
strategies never trade SPY or QQQ at all, which keeps them clear of the futures paper
books (they trade SPY/QQQ long and short on the same account from their own tables).
These checks run inside approve_and_place under the PAUSE_LOCK advisory lock, which
every order holds until its row is written, so two books cannot race past them.
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
OPEN = ("submitting", "submitted", "partially_filled")
# The futures paper books trade these on the same Alpaca account: the news strategies
# never touch them (coordinator.news_trading.NEWS_EXCLUDED is the same pair).
NEWS_EXCLUDED = ("SPY", "QQQ")


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
    """(net invested dollars, {symbol: position value}) of a book at the given prices; a
    short counts negative. The limits use exposure(), which counts shorts as exposure."""
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


def _pending_shorts(conn: psycopg.Connection, book_id: int) -> dict[str, float]:
    """{symbol: shares} of the book's sales in flight that open or extend a short (the
    book holds nothing or is short there); the unbooked part only."""
    rows = conn.execute(
        """
        SELECT o.symbol, sum(COALESCE(o.qty, 0) - o.booked_qty) AS q FROM orders o
          LEFT JOIN positions p ON p.book_id = o.book_id AND p.symbol = o.symbol
         WHERE o.book_id = %s AND o.side = 'sell' AND o.status = ANY(%s) AND COALESCE(p.qty, 0) <= 0
         GROUP BY o.symbol
        """,
        (book_id, list(OPEN)),
    ).fetchall()
    return {r["symbol"]: max(0.0, float(r["q"])) for r in rows}


def exposure(conn: psycopg.Connection, book_id: int, prices: dict[str, float]) -> dict[str, Any]:
    """The book's exposure at the given prices: {"long", "short" (a positive number),
    "gross" (long + short + buys and short sales in flight), "values" {symbol: signed
    value}, "missing" [held symbols with no price]}."""
    rows = conn.execute("SELECT symbol, qty FROM positions WHERE book_id = %s", (book_id,)).fetchall()
    values = {r["symbol"]: r["qty"] * prices.get(r["symbol"], 0.0) for r in rows}
    missing = [r["symbol"] for r in rows if r["qty"] != 0 and not prices.get(r["symbol"])]
    long = sum(v for v in values.values() if v > 0)
    short = -sum(v for v in values.values() if v < 0)
    in_flight = _pending_shorts(conn, book_id)
    pending_short = sum(q * prices.get(sym, 0.0) for sym, q in in_flight.items())
    missing += [sym for sym in in_flight if not prices.get(sym)]
    return {"long": long, "short": short + pending_short, "values": values, "missing": sorted(set(missing)),
            "gross": long + short + pending_short + pending_buys(conn, book_id)}


def _origin(conn: psycopg.Connection, model_id: str) -> str | None:
    row = conn.execute("SELECT origin FROM models WHERE id = %s", (model_id,)).fetchone()
    return row["origin"] if row else None


def is_reducing(conn: psycopg.Connection, book: dict[str, Any], order: dict[str, Any]) -> bool:
    """True when the order only reduces a position the book holds: a sell of at most
    the long held, or a buy by qty (a cover) of at most the shares short."""
    row = conn.execute("SELECT qty FROM positions WHERE book_id = %s AND symbol = %s",
                       (book["id"], order["symbol"])).fetchone()
    held = float(row["qty"]) if row else 0.0
    qty = float(order.get("qty") or 0.0)
    if qty <= 0:
        return False
    if order["side"] == "sell":
        return held > 0 and qty <= held * (1 + 1e-9)
    return order.get("notional") is None and held < 0 and qty <= -held * (1 + 1e-9)


def limit_problem(conn: psycopg.Connection, book: dict[str, Any], order: dict[str, Any], prices: dict[str, float],
                  limits: Limits, broker: Any = None) -> str | None:
    """Why the order breaks a money limit or a short rule, or None (module docstring).

    `broker` answers whether a symbol can be shorted; without one no short is allowed."""
    symbol = order["symbol"]
    news = _origin(conn, book["model_id"]) == "news"
    if news and symbol in NEWS_EXCLUDED:
        return (f"the news strategies never trade {symbol}: the futures paper books trade SPY and QQQ "
                "on the same Alpaca account")
    row = conn.execute("SELECT qty FROM positions WHERE book_id = %s AND symbol = %s", (book["id"], symbol)).fetchone()
    held = float(row["qty"]) if row else 0.0
    if order["side"] == "sell":
        qty = float(order.get("qty") or 0.0)
        if qty <= 0:
            return "a sale must be for more than 0 shares"
        if held > 0:
            if qty > held * (1 + 1e-9):
                return (f"sell of {qty:.6g} is more than the {held:.6g} held: it would cross from long to short "
                        "in one order")
            return None
        if not news:
            return f"nothing to sell: the model holds no {symbol}"
        return _short_problem(conn, book, symbol, qty, held, prices, limits, broker)
    if order.get("notional") is None:
        return _cover_problem(symbol, float(order.get("qty") or 0.0), held)
    if order.get("qty") is not None:
        return "a buy is by dollars or by shares, not both"
    dollars = float(order["notional"])
    if dollars <= 0:
        return "a buy must be for more than $0"
    if held < 0:
        return f"cover the short first: the model is short {-held:g} {symbol}"
    free_cash = book["cash"] - pending_buys(conn, book["id"])
    if dollars > free_cash + 0.01:
        return f"not enough cash in the model's book (${free_cash:,.2f} free)"
    ex = exposure(conn, book["id"], prices)
    if news and ex["missing"]:
        return f"no current price for {', '.join(ex['missing'])}, so the model's exposure cannot be checked"
    if ex["values"].get(symbol, 0.0) + dollars > limits.max_per_position + CAP_TOLERANCE:
        return f"over the ${limits.max_per_position:,.0f} limit per position"
    if ex["gross"] + dollars > limits.max_per_model + CAP_TOLERANCE:
        return f"over the ${limits.max_per_model:,.0f} limit per model"
    other = conn.execute(
        """
        SELECT m.name FROM positions p JOIN books b ON b.id = p.book_id JOIN models m ON m.id = b.model_id
         WHERE p.symbol = %s AND p.qty < 0 AND b.id <> %s AND b.status IN ('active', 'closing') LIMIT 1
        """,
        (symbol, book["id"]),
    ).fetchone()
    if other is not None:
        return (f"{other['name']} is short {symbol}, and Alpaca keeps one position per symbol: "
                "no model buys it until that short is covered")
    shorting = conn.execute(
        """
        SELECT m.name FROM orders o JOIN books b ON b.id = o.book_id JOIN models m ON m.id = b.model_id
          LEFT JOIN positions p ON p.book_id = o.book_id AND p.symbol = o.symbol
         WHERE o.symbol = %s AND o.side = 'sell' AND o.status = ANY(%s) AND o.book_id <> %s
           AND b.status IN ('active', 'closing') AND COALESCE(p.qty, 0) <= 0 LIMIT 1
        """,
        (symbol, list(OPEN), book["id"]),
    ).fetchone()
    if shorting is not None:
        return f"{shorting['name']} has a short sale of {symbol} in flight; no model buys it until that settles"
    return None


def _cover_problem(symbol: str, qty: float, held: float) -> str | None:
    """A buy by shares only covers a short, by at most what is short."""
    if qty <= 0:
        return "a buy must be for more than $0"
    if held >= 0:
        return f"a buy by shares only covers a short, and the model is not short {symbol}"
    if qty > -held * (1 + 1e-9):
        return f"buy of {qty:.6g} is more than the {-held:.6g} {symbol} short"
    return None


def _short_problem(conn: psycopg.Connection, book: dict[str, Any], symbol: str, qty: float, held: float,
                   prices: dict[str, float], limits: Limits, broker: Any) -> str | None:
    """Why a news book may not open or extend a short of `qty` shares of `symbol` now."""
    if "/" in symbol:
        return "Alpaca cannot short crypto"
    if symbol in NEWS_EXCLUDED:
        return f"{symbol} is never shorted (the futures paper books trade it on the same account)"
    if qty < 1 or abs(qty - round(qty)) > 1e-9:
        return f"shorts are whole shares only, not {qty:.6g}"
    price = prices.get(symbol)
    if not price or price <= 0:
        return f"no current price for {symbol}, so the short cannot be checked against the limits"
    ex = exposure(conn, book["id"], prices)
    if ex["missing"]:
        return f"no current price for {', '.join(ex['missing'])}, so the model's exposure cannot be checked"
    value = qty * price
    pending_here = _pending_shorts(conn, book["id"]).get(symbol, 0.0) * price
    if -held * price + pending_here + value > limits.max_per_short_position + CAP_TOLERANCE:
        return f"over the ${limits.max_per_short_position:,.0f} limit per short position"
    if ex["short"] + value > limits.max_short_per_model + CAP_TOLERANCE:
        return f"over the ${limits.max_short_per_model:,.0f} limit for all of a model's shorts"
    if ex["gross"] + value > limits.max_per_model + CAP_TOLERANCE:
        return f"over the ${limits.max_per_model:,.0f} limit per model (longs and shorts together)"
    # Alpaca keeps one net position per symbol: a short must not eat into another
    # book's long, nor be partly covered by another book's buy in flight.
    other = conn.execute(
        """
        SELECT m.name FROM positions p JOIN books b ON b.id = p.book_id JOIN models m ON m.id = b.model_id
         WHERE p.symbol = %s AND p.qty > 0 AND b.id <> %s AND b.status IN ('active', 'closing') LIMIT 1
        """,
        (symbol, book["id"]),
    ).fetchone()
    if other is not None:
        return f"{other['name']} holds {symbol}, and Alpaca keeps one position per symbol: no short until it is sold"
    buying = conn.execute(
        """
        SELECT m.name FROM orders o JOIN books b ON b.id = o.book_id JOIN models m ON m.id = b.model_id
         WHERE o.symbol = %s AND o.side = 'buy' AND o.status = ANY(%s) AND o.book_id <> %s
           AND b.status IN ('active', 'closing') LIMIT 1
        """,
        (symbol, list(OPEN), book["id"]),
    ).fetchone()
    if buying is not None:
        return f"{buying['name']} has a buy of {symbol} in flight; no short until it settles"
    if broker is None:
        return f"could not check with Alpaca that {symbol} can be shorted"
    try:
        asset = broker.asset_info(symbol)
    except BrokerUnavailable as exc:
        return f"could not check with Alpaca that {symbol} can be shorted ({exc})"
    if not asset.shortable:
        return f"Alpaca says {symbol} cannot be shorted"
    if not asset.easy_to_borrow:
        return f"{symbol} is hard to borrow at Alpaca, so it is not shorted"
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

    order: {"symbol", "side": "buy"|"sell", "notional" (buys) | "qty" (sells, and buys
    that cover a short), "reason", "worker_id", optionally "protective" (news books
    only: a stop-loss exit or the closing of a stopped book)}. Checks, in order: the
    pause switch (a protective reducing order of a news book passes it), the broker's
    mode against the book's, a fresh account reading and the daily loss limit (no
    reading means no order: it fails closed; an order that only reduces a position
    skips the loss check, see is_reducing), the stock market being open (no clock means
    closed), then the money limits and the short rules (limit_problem). The row is
    committed as "submitting" before Alpaca is called, so an order can never reach
    Alpaca without a record of it. The pause lock is held until Alpaca has answered, so
    once pause_trading() returns no order is in flight.
    """
    broker = status.broker
    conn.execute("SELECT pg_advisory_lock(%s)", (PAUSE_LOCK,))
    try:
        problem = None
        reducing = is_reducing(conn, book, order)
        if is_paused(conn) and not (reducing and order.get("protective")
                                    and _origin(conn, book["model_id"]) == "news"):
            problem = PAUSED
        elif broker.mode != book["mode"]:
            problem = f"this model trades {book['mode']} money but the coordinator is in {broker.mode} mode"
        if problem is None:
            status.refresh(force=status.age() > STATUS_MAX_AGE_S)
            account = status.account_info
            if account is None or account.last_equity <= 0:
                problem = NO_ACCOUNT
            elif not reducing:
                problem = check_daily_loss(conn, status, limits)
        if problem is None and "/" not in order["symbol"]:
            if status.clock_info is None:
                problem = NO_CLOCK
            elif not status.clock_info.is_open:
                problem = MARKET_CLOSED
        if problem is None:
            problem = limit_problem(conn, book, order, prices, limits, broker)
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


# ------------------------------------------------------------------ futures (Alpaca paper and Topstep)
#
# Futures models trade through approve_futures_order(), the futures twin of
# approve_and_place(). On top of the pause switch:
# - Alpaca paper only: a futures model never trades the Alpaca account in live mode;
#   the account's daily loss limit and market hours apply as for stocks; a model holds
#   at most futures_paper_max_contracts and all of them together at most
#   futures_paper_max_dollars (config/limits.toml).
# - Topstep (that account only): trading stops once stop_at_loss_share (80%) of the loss
#   limit is used; a model holds at most its own size and all of them together at most
#   Topstep's contract limit (config/topstep.toml).
# - Both: nothing new is opened near the close, and everything is closed by
#   flat_margin_minutes minutes before 15:00 Chicago time. Closing a position to
#   be flat by then is allowed even while trading is paused, so an account is never
#   left holding a position overnight.

TOPSTEP_PAUSED = "Topstep trading is paused"


def topstep_paused(conn: psycopg.Connection) -> str | None:
    if get_setting(conn, "topstep_paused", False) is True:
        return get_setting(conn, "topstep_paused_reason", None) or TOPSTEP_PAUSED
    return None


def pause_topstep(conn: psycopg.Connection, actor: str | None, reason: str) -> bool:
    conn.execute("SELECT pg_advisory_xact_lock(%s)", (PAUSE_LOCK,))
    if topstep_paused(conn):
        return False
    put_setting(conn, "topstep_paused", True, actor, "topstep_paused")
    put_setting(conn, "topstep_paused_reason", reason, actor, "topstep_paused_reason")
    log.warning("Topstep trading paused by %s: %s", actor, reason)
    return True


def resume_topstep(conn: psycopg.Connection, actor: str | None) -> bool:
    conn.execute("SELECT pg_advisory_xact_lock(%s)", (PAUSE_LOCK,))
    if not topstep_paused(conn):
        return False
    put_setting(conn, "topstep_paused", False, actor, "topstep_resumed")
    put_setting(conn, "topstep_paused_reason", None, actor, "topstep_paused_reason")
    return True


def topstep_floor(conn: psycopg.Connection, account: str, rules: Any) -> float:
    """Topstep's loss floor as the coordinator counts it: the starting balance minus the
    loss limit, raised to (highest end-of-day balance - loss limit), never above the
    starting balance (fleet2.sim.topstep has the same rule for backtests)."""
    start = float(rules.account_start_balance)
    high = conn.execute("SELECT max(balance) AS b FROM topstep_days WHERE account = %s", (account,)).fetchone()["b"]
    high = max(start, float(high)) if high is not None else start
    return min(max(start - rules.max_loss_limit, high - rules.max_loss_limit), start)


def check_topstep_loss(conn: psycopg.Connection, account: str, equity: float, rules: Any) -> str | None:
    """Pause Topstep trading once stop_at_loss_share of the loss limit is used up.
    `equity` is the balance with open losses counted (open gains are not)."""
    floor = topstep_floor(conn, account, rules)
    room = equity - floor
    allowed = (1.0 - rules.stop_at_loss_share) * rules.max_loss_limit
    if room > allowed:
        return None
    reason = (f"Topstep loss limit: ${room:,.0f} left above the loss floor of ${floor:,.0f}, under the "
              f"${allowed:,.0f} safety margin ({rules.stop_at_loss_share * 100:.0f}% of the limit used)")
    pause_topstep(conn, SYSTEM, reason)
    return reason


def _futures_insert(conn: psycopg.Connection, book: dict[str, Any], order: dict[str, Any], status: str,
                    error: str | None) -> dict[str, Any]:
    return conn.execute(
        """
        INSERT INTO futures_orders (book_id, model_id, worker_id, venue, symbol, instrument, side, qty, contracts, reason,
                                    status, error, finished_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CASE WHEN %s = 'blocked' THEN now() END) RETURNING *
        """,
        (book["id"], book["model_id"], order.get("worker_id"), book["venue"], order["symbol"], order["instrument"],
         order["side"], order["qty"], order["contracts"], order["reason"], status, error, status),
    ).fetchone()


def futures_problem(conn: psycopg.Connection, venues: Any, limits: Limits, rules: Any, book: dict[str, Any],
                    order: dict[str, Any]) -> str | None:
    """Why a futures order may not go out now, or None."""
    reducing = bool(order.get("reducing"))
    if is_paused(conn) and not (reducing and order.get("closing_time")):
        return PAUSED
    after = float(order["after_contracts"])
    if book["venue"] == "alpaca_paper":
        status = venues.alpaca
        broker = status.broker
        if not broker.connected:
            return broker.problem or "Alpaca is not connected"
        if broker.mode != "paper":
            return "futures models only paper trade on Alpaca, and the coordinator is in live mode"
        status.refresh(force=status.age() > STATUS_MAX_AGE_S)
        account = status.account_info
        if account is None or account.last_equity <= 0:
            return NO_ACCOUNT
        if not reducing:
            loss = check_daily_loss(conn, status, limits)
            if loss:
                return loss
        if status.clock_info is None:
            return NO_CLOCK
        if not status.clock_info.is_open:
            return MARKET_CLOSED
        if not reducing:
            if abs(after) > limits.futures_paper_max_contracts + 1e-9:
                return f"over the {limits.futures_paper_max_contracts:g} contract limit per futures paper model"
            exposure = float(order.get("exposure_after") or 0.0)
            if exposure > limits.futures_paper_max_dollars:
                return f"over the ${limits.futures_paper_max_dollars:,.0f} limit for all futures paper models"
        return None
    link = venues.topstep
    if not link.on:
        return link.problem or "Topstep is not connected"
    if not reducing:
        stopped = topstep_paused(conn)
        if stopped:
            return stopped
        if abs(after) > int(book["contracts"]):
            return f"over this model's size of {book['contracts']} contracts"
        if float(order.get("account_contracts_after") or 0.0) > rules.max_micro_contracts:
            return f"over Topstep's limit of {rules.max_micro_contracts} micro contracts"
    return None


def approve_futures_order(conn: psycopg.Connection, venues: Any, limits: Limits, rules: Any, book: dict[str, Any],
                          order: dict[str, Any]) -> dict[str, Any]:
    """Check one futures order and, when it passes, send it. Always logs a row, which is
    committed as "submitting" before the broker is called (as for stocks), under the
    pause lock.

    order: {"symbol", "instrument", "side", "qty" (instrument units), "contracts"
    (signed, in micros), "after_contracts" (the model's position after it), "reducing",
    "closing_time", "exposure_after" (Alpaca), "account_contracts_after" (Topstep),
    "reason", "worker_id"}."""
    conn.execute("SELECT pg_advisory_lock(%s)", (PAUSE_LOCK,))
    try:
        problem = futures_problem(conn, venues, limits, rules, book, order)
        if problem is not None:
            row = _futures_insert(conn, book, order, "blocked", problem)
            conn.commit()
            return row
        row = _futures_insert(conn, book, order, "submitting", None)
        conn.commit()
        try:
            if book["venue"] == "alpaca_paper":
                state = venues.alpaca.broker.place_order(str(row["id"]), order["instrument"], order["side"],
                                                         qty=float(order["qty"]))
            else:
                state = venues.topstep.client.place_order(str(row["id"]), order["instrument"], order["side"],
                                                          int(order["qty"]))
        except Exception as exc:  # noqa: BLE001 - no clear answer: the fill poller asks again
            state = OrderState("submitted", error=f"no clear answer ({exc}); checking")
        if state.status == "rejected":
            row = conn.execute("UPDATE futures_orders SET status = 'rejected', error = %s, finished_at = now() "
                               "WHERE id = %s RETURNING *", (state.error, row["id"])).fetchone()
        else:
            row = conn.execute("UPDATE futures_orders SET status = 'submitted', broker_order_id = %s, error = %s, "
                               "submitted_at = now() WHERE id = %s RETURNING *",
                               (state.broker_order_id, state.error, row["id"])).fetchone()
        conn.commit()
        return row
    finally:
        conn.rollback()
        conn.execute("SELECT pg_advisory_unlock(%s)", (PAUSE_LOCK,))
        conn.commit()
