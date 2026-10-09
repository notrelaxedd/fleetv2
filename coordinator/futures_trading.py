"""Futures models trading: first on Alpaca paper, then on Topstep.

Two places a futures model can trade (a "venue"):
- alpaca_paper: the Alpaca paper account. Alpaca has no futures, so each micro
  contract is traded as its share equivalent: 1 MES = 50 SPY shares (S&P 500 = about
  10 x SPY, $5 a point), 1 MNQ = 82 QQQ shares (Nasdaq-100 = about 41 x QQQ, $2 a
  point). A SPY dollar is then worth what an MES point-dollar is, so results read in
  futures dollars. Long and short. Never in live mode.
- topstep: real micro contracts in the TopstepX account, only for a model whose "ready
  for a Combine" checklist is complete (which needs 20 days on Alpaca paper first).

How it runs: the model's worker job (fleet2/worker/futures_live_job.py) replays today
through the backtester on the newest closed minute and sends the contracts it wants
(receive_signal). The executor (execute_pass, every few seconds) moves each book to
that position with market orders through safety.approve_futures_order(), and on its own
authority: closes everything near 15:00 Chicago time, opens nothing new in the last
minutes, and closes a book whose worker stopped sending decisions. Fills are booked into
futures_positions and each day's result into futures_days, with the commission per
contract counted (on Alpaca paper too, where it is not charged) so the days compare
with the backtest.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import psycopg
from psycopg.types.json import Jsonb

from coordinator import futures_data, futures_live, queue, safety
from coordinator.broker import NEVER_REACHED, BrokerStatus
from coordinator.errors import BadRequest, Conflict, NotFound
from coordinator.limits import Limits
from coordinator.topstep_broker import TopstepLink
from fleet2.models.futures import get_module
from fleet2.sim import cme_session, topstep
from fleet2.universe import CONTRACTS

log = logging.getLogger(__name__)

VENUES = ("alpaca_paper", "topstep")
VENUE_TEXT = {"alpaca_paper": "Alpaca paper", "topstep": "Topstep"}
OPEN_ORDER = ("submitting", "submitted", "partially_filled")
STALE_SUBMITTING_S = 60
CUTOFF_MINUTES = 10


@dataclass
class Venues:
    """Everything the futures executor talks to."""

    alpaca: BrokerStatus
    topstep: TopstepLink
    sources: dict[str, Any] = field(default_factory=dict)
    fake: bool = False
    topstep_state: dict[str, Any] | None = None  # the latest account check, for the dashboard


def units_per_contract(venue: str, symbol: str) -> float:
    spec = CONTRACTS[symbol]
    return spec["point_value"] * spec["proxy_scale"] if venue == "alpaca_paper" else 1.0


def dollars_per_point(venue: str, symbol: str) -> float:
    """Dollars per unit per one-point move of the traded instrument."""
    return 1.0 if venue == "alpaca_paper" else CONTRACTS[symbol]["point_value"]


def instrument(venues: Venues, venue: str, symbol: str) -> str:
    if venue == "alpaca_paper":
        return CONTRACTS[symbol]["proxy"]
    return venues.topstep.client.contract(symbol)


def trading_day(now: datetime) -> Any:
    return now.astimezone(cme_session.CHICAGO).date()


# ------------------------------------------------------------------ start, stop, signals


def open_book(conn: psycopg.Connection, model_id: str, venue: str) -> dict[str, Any] | None:
    return conn.execute("SELECT * FROM futures_books WHERE model_id = %s AND venue = %s AND status IN "
                        "('active', 'closing')", (model_id, venue)).fetchone()


def start(conn: psycopg.Connection, model_id: str, venue: str, venues: Venues, rules: topstep.Rules, limits: Limits,
          target: str = "auto", ready: bool = False) -> dict[str, Any]:
    """Open a book for the model at a venue and give a worker its live job. `ready` is the
    model's "ready for a Combine" verdict (required for Topstep)."""
    if venue not in VENUES:
        raise BadRequest(f"unknown place to trade {venue!r}")
    model = conn.execute("SELECT * FROM models WHERE id = %s", (model_id,)).fetchone()
    if model is None:
        raise NotFound(f"no model {model_id!r}")
    if model["market"] != "futures":
        raise BadRequest("This is for futures models")
    if model["status"] == "retired":
        raise Conflict("This model is retired")
    held = (model["metrics"] or {}).get("held_out")
    if not held:
        raise Conflict("Run a backtest first, so you know what you are trading")
    missing = rules.missing_trading_fees()
    if missing:
        raise Conflict(f"{topstep.FEE_MESSAGE} ({', '.join(missing)}): live results count the commission")
    existing = open_book(conn, model_id, venue)
    if existing is not None:
        if existing["status"] == "closing":
            raise Conflict(f"{model['name']} is still closing its position on {VENUE_TEXT[venue]}: try again in a moment")
        raise Conflict(f"{model['name']} is already trading on {VENUE_TEXT[venue]}")
    best = topstep.best_size(held, rules)
    size = int(best["contracts"]) if best else 1
    if venue == "alpaca_paper":
        broker = venues.alpaca.broker
        if not broker.connected:
            raise Conflict(broker.problem or "Alpaca is not connected")
        if broker.mode != "paper":
            raise Conflict("Futures models only paper trade on Alpaca, and the coordinator is in live mode")
        size = max(1, min(size, int(limits.futures_paper_max_contracts)))
    else:
        if not venues.topstep.on:
            raise Conflict(venues.topstep.problem or "Topstep is not connected")
        if not ready:
            raise Conflict(f"{model['name']} is not ready for a Combine: every line of its checklist must be ticked "
                           "before it trades on Topstep")
        if safety.topstep_paused(conn):
            raise Conflict(safety.topstep_paused(conn))
        size = max(1, min(size, rules.max_micro_contracts))
    account = venues.topstep.client.account().id if venue == "topstep" else None
    book = conn.execute("INSERT INTO futures_books (model_id, venue, contracts, account) VALUES (%s, %s, %s, %s) "
                        "RETURNING *", (model_id, venue, size, account)).fetchone()
    source = futures_live.source_for(venue, venues.fake)
    params = {"model_id": model_id, "module": model["module"], "params": model["params"], "market": "futures",
              "venue": venue, "source": source, "contracts": size, "rules": rules.to_dict()}
    result = queue.create_job(conn, "paper_trade", params, target, model_id)
    conn.execute("UPDATE futures_books SET job_id = %s WHERE id = %s", (result.jobs[0]["id"], book["id"]))
    where = "a worker" if result.waiting else "its worker"
    units = units_per_contract(venue, model["params"]["symbol"])
    shares = (f" ({units * size:g} {CONTRACTS[model['params']['symbol']]['proxy']} shares)"
              if venue == "alpaca_paper" else "")
    return {"jobs": result.jobs, "waiting": result.waiting,
            "message": f"{model['name']} is trading on {VENUE_TEXT[venue]} at {size} contract{'s' if size != 1 else ''}"
                       f"{shares}; {where} sends its decisions to the coordinator, which places the orders"}


def stop(conn: psycopg.Connection, model_id: str, venue: str) -> dict[str, Any]:
    book = open_book(conn, model_id, venue)
    if book is None or book["status"] != "active":
        raise Conflict(f"This model is not trading on {VENUE_TEXT.get(venue, venue)}")
    conn.execute("UPDATE futures_books SET status = 'closing', target = '{}'::jsonb, waiting = NULL WHERE id = %s",
                 (book["id"],))
    if book["job_id"] is not None:
        conn.execute(
            "UPDATE jobs SET status = CASE WHEN status = 'queued' THEN 'cancelled' ELSE 'cancel_requested' END,"
            " finished_at = CASE WHEN status = 'queued' THEN now() ELSE finished_at END, updated_at = now()"
            " WHERE id = %s AND status IN ('queued', 'leased')", (book["job_id"],))
    return {"message": f"Stopped on {VENUE_TEXT[venue]}: any open position is closed now"}


def receive_signal(conn: psycopg.Connection, worker_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """The contracts a futures model wants from the next minute on, from its worker."""
    job = conn.execute("SELECT * FROM jobs WHERE id = %s::uuid", (body.get("job_id"),)).fetchone() \
        if body.get("job_id") else None
    if (job is None or job["kind"] != "paper_trade" or (job["params"] or {}).get("market") != "futures"
            or job["lease_worker_id"] != worker_id or job["status"] != "leased"):
        raise Conflict("this worker is not running that futures job")
    book = conn.execute("SELECT * FROM futures_books WHERE job_id = %s AND status = 'active' FOR UPDATE",
                        (job["id"],)).fetchone()
    if book is None:
        raise Conflict("the model is not trading any more")
    module = get_module(job["params"]["module"])
    wanted: dict[str, int] = {}
    for symbol, q in (body.get("contracts") or {}).items():
        if symbol not in module.SYMBOLS:
            raise BadRequest(f"{symbol!r} is not one of this model's symbols")
        if not isinstance(q, int) or isinstance(q, bool) or abs(q) > book["contracts"]:
            raise BadRequest(f"{symbol}: contracts must be a whole number from -{book['contracts']} to {book['contracts']}")
        wanted[symbol] = q
    minute_t = int(body.get("minute_t") or 0)
    if book["target_t"] is not None and minute_t <= book["target_t"]:
        return {"outcome": "already have this decision"}
    conn.execute("UPDATE futures_books SET target = %s, target_t = %s, target_reason = %s, target_worker = %s, "
                 "target_at = now() WHERE id = %s",
                 (Jsonb(wanted), minute_t, str(body.get("reason") or "")[:500], worker_id, book["id"]))
    return {"outcome": "kept"}


# ------------------------------------------------------------------ the clock


@dataclass(frozen=True)
class Clock:
    """Where `now` sits in today's session."""

    open: bool          # within today's regular session
    entries: bool       # new trades allowed (before the cut-off)
    closing: bool       # past the flat time: everything must be closed


def clock(now: datetime, rules: topstep.Rules) -> Clock:
    s = cme_session.session(trading_day(now))
    if s is None:
        return Clock(False, False, True)
    t = now.timestamp()
    flat_at = s[1] - 60 * (rules.flat_margin_minutes + rules.flat_minutes_before_close())
    if not s[0] <= t < s[1]:
        return Clock(False, False, True)
    return Clock(True, t < flat_at - 60 * CUTOFF_MINUTES, t >= flat_at)


# ------------------------------------------------------------------ the executor


def positions(conn: psycopg.Connection, book_id: int) -> dict[str, dict[str, Any]]:
    return {r["symbol"]: r for r in conn.execute("SELECT * FROM futures_positions WHERE book_id = %s", (book_id,))}


def held_contracts(venue: str, row: dict[str, Any] | None, symbol: str) -> float:
    return 0.0 if row is None else float(row["qty"]) / units_per_contract(venue, symbol)


def plan(book: dict[str, Any], held: dict[str, float], now: datetime, rules: topstep.Rules, paused: bool,
         topstep_stopped: str | None) -> tuple[dict[str, int], str | None, bool]:
    """(contracts wanted per symbol, why the book waits or None, closing_time)."""
    c = clock(now, rules)
    wanted = {s: int(q) for s, q in (book["target"] or {}).items()} if book["status"] == "active" else {}
    note = None
    if book["status"] == "active":
        if not c.open:
            return {s: 0 for s in held}, "Waiting for the session (8:30 to 15:00 Chicago time)", True
        if c.closing:
            return {s: 0 for s in held}, "Flat for the day: past the closing time", True
        if paused:
            return held_round(held), "Waiting: trading is paused", False
        fresh = book["target_at"] is not None and (now - book["target_at"]).total_seconds() <= rules.stale_decision_seconds
        if not fresh:
            wanted, note = {}, "No fresh decision from the worker: kept flat for safety"
        elif book["venue"] == "topstep" and topstep_stopped:
            wanted, note = {}, topstep_stopped
        if not c.entries:  # near the close: only reduce or close
            for s in set(wanted) | set(held):
                h, w = held.get(s, 0.0), wanted.get(s, 0)
                same = w != 0 and h != 0 and (w > 0) == (h > 0)
                wanted[s] = w if same and abs(w) < abs(h) else 0 if not same else int(round(h))
    elif paused and c.open and not c.closing:
        return held_round(held), "Waiting: trading is paused", False
    out = {s: 0 for s in held}
    out.update(wanted)
    return out, note, c.closing or not c.open


def held_round(held: dict[str, float]) -> dict[str, int]:
    return {s: int(round(q)) for s, q in held.items()}


def execute_pass(conn: psycopg.Connection, venues: Venues, rules: topstep.Rules, limits: Limits,
                 now: datetime | None = None) -> int:
    """Move every futures book toward what its model wants (see plan()). Returns orders sent."""
    now = now or datetime.now(timezone.utc)
    sent = 0
    books = conn.execute("SELECT b.*, m.name AS model_name FROM futures_books b JOIN models m ON m.id = b.model_id "
                         "WHERE b.status IN ('active', 'closing') ORDER BY b.id").fetchall()
    conn.commit()
    paused = safety.is_paused(conn)
    stopped = safety.topstep_paused(conn)
    for book in books:
        if conn.execute("SELECT 1 FROM futures_orders WHERE book_id = %s AND status = ANY(%s)",
                        (book["id"], list(OPEN_ORDER))).fetchone():
            continue
        if book["venue"] == "topstep" and not venues.topstep.on:
            _wait(conn, book["id"], venues.topstep.problem)
            continue
        pos = positions(conn, book["id"])
        held = {s: held_contracts(book["venue"], r, s) for s, r in pos.items()}
        wanted, note, closing_time = plan(book, held, now, rules, paused, stopped)
        halted = None
        for symbol in sorted(set(wanted) | set(held)):
            have, want = held.get(symbol, 0.0), float(wanted.get(symbol, 0))
            if abs(want - have) < 1e-9:
                continue
            if have != 0 and want != 0 and (want > 0) != (have > 0):
                want = 0.0  # a flip: close first, open the other side on the next pass
            row = _order(conn, venues, rules, limits, book, symbol, have, want, closing_time, now)
            sent += row["status"] != "blocked"
            if row["status"] == "blocked":
                halted = row["error"]
                break
        poll_fills(conn, venues, rules, book_id=book["id"], now=now)
        _wait(conn, book["id"], ("Waiting: " + halted) if halted else note)
        if book["status"] == "closing":
            close_if_flat(conn, book["id"])
        conn.commit()
    return sent


def _wait(conn: psycopg.Connection, book_id: int, text: str | None) -> None:
    conn.execute("UPDATE futures_books SET waiting = %s WHERE id = %s", (text, book_id))
    conn.commit()


def _order(conn: psycopg.Connection, venues: Venues, rules: topstep.Rules, limits: Limits, book: dict[str, Any],
           symbol: str, have: float, want: float, closing_time: bool, now: datetime) -> dict[str, Any]:
    venue = book["venue"]
    units = units_per_contract(venue, symbol)
    delta = want - have
    reducing = abs(want) < abs(have) and (want == 0 or (want > 0) == (have > 0))
    order: dict[str, Any] = {
        "symbol": symbol, "side": "buy" if delta > 0 else "sell", "qty": round(abs(delta) * units, 6),
        "contracts": delta, "after_contracts": want, "reducing": reducing, "closing_time": closing_time,
        "worker_id": book["target_worker"],
    }
    try:
        order["instrument"] = instrument(venues, venue, symbol)
    except Exception as exc:  # noqa: BLE001 - TopstepX unreachable: logged as blocked
        order["instrument"] = symbol
        order["reason"] = f"{book['model_name']}: could not look up the {symbol} contract"
        return safety._futures_insert(conn, book, order, "blocked", str(exc)[:300])
    if venue == "alpaca_paper":
        order["exposure_after"] = _alpaca_exposure(conn, venues, book["id"], symbol, want)
    else:
        order["account_contracts_after"] = _topstep_contracts(conn, book["id"], symbol, want)
    side = "flat" if want == 0 else f"long {want:g}" if want > 0 else f"short {-want:g}"
    why = book["target_reason"] or "no decision yet"
    if book["status"] == "closing":
        why = "trading stopped"
    elif closing_time:
        why = "flat by the close"
    order["reason"] = f"{book['model_name']} wants {side} {symbol} ({why})"
    recent = conn.execute(
        "SELECT * FROM futures_orders WHERE book_id = %s AND symbol = %s AND status = 'blocked' AND contracts = %s "
        "AND created_at > now() - interval '60 seconds' ORDER BY created_at DESC LIMIT 1",
        (book["id"], symbol, delta)).fetchone()
    if recent is not None and safety.futures_problem(conn, venues, limits, rules, book, order) == recent["error"]:
        return recent  # refused a moment ago for the same reason: wait, without logging it again
    return safety.approve_futures_order(conn, venues, limits, rules, book, order)


def _alpaca_exposure(conn: psycopg.Connection, venues: Venues, book_id: int, symbol: str, want: float) -> float:
    """Dollars all futures paper books would hold in SPY/QQQ after this order."""
    total = 0.0
    for r in conn.execute("SELECT p.* FROM futures_positions p JOIN futures_books b ON b.id = p.book_id "
                          "WHERE b.venue = 'alpaca_paper' AND b.status IN ('active', 'closing')"):
        if r["book_id"] == book_id and r["symbol"] == symbol:
            continue
        total += abs(r["qty"]) * (futures_live.share_price(conn, r["instrument"]) or r["avg_price"])
    stand_in = CONTRACTS[symbol]["proxy"]
    price = futures_live.share_price(conn, stand_in) or 0.0
    return total + abs(want) * units_per_contract("alpaca_paper", symbol) * price


def _topstep_contracts(conn: psycopg.Connection, book_id: int, symbol: str, want: float) -> float:
    total = 0.0
    for r in conn.execute("SELECT p.* FROM futures_positions p JOIN futures_books b ON b.id = p.book_id "
                          "WHERE b.venue = 'topstep' AND b.status IN ('active', 'closing')"):
        if not (r["book_id"] == book_id and r["symbol"] == symbol):
            total += abs(r["qty"])
    return total + abs(want)


def close_if_flat(conn: psycopg.Connection, book_id: int) -> bool:
    if conn.execute("SELECT 1 FROM futures_positions WHERE book_id = %s", (book_id,)).fetchone():
        return False
    if conn.execute("SELECT 1 FROM futures_orders WHERE book_id = %s AND status = ANY(%s)",
                    (book_id, list(OPEN_ORDER))).fetchone():
        return False
    conn.execute("UPDATE futures_books SET status = 'closed', closed_at = now(), waiting = NULL WHERE id = %s", (book_id,))
    return True


# ------------------------------------------------------------------ fills and days


def book_fill(conn: psycopg.Connection, order: dict[str, Any], new_qty: float, price: float, rules: topstep.Rules,
              now: datetime) -> None:
    """Move a newly filled quantity into the book; realised profit (in futures dollars,
    commission taken off) into the day's record."""
    venue, symbol = order["venue"], order["symbol"]
    pos = conn.execute("SELECT * FROM futures_positions WHERE book_id = %s AND symbol = %s FOR UPDATE",
                       (order["book_id"], symbol)).fetchone()
    signed = new_qty if order["side"] == "buy" else -new_qty
    q, avg = (float(pos["qty"]), float(pos["avg_price"])) if pos else (0.0, 0.0)
    realised, opened = 0.0, False
    if q == 0 or (q > 0) == (signed > 0):
        avg = (abs(q) * avg + abs(signed) * price) / (abs(q) + abs(signed))
        opened = q == 0
        q += signed
    else:
        closed = min(abs(signed), abs(q))
        realised = closed * (price - avg) * (1 if q > 0 else -1) * dollars_per_point(venue, symbol)
        q += signed
        if abs(signed) > closed:
            avg, opened = price, True
    fees = float(rules.commission(symbol) or 0.0) * new_qty / units_per_contract(venue, symbol)
    if abs(q) < 1e-9:
        conn.execute("DELETE FROM futures_positions WHERE book_id = %s AND symbol = %s", (order["book_id"], symbol))
    else:
        conn.execute("INSERT INTO futures_positions (book_id, symbol, instrument, qty, avg_price) VALUES "
                     "(%s, %s, %s, %s, %s) ON CONFLICT (book_id, symbol) DO UPDATE SET qty = EXCLUDED.qty, "
                     "avg_price = EXCLUDED.avg_price, instrument = EXCLUDED.instrument",
                     (order["book_id"], symbol, order["instrument"], q, avg))
    conn.execute("INSERT INTO futures_days (book_id, day, pnl, fees, trades) VALUES (%s, %s, %s, %s, %s) "
                 "ON CONFLICT (book_id, day) DO UPDATE SET pnl = futures_days.pnl + EXCLUDED.pnl, "
                 "fees = futures_days.fees + EXCLUDED.fees, trades = futures_days.trades + EXCLUDED.trades",
                 (order["book_id"], trading_day(now), realised - fees, fees, 1 if opened else 0))


def poll_fills(conn: psycopg.Connection, venues: Venues, rules: topstep.Rules, book_id: int | None = None,
               now: datetime | None = None) -> int:
    """Ask the broker about every open futures order and book what filled."""
    now = now or datetime.now(timezone.utc)
    rows = conn.execute(
        "SELECT * FROM futures_orders WHERE (status IN ('submitted', 'partially_filled') OR (status = 'submitting' "
        "AND created_at < now() - make_interval(secs => %s))) AND (%s::bigint IS NULL OR book_id = %s) ORDER BY created_at",
        (STALE_SUBMITTING_S, book_id, book_id)).fetchall()
    changed = 0
    for order in rows:
        try:
            if order["venue"] == "alpaca_paper":
                state = venues.alpaca.broker.get_order(str(order["id"]))
            else:
                state = venues.topstep.client.get_order(str(order["id"]), order["broker_order_id"])
        except Exception as exc:  # noqa: BLE001
            log.warning("futures fill check for %s failed: %s", order["id"], exc)
            continue
        new_qty = max(0.0, state.filled_qty - order["booked_qty"])
        booked = 0.0
        if new_qty > 0 and state.filled_avg_price:
            book_fill(conn, order, new_qty, state.filled_avg_price, rules, now)
            booked = new_qty
        done = state.status in ("filled", "cancelled", "rejected")
        conn.execute(
            "UPDATE futures_orders SET status = %s, filled_qty = %s, filled_avg_price = %s, booked_qty = booked_qty + %s,"
            " broker_order_id = COALESCE(%s, broker_order_id),"
            " error = CASE WHEN %s IN ('filled', 'partially_filled') THEN NULL ELSE COALESCE(%s, error) END,"
            " finished_at = CASE WHEN %s THEN now() ELSE finished_at END WHERE id = %s",
            (state.status, state.filled_qty, state.filled_avg_price, booked, state.broker_order_id, state.status,
             state.error if state.error != NEVER_REACHED or order["status"] == "submitting" else state.error, done,
             order["id"]))
        conn.commit()
        changed += 1
    return changed


def mark_days(conn: psycopg.Connection, venues: Venues, now: datetime | None = None) -> None:
    """Today's worst moment per book so far: realised today plus open trades valued at
    the latest minute's worst price (the dip the loss limit watches)."""
    now = now or datetime.now(timezone.utc)
    day = trading_day(now)
    for book in conn.execute("SELECT * FROM futures_books WHERE status IN ('active', 'closing')").fetchall():
        source = futures_live.source_for(book["venue"], venues.fake)
        open_value = 0.0
        for p in conn.execute("SELECT * FROM futures_positions WHERE book_id = %s", (book["id"],)).fetchall():
            bar = futures_live.latest(conn, source, p["symbol"])
            if bar is None:
                continue
            spec = CONTRACTS[p["symbol"]]
            worst = bar["low"] if p["qty"] > 0 else bar["high"]
            if book["venue"] == "alpaca_paper":
                worst = worst / spec["proxy_scale"]
            open_value += (worst - p["avg_price"]) * p["qty"] * dollars_per_point(book["venue"], p["symbol"])
        realised = conn.execute("SELECT pnl FROM futures_days WHERE book_id = %s AND day = %s",
                                (book["id"], day)).fetchone()
        value = (realised["pnl"] if realised else 0.0) + open_value
        conn.execute("INSERT INTO futures_days (book_id, day, dip) VALUES (%s, %s, LEAST(0, %s)) "
                     "ON CONFLICT (book_id, day) DO UPDATE SET dip = LEAST(futures_days.dip, EXCLUDED.dip)",
                     (book["id"], day, value))
    conn.commit()


# ------------------------------------------------------------------ Topstep account


def topstep_check(conn: psycopg.Connection, venues: Venues, rules: topstep.Rules, now: datetime | None = None) -> dict[str, Any] | None:
    """The Topstep account's balance, loss floor and room; pauses Topstep trading at the
    safety margin; records the balance once the day is over. None when Topstep is off."""
    if not venues.topstep.on:
        return None
    now = now or datetime.now(timezone.utc)
    acc = venues.topstep.client.account(refresh=True)
    source = futures_live.source_for("topstep", venues.fake)
    open_value = 0.0
    for p in conn.execute("SELECT p.* FROM futures_positions p JOIN futures_books b ON b.id = p.book_id "
                          "WHERE b.venue = 'topstep' AND b.status IN ('active', 'closing')").fetchall():
        bar = futures_live.latest(conn, source, p["symbol"])
        if bar is not None:
            open_value += (float(bar["close"]) - p["avg_price"]) * p["qty"] * CONTRACTS[p["symbol"]]["point_value"]
    equity = acc.balance + min(0.0, open_value)  # open losses count, open gains do not
    safety.check_topstep_loss(conn, acc.id, equity, rules)
    day = trading_day(now)
    s = cme_session.session(day)
    if s is not None and now.timestamp() >= s[1] + 600:  # 15:10 Chicago time: the day is over
        conn.execute("INSERT INTO topstep_days (account, day, balance) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                     (acc.id, day, acc.balance))
    floor = safety.topstep_floor(conn, acc.id, rules)
    conn.commit()
    return {"account": acc.name, "balance": acc.balance, "equity": equity, "floor": floor, "room": equity - floor,
            "can_trade": acc.can_trade}


# ------------------------------------------------------------------ results for the dashboard


def paper_record(conn: psycopg.Connection, model_id: str, band: dict[str, float] | None, contracts: int,
                 rules: topstep.Rules, today: Any) -> dict[str, Any]:
    """The model's finished Alpaca paper days compared with the backtest's range of daily
    results (the held-out 1st to 99th percentile, scaled to the contracts traded)."""
    rows = conn.execute("SELECT d.day, d.pnl FROM futures_days d JOIN futures_books b ON b.id = d.book_id "
                        "WHERE b.model_id = %s AND b.venue = 'alpaca_paper' AND d.day < %s ORDER BY d.day",
                        (model_id, today)).fetchall()
    by_day: dict[Any, float] = {}
    for r in rows:
        by_day[r["day"]] = by_day.get(r["day"], 0.0) + float(r["pnl"])
    pnl = np.asarray(list(by_day.values()))
    out: dict[str, Any] = {"days": int(pnl.size), "total": float(pnl.sum()) if pnl.size else 0.0,
                           "outside": None, "low": None, "high": None}
    if band:
        low, high = band["p01"] * contracts, band["p99"] * contracts
        out.update(low=low, high=high, outside=int(((pnl < low - 1e-9) | (pnl > high + 1e-9)).sum()))
    return out


def paper_ok(record: dict[str, Any], rules: topstep.Rules) -> str:
    """"ok", "no" or "pending" for the checklist's Alpaca paper line: enough days, few
    enough of them outside the backtest's range, and a profit overall."""
    if record["outside"] is None:
        return "no"  # tested before the range was kept: run the backtest again
    if record["days"] < rules.min_shadow_days:
        return "pending"
    inside = record["outside"] <= rules.max_days_outside * record["days"]
    return "ok" if inside and record["total"] > 0 else "no"


def book_line(conn: psycopg.Connection, book: dict[str, Any], now: datetime) -> dict[str, Any]:
    """One plain sentence about a book: what it holds, today and in total."""
    pos = positions(conn, book["id"])
    parts = []
    for s, r in pos.items():
        n = held_contracts(book["venue"], r, s)
        text = f"{'long' if n > 0 else 'short'} {abs(n):g} {s}"
        if book["venue"] == "alpaca_paper":
            text += f" ({abs(r['qty']):g} {r['instrument']})"
        parts.append(text)
    days = conn.execute("SELECT day, pnl FROM futures_days WHERE book_id = %s ORDER BY day", (book["id"],)).fetchall()
    today = trading_day(now)
    today_pnl = sum(float(d["pnl"]) for d in days if d["day"] == today)
    total = sum(float(d["pnl"]) for d in days)
    since = book["started_at"].astimezone(cme_session.CHICAGO)
    line = (f"{VENUE_TEXT[book['venue']]} since {since:%b} {since.day}: "
            f"{', '.join(parts) if parts else 'flat'} · today {_money(today_pnl)} · {len(days)} day"
            f"{'s' if len(days) != 1 else ''}, {_money(total)} in all")
    if book["status"] == "closing":
        line += " · stopping"
    signal = None
    if book["status"] == "active" and book["target"] is not None:
        wanted = ", ".join(f"{'long' if q > 0 else 'short'} {abs(q)} {s}" for s, q in book["target"].items() if q) or "flat"
        signal = f"Live signal: {wanted} ({book['target_reason'] or 'no reason given'})"
    return {"venue": book["venue"], "line": line, "waiting": book["waiting"], "signal": signal,
            "tone": "gain" if total > 0 else "loss" if total < 0 else "plain"}


def _money(v: float) -> str:
    return ("+" if v >= 0 else "−") + f"${abs(v):,.0f}"
