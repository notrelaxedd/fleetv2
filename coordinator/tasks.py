"""Background work on the coordinator, run by the loop every few seconds (each task in
its own transaction; one failing never stops the others):

- broker status: the market clock and the account, at most every 30 s
- daily loss: pause all trading when the account is down more than the limit today
- trading: turn models' decisions into orders and book fills
- price data: keep the bars fresh for markets that have a model paper trading
  (crypto every hour, stocks every few hours), so workers decide on current data
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable

from psycopg_pool import ConnectionPool

from coordinator import data, safety, trading
from coordinator.broker import BrokerStatus
from coordinator.limits import Limits
from fleet2.universe import MARKETS

log = logging.getLogger(__name__)
REFRESH_EVERY_S = {"crypto": 15 * 60, "stocks": 3 * 3600}


def make_tasks(status: BrokerStatus, limits: Limits, bar_source: Any) -> tuple[Callable[[ConnectionPool], None], ...]:
    last_refresh: dict[str, float] = {}

    def broker_status(pool: ConnectionPool) -> None:
        status.refresh()

    def daily_loss(pool: ConnectionPool) -> None:
        with pool.connection() as conn:
            safety.check_daily_loss(conn, status, limits)

    def trade(pool: ConnectionPool) -> None:
        if not status.broker.connected:
            return
        with pool.connection() as conn:
            trading.poll_fills(conn, status)
            trading.execute_pass(conn, status, limits)

    def fresh_prices(pool: ConnectionPool) -> None:
        with pool.connection() as conn:
            markets = [r["market"] for r in conn.execute(
                "SELECT DISTINCT m.market FROM books b JOIN models m ON m.id = b.model_id WHERE b.status = 'active'")]
        now = time.monotonic()
        for market in markets:
            if now - last_refresh.get(market, -1e9) < REFRESH_EVERY_S[market]:
                continue
            last_refresh[market] = now
            refresh_market(pool, bar_source, market)

    return (broker_status, daily_loss, trade, fresh_prices)


def refresh_market(pool: ConnectionPool, bar_source: Any, market: str, max_steps: int = 50) -> int:
    """Bring one market's bars up to date on the coordinator (no worker needed)."""
    added = 0
    for symbol in MARKETS[market]["symbols"]:
        cursor = None
        for _ in range(max_steps):
            with pool.connection() as conn:
                step = data.refresh_step(conn, bar_source, market, symbol, datetime.now(timezone.utc), cursor)
            added += int(step.get("added") or 0)
            if step.get("done") or step.get("error"):
                break
            cursor = datetime.fromisoformat(step["cursor"]) if step.get("cursor") else None
    if added:
        log.info("price refresh %s: %d new bars", market, added)
    return added


def make_futures_tasks(venues: Any, rules: Any, limits: Limits) -> tuple[Callable[[ConnectionPool], None], ...]:
    """Futures models trading (coordinator.futures_trading), each in its own transaction:

    - live prices: every few seconds while a futures book trades and the session is open,
      the newest closed minutes of each place's own prices (futures_live)
    - trade: book fills, move each book to what its model wants, value today's dips
    - Topstep: the account's balance and loss floor every 30 s (pausing Topstep trading at
      its safety margin) and the balance at the end of each day
    """
    from coordinator import futures_live, futures_trading

    last: dict[str, float] = {}

    def live_prices(pool: ConnectionPool) -> None:
        now_m = time.monotonic()
        if now_m - last.get("prices", -1e9) < futures_live.REFRESH_S:
            return
        last["prices"] = now_m
        now = datetime.now(timezone.utc)
        with pool.connection() as conn:
            rows = conn.execute("SELECT DISTINCT venue FROM futures_books WHERE status IN ('active', 'closing')").fetchall()
            for r in rows:
                name = futures_live.source_for(r["venue"], venues.fake)
                source = venues.sources.get(name)
                if source is None:
                    continue
                empty = futures_live.latest(conn, name, "MES") is None
                if not (empty or futures_live.session_window(now)):
                    continue
                for symbol in MARKETS_FUTURES:
                    try:
                        futures_live.refresh(conn, name, source, symbol, now)
                        conn.commit()
                    except Exception as exc:  # noqa: BLE001 - reported, retried next time
                        conn.rollback()
                        log.warning("live futures prices %s %s: %s", name, symbol, exc)

    def trade(pool: ConnectionPool) -> None:
        with pool.connection() as conn:
            if not conn.execute("SELECT 1 FROM futures_books WHERE status IN ('active', 'closing') LIMIT 1").fetchone():
                return
            futures_trading.poll_fills(conn, venues, rules)
            futures_trading.execute_pass(conn, venues, rules, limits)
            futures_trading.mark_days(conn, venues)

    def topstep_account(pool: ConnectionPool) -> None:
        now_m = time.monotonic()
        if not venues.topstep.on or now_m - last.get("topstep", -1e9) < 30:
            return
        last["topstep"] = now_m
        with pool.connection() as conn:
            try:
                venues.topstep_state = futures_trading.topstep_check(conn, venues, rules)
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                venues.topstep_state = {"error": str(exc)[:200]}
                log.warning("Topstep account check: %s", exc)

    return (live_prices, trade, topstep_account)


MARKETS_FUTURES = ("MES", "MNQ")
