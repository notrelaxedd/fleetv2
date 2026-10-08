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
