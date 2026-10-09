"""Coordinator entry point: migrate, start the background loop, serve the API and dashboard."""
from __future__ import annotations

import logging

import uvicorn

from coordinator import db, recovery
from coordinator.leases import lease_seconds
from coordinator.api.app import create_app
from coordinator.config import Config
from coordinator.loop import LoopThread
from coordinator.tasks import make_ai_tasks, make_futures_tasks, make_tasks
from coordinator import trading

log = logging.getLogger("coordinator.main")


def main() -> None:
    """python -m coordinator.main"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = Config.from_env()
    applied = db.migrate(config.database_url)
    log.info("migrations applied: %s", applied or "none")
    with db.connect(config.database_url) as conn:
        kept = recovery.startup_grace(conn, lease_seconds(conn))
    if kept:
        log.info("kept %d running job(s) alive while their workers reconnect", kept)
    app = create_app(config)
    status = app.state.broker_status
    log.info("trading mode: %s (broker connected: %s)", status.broker.mode, status.broker.connected)
    loop_pool = db.make_pool(config.database_url, min_size=1, max_size=3)
    if getattr(status.broker, "fake", False):
        status.broker.price_of = lambda symbol: (_live_share_price(loop_pool, symbol)
                                                 or next(iter(trading.latest_prices_from_pool(loop_pool, [symbol]).values()), 100.0))
    venues = app.state.venues
    if getattr(venues.topstep.client, "fake", False):
        venues.topstep.client.price_of = lambda symbol: _live_close(loop_pool, symbol) or 5000.0
    extra = make_tasks(status, app.state.limits, app.state.bar_source) + make_futures_tasks(
        venues, app.state.topstep, app.state.limits) + make_ai_tasks(app.state.ai, config.anthropic_api_key)
    loop = LoopThread(loop_pool, config.loop_seconds, extra=extra)
    loop.start()
    log.info("serving on %s (public url %s, dev=%s)", config.bind, config.public_url, config.dev)
    try:
        uvicorn.run(app, host=config.bind_host, port=config.bind_port, log_level="warning")
    finally:
        loop.stop()
        loop_pool.close()


def _live_share_price(pool, symbol: str) -> float | None:
    """Demo fills of SPY/QQQ for futures paper books follow the live synthetic prices."""
    from coordinator import futures_live

    with pool.connection() as conn:
        return futures_live.share_price(conn, symbol)


def _live_close(pool, symbol: str) -> float | None:
    from coordinator import futures_live

    with pool.connection() as conn:
        row = futures_live.latest(conn, "synthetic", symbol)
    return float(row["close"]) if row else None


if __name__ == "__main__":
    main()
