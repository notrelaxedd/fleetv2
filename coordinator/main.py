"""Coordinator entry point: migrate, start the background loop, serve the API and dashboard."""
from __future__ import annotations

import logging

import uvicorn

from coordinator import db
from coordinator.api.app import create_app
from coordinator.config import Config
from coordinator.loop import LoopThread

log = logging.getLogger("coordinator.main")


def main() -> None:
    """python -m coordinator.main"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = Config.from_env()
    applied = db.migrate(config.database_url)
    log.info("migrations applied: %s", applied or "none")
    app = create_app(config)
    status = app.state.broker_status
    log.info("trading mode: %s (broker connected: %s)", status.broker.mode, status.broker.connected)
    loop_pool = db.make_pool(config.database_url, min_size=1, max_size=3)
    loop = LoopThread(loop_pool, config.loop_seconds, extra=(lambda _pool: status.refresh(),))
    loop.start()
    log.info("serving on %s (public url %s, dev=%s)", config.bind, config.public_url, config.dev)
    try:
        uvicorn.run(app, host=config.bind_host, port=config.bind_port, log_level="warning")
    finally:
        loop.stop()
        loop_pool.close()


if __name__ == "__main__":
    main()
