"""The background loop (copied from v1): reaper and dispatcher every few seconds.

Stage 4 adds the daily loss check here; stage 2 the scheduled data refresh.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from psycopg_pool import ConnectionPool

from coordinator import queue

log = logging.getLogger(__name__)

Task = Callable[[ConnectionPool], Any]


def run_once(pool: ConnectionPool, extra: tuple[Task, ...] = ()) -> dict[str, Any]:
    """One reaper pass and one dispatcher pass, each in its own transaction, then any
    extra tasks (each isolated: one failing never stops the others)."""
    with pool.connection() as conn:
        reaped = queue.reap(conn)
    with pool.connection() as conn:
        queue.release_dead_targets(conn)
        dispatched = queue.dispatch(conn)
    for task in extra:
        try:
            task(pool)
        except Exception:  # noqa: BLE001 - keep the loop alive
            log.exception("loop task %s failed", getattr(task, "__name__", task))
    if reaped:
        log.warning("loop: %d job(s) failed because their worker went offline", len(reaped))
    return {"reaped": len(reaped), "dispatched": len(dispatched)}


class LoopThread(threading.Thread):
    """Runs run_once every `interval` seconds until stopped; never dies on errors."""

    def __init__(self, pool: ConnectionPool, interval: float, extra: tuple[Task, ...] = ()) -> None:
        super().__init__(name="fleet2-loop", daemon=True)
        self.pool = pool
        self.interval = interval
        self.extra = extra
        # Not named _stop: threading.Thread has a private _stop() that join() calls.
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                run_once(self.pool, self.extra)
            except Exception:  # noqa: BLE001 - keep the loop alive
                log.exception("background loop iteration failed")
            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        """Ask the thread to exit after the current iteration."""
        self._stop_event.set()
