"""Memory watchdog: finds runners whose process tree uses too much RAM.

The agent's 0.1 s service loop calls MemoryWatchdog.over_limit() with its running
jobs. At most once per second per runner the watchdog sums the proportional
anonymous memory (Pss_Anon + Pss_Shmem, see fleet2.common.sysinfo.process_mem_kb)
over the runner's session (sysinfo.session_rss_kb; the runner child is its own
session leader, so grandchildren count) and compares the sum with
fraction * ram_total_mb. Shared copy-on-write pages of forked children are counted
once and file cache is not counted, so a healthy fork pool does not trip it. The agent stops and releases (reason "oom") every job the
watchdog returns. The first check of a runner happens one interval after it was
first seen, so a child that is still starting is never signalled before its
SIGTERM handler is installed.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from fleet2.common import sysinfo

log = logging.getLogger("fleet2.watchdog")

DEFAULT_FRACTION = 0.8
DEFAULT_INTERVAL = 1.0


class MemoryWatchdog:
    """Per-runner RSS check with a trip counter."""

    def __init__(
        self,
        fraction: float = DEFAULT_FRACTION,
        ram_total_mb: int | None = None,
        interval: float = DEFAULT_INTERVAL,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.fraction = fraction
        self._ram_total_mb = ram_total_mb
        self.interval = interval
        self._clock = clock
        self.trips = 0
        self._next_check: dict[str, float] = {}

    def ram_total_mb(self) -> int | None:
        """Configured total (tests) or the machine's MemTotal; None when unknown."""
        if self._ram_total_mb is not None:
            return self._ram_total_mb
        return sysinfo.ram_total_mb()

    def limit_mb(self) -> float | None:
        total = self.ram_total_mb()
        if total is None or total <= 0 or self.fraction <= 0:
            return None
        return self.fraction * total

    def forget(self, job_id: str) -> None:
        self._next_check.pop(job_id, None)

    def over_limit(self, running: dict[str, Any]) -> list[str]:
        """Job ids whose runner session exceeds the limit. A warning with the MB
        figures is logged and the trip counter bumped for each one."""
        now = self._clock()
        for job_id in list(self._next_check):
            if job_id not in running:
                del self._next_check[job_id]
        limit = self.limit_mb()
        offenders: list[str] = []
        for job_id, rj in running.items():
            due = self._next_check.get(job_id)
            if due is None:
                self._next_check[job_id] = now + self.interval
                continue
            if now < due:
                continue
            self._next_check[job_id] = now + self.interval
            pid = getattr(rj.runner, "pid", None)
            if limit is None or pid is None:
                continue
            rss_mb = sysinfo.session_rss_kb(pid) / 1024.0
            if rss_mb > limit:
                self.trips += 1
                log.warning(
                    "memory watchdog: job %s uses %.0f MB, over %.0f MB (%.0f%% of %d MB); stopping it",
                    job_id, rss_mb, limit, self.fraction * 100.0, self.ram_total_mb() or 0,
                )
                offenders.append(job_id)
        return offenders
