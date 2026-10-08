"""Cooperative stop signal shared by the sim modules and the worker job registry.

JobStopped lives here (and not in fleet2.worker.jobs) so the sim package has no
dependency on the worker package; fleet2.worker.jobs re-exports it.
"""

from __future__ import annotations

from typing import Callable

ShouldStop = Callable[[], bool]


class JobStopped(Exception):
    """Raised by a job when should_stop() is true; the runner reports the last checkpoint."""


def check_stop(should_stop: ShouldStop) -> None:
    """Raise JobStopped when the runner asked the job to stop."""
    if should_stop():
        raise JobStopped()
