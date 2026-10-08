"""Job registry for the runner child process (copied from v1, trimmed).

A job function has the signature run(params, checkpoint, emit, should_stop) -> result.
It calls emit(checkpoint, progress, detail) after every unit of work (units must take
under 3 s, so a stop is always quick) and raises JobStopped when should_stop() turns
true between units. progress is 0..1, or None for a job with no end (paper trading
shows "Live"). detail is the one line the Fleet card shows under the task name.

Stage 1 ships only `sleep`, the test job used to watch progress move on the Fleet
screen. Stage 2 adds data_refresh and backtest, stage 4 paper_trade, stage 5
model_search.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from fleet2.sim.control import JobStopped

Emit = Callable[..., None]
ShouldStop = Callable[[], bool]
JobFunc = Callable[[dict[str, Any], dict[str, Any] | None, Emit, ShouldStop], Any]


def run_sleep(
    params: dict[str, Any],
    checkpoint: dict[str, Any] | None,
    emit: Emit,
    should_stop: ShouldStop,
) -> dict[str, Any]:
    """Sleep params["seconds"] in 1 s units; checkpoint {"elapsed": n}; result {"slept": seconds}."""
    seconds = max(1, int(params.get("seconds", 30)))
    elapsed = 0
    if checkpoint:
        elapsed = int(checkpoint.get("elapsed", 0))
    elapsed = max(0, min(elapsed, seconds))
    while elapsed < seconds:
        if should_stop():
            raise JobStopped()
        time.sleep(1.0)
        elapsed += 1
        emit({"elapsed": elapsed}, elapsed / seconds, f"Test job: {elapsed} of {seconds} seconds")
    return {"slept": seconds, "summary": f"Slept {seconds} s"}


JOBS: dict[str, JobFunc] = {
    "sleep": run_sleep,
}
