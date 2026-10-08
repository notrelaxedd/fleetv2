"""Job registry for the runner child process (copied from v1, trimmed).

A job function has the signature run(params, checkpoint, emit, should_stop) -> result.
It calls emit(checkpoint, progress, detail) after every unit of work (units must take
under 3 s, so a stop is always quick) and raises JobStopped when should_stop() turns
true between units. progress is 0..1, or None for a job with no end (paper trading
shows "Live"). detail is the one line the Fleet card shows under the task name.

Kinds: sleep (a test job for watching progress on the Fleet screen), data_refresh,
backtest, paper_trade, model_search and final_check. A backtest of a futures model and
a model search on the futures market run the futures versions (fleet2/worker/futures_*).
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


def run_data_refresh(params: dict[str, Any], checkpoint: dict[str, Any] | None, emit: Emit, should_stop: ShouldStop) -> Any:
    from fleet2.worker.data_job import run_data_refresh as run

    return run(params, checkpoint, emit, should_stop)


def run_backtest(params: dict[str, Any], checkpoint: dict[str, Any] | None, emit: Emit, should_stop: ShouldStop) -> Any:
    if params.get("market") == "futures":
        from fleet2.worker.futures_jobs import run_futures_backtest

        return run_futures_backtest(params, checkpoint, emit, should_stop)
    from fleet2.worker.backtest_job import run_backtest_job

    return run_backtest_job(params, checkpoint, emit, should_stop)


def run_paper_trade(params: dict[str, Any], checkpoint: dict[str, Any] | None, emit: Emit, should_stop: ShouldStop) -> Any:
    from fleet2.worker.paper_job import run_paper_trade as run

    return run(params, checkpoint, emit, should_stop)


def run_model_search(params: dict[str, Any], checkpoint: dict[str, Any] | None, emit: Emit, should_stop: ShouldStop) -> Any:
    if params.get("markets") == ["futures"]:
        from fleet2.worker.futures_search_job import run_futures_search

        return run_futures_search(params, checkpoint, emit, should_stop)
    from fleet2.worker.search_job import run_search

    return run_search(params, checkpoint, emit, should_stop)


def run_final_check(params: dict[str, Any], checkpoint: dict[str, Any] | None, emit: Emit, should_stop: ShouldStop) -> Any:
    from fleet2.worker.futures_jobs import run_final_check as run

    return run(params, checkpoint, emit, should_stop)


JOBS: dict[str, JobFunc] = {
    "sleep": run_sleep,
    "data_refresh": run_data_refresh,
    "backtest": run_backtest,
    "paper_trade": run_paper_trade,
    "model_search": run_model_search,
    "final_check": run_final_check,
}
