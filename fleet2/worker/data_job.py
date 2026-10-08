"""The Data refresh job: drives the coordinator, one symbol at a time.

The coordinator holds the Alpaca keys, downloads the bars and caches them; this job
only asks it to refresh each symbol of the chosen markets (POST /api/v1/data/refresh-step)
so the progress shows on the Fleet screen. Nothing is written to disk and the job never
talks to Alpaca. Standard library only.

params: {"markets": ["stocks", "crypto"]} (default both). checkpoint:
{"symbol_index": i, "bars_added": n, "errors": {...}}, i being the symbol in progress
(a resumed job starts there again; refreshing a symbol twice is harmless).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable

from fleet2 import universe
from fleet2.common import http
from fleet2.sim.control import JobStopped

STEP_TIMEOUT = 90.0  # one step may wait in the coordinator's Alpaca rate limiter
RETRIES = 3  # attempts per call for a 5xx answer or a lost connection
RETRY_PAUSE = 1.5  # seconds before the second attempt, doubled before the third
MAX_STEPS_PER_SYMBOL = 100  # crypto windows are 120 days; far more than any real history needs

_sleep: Callable[[float], None] = time.sleep  # replaced in tests


def _markets(params: dict[str, Any]) -> list[str]:
    raw = params.get("markets") or ["stocks", "crypto"]
    names = [raw] if isinstance(raw, str) else list(raw)
    for name in names:
        if name not in universe.MARKETS:
            raise ValueError(f"unknown market {name!r}")
    return names


def _parse(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _call(url: str, token: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST one step; a 5xx or a lost connection is retried, anything else is raised."""
    for attempt in range(RETRIES):
        try:
            reply = http.post_json(url, body, token=token, timeout=STEP_TIMEOUT)
            return reply if isinstance(reply, dict) else {}
        except http.HttpConnectionError:
            if attempt == RETRIES - 1:
                raise
        except http.HttpError as exc:
            if exc.status < 500 or attempt == RETRIES - 1:
                raise
        _sleep(RETRY_PAUSE * (2 ** attempt))
    raise AssertionError("unreachable")


def _describe(exc: Exception) -> str:
    if isinstance(exc, http.HttpError):
        return exc.detail or f"HTTP {exc.status}"
    return str(exc)


def run_data_refresh(
    params: dict[str, Any],
    checkpoint: dict[str, Any] | None,
    emit: Callable[..., None],
    should_stop: Callable[[], bool],
) -> dict[str, Any]:
    context = params.get("_context") or {}
    url = str(context.get("host_url", "")).rstrip("/") + "/api/v1/data/refresh-step"
    token = str(context.get("worker_token", ""))
    todo = [(m, s) for m in _markets(params) for s in universe.MARKETS[m]["symbols"]]
    total = len(todo)

    start_index = 0
    bars_added = 0
    errors: dict[str, str] = {}
    if checkpoint:
        start_index = max(0, min(total, int(checkpoint.get("symbol_index", 0) or 0)))
        bars_added = int(checkpoint.get("bars_added", 0) or 0)
        prior = checkpoint.get("errors")
        if isinstance(prior, dict):
            errors = {str(k): str(v) for k, v in prior.items()}
    run_started = datetime.now(timezone.utc)

    for i in range(start_index, total):
        market, symbol = todo[i]
        errors.pop(symbol, None)
        cursor: str | None = None
        base: datetime | None = None  # where this symbol's first step started (crypto progress)
        closed = False  # the symbol's closing progress line was emitted
        for _ in range(MAX_STEPS_PER_SYMBOL):
            if should_stop():
                raise JobStopped()
            body: dict[str, Any] = {"market": market, "symbol": symbol}
            if cursor:
                body["after"] = cursor
            try:
                reply = _call(url, token, body)
            except (http.HttpError, http.HttpConnectionError) as exc:
                errors[symbol] = _describe(exc)
                break
            bars_added += int(reply.get("added") or 0)
            done = bool(reply.get("done"))
            if reply.get("error"):
                errors[symbol] = str(reply["error"])
                done = True
            fraction = 1.0 if done else 0.0
            if not done:
                start, end = _parse(reply.get("from")), _parse(reply.get("cursor"))
                base = base or start
                if base and end and run_started > base:
                    fraction = min(0.95, max(0.0, (end - base) / (run_started - base)))
            progress = (i + fraction) / total
            emit(
                {"symbol_index": i, "bars_added": bars_added, "errors": dict(errors)},
                progress,
                f"Downloading {symbol} ({i + 1} of {total})",
            )
            if done:
                closed = True
                break
            next_cursor = reply.get("cursor")
            if not next_cursor or next_cursor == cursor:
                errors[symbol] = "the coordinator did not move on to newer prices"
                break
            cursor = str(next_cursor)
        else:
            errors[symbol] = "too many download steps for one symbol"
        if not closed:  # failed before a closing line (lost connection, no progress, too many steps)
            emit(
                {"symbol_index": i + 1, "bars_added": bars_added, "errors": dict(errors)},
                (i + 1) / total,
                f"Downloading {symbol} ({i + 1} of {total})",
            )

    if total and len(errors) >= total:
        raise RuntimeError(next(iter(errors.values())))
    summary = f"Downloaded {bars_added:,} new bars for {total} symbols"
    if errors:
        summary += f" · {len(errors)} failed: {', '.join(errors)}"
    return {"summary": summary, "bars_added": bars_added, "errors": errors}
