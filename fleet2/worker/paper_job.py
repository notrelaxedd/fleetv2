"""The paper trade job: run one model live. It never ends on its own (the card says
"Live"); it stops when the owner stops paper trading or the worker shuts down.

Every CHECK_SECONDS it asks the coordinator for the latest bars of the model's market
(kept in memory; the coordinator answers 304 when nothing changed). When a new bar has
closed and it is the model's turn to decide (its rebalance cadence), it computes the
target weights exactly as the backtester would at that bar and posts them to the
coordinator. The coordinator, not the worker, decides whether and how to trade them.
"""
from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from fleet2.common import http
from fleet2.models import get_module
from fleet2.models.base import clean_targets, params_with_defaults
from fleet2.sim.control import JobStopped
from fleet2.sim.marketdata import History, MarketData, from_payload

CHECK_SECONDS = 60.0
EMIT_SECONDS = 4.0


def fetch_if_changed(host_url: str, token: str, market: str, etag: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """(payload, etag), or (None, etag) when the coordinator says nothing changed."""
    headers = {"Authorization": "Bearer " + token}
    if etag:
        headers["If-None-Match"] = etag
    req = urllib.request.Request(f"{host_url}/api/v1/data/bars?market={market}", headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=120) as resp:
            raw = resp.read()
            new_etag = resp.headers.get("ETag")
    except urllib.error.HTTPError as exc:
        if exc.code == 304:
            return None, etag
        raise http.HttpError(exc.code, exc.read().decode("utf-8", "replace")[:300], req.full_url) from None
    except (urllib.error.URLError, OSError) as exc:
        raise http.HttpConnectionError(str(exc)) from None
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw), new_etag


def decision(data: MarketData, module: Any, params: dict[str, Any], first: bool = False) -> tuple[int, dict[str, float]] | None:
    """(decision bar index, weights) when the next bar is a decision bar, else None.
    The next bar to open is index n_bars; the backtester decides at bars where
    (index - start) % every == 0, here anchored at index 0 of the cached history. The
    very first decision of a job is made at once (as a backtest decides at its first
    bar), so a newly started model does not sit in cash until its next turn."""
    n = data.n_bars
    every = max(1, int(module.rebalance_every(params)))
    if n < module.warmup(params) or (n % every != 0 and not first):
        return None
    return n, clean_targets(module.target_positions(History(data, n), params), module.SYMBOLS)


def _when(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%b %d %H:%M UTC")


def run_paper_trade(params: dict[str, Any], checkpoint: dict[str, Any] | None, emit: Any, should_stop: Any) -> dict[str, Any]:
    ctx = params["_context"]
    host, token = str(ctx["host_url"]), str(ctx["worker_token"])
    module = get_module(str(params["module"]))
    model_params = params_with_defaults(module, params.get("params"))
    market = str(params.get("market") or module.MARKET)
    job_id = str(params.get("_job_id") or "")
    data: MarketData | None = None
    etag: str | None = None
    sent_bar = int((checkpoint or {}).get("sent_bar_t") or 0)
    status = "Starting"
    next_check = 0.0
    last_emit = 0.0
    while True:
        if should_stop():
            raise JobStopped()
        now = time.monotonic()
        if now >= next_check:
            next_check = now + CHECK_SECONDS
            try:
                payload, etag = fetch_if_changed(host, token, market, etag)
                if payload is not None:
                    data = from_payload(payload)
                if data is not None:
                    last_t = int(data.times[-1])
                    picked = decision(data, module, model_params, first=sent_bar == 0)
                    if picked is None:
                        status = f"Prices through {_when(last_t)}; not this model's turn to decide"
                    elif last_t > sent_bar:
                        _, weights = picked
                        held = ", ".join(f"{s} {w * 100:.0f}%" for s, w in sorted(weights.items())) or "all cash"
                        http.post_json(f"{host}/api/v1/paper/signal",
                                       {"job_id": job_id, "bar_t": last_t, "targets": weights,
                                        "reason": f"After the bar of {_when(last_t)}: {held}"},
                                       token=token, timeout=10.0)
                        sent_bar = last_t
                        status = f"Decided after {_when(last_t)}: {held}"
            except (http.HttpError, http.HttpConnectionError) as exc:
                status = f"Could not reach the coordinator ({str(exc)[:80]}); trying again in a minute"
        if now - last_emit >= EMIT_SECONDS:
            emit({"sent_bar_t": sent_bar}, None, status)
            last_emit = now
        time.sleep(0.5)
