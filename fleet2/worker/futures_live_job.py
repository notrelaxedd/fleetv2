"""Live trading of one futures model, on Alpaca paper (SPY/QQQ shares standing in for
MES/MNQ) or on Topstep (real contracts). It never ends on its own: the Fleet card says
"Live" until the owner stops it.

Every CHECK_SECONDS it fetches the latest weeks of 1-minute prices of its source from
the coordinator (kept in memory, fetched again only when the ETag changes), cuts today
at the newest closed minute, and replays today through the futures backtester in its
open-end mode. That gives the contracts the model wants from the next minute on, after
its stop, its target and the cut-off, exactly as the backtest would have traded them.
It posts that number whenever a new minute has closed. The coordinator, not the worker,
decides whether and how to trade it, and applies its own safety rules (flat by 15:00
Chicago time, the contract limits, the pause switch, Topstep's loss limit).

params: {"model_id", "module", "params", "market": "futures", "venue", "source",
         "contracts", "rules"}
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from fleet2.common import http
from fleet2.models.futures import module_for
from fleet2.models.futures.base import params_with_defaults
from fleet2.sim import cme_session, topstep
from fleet2.sim import futures_backtest as fb
from fleet2.sim.control import JobStopped
from fleet2.sim.futures_data import FuturesData, PriceCache

CHECK_SECONDS = 10.0
EMIT_SECONDS = 4.0


def newest_closed(data: FuturesData) -> int:
    """One past the newest minute with a real bar of any symbol (0 when none)."""
    real = np.flatnonzero(data.real.any(axis=0))
    return int(real[-1]) + 1 if real.size else 0


def decide(data: FuturesData, module: Any, params: dict[str, Any], contracts: int, rules: topstep.Rules,
           today: int) -> dict[str, Any]:
    """{"minute_t", "contracts": {symbol: n}, "day_pnl", "reason"} from prices up to the
    newest closed minute. Flat when today's prices have not started (or the day ended)."""
    end = newest_closed(data)
    symbol = str(params["symbol"])
    if end == 0:
        return {"minute_t": 0, "contracts": {symbol: 0}, "day_pnl": 0.0, "reason": "no prices yet"}
    cut = data.until_minute(end)
    minute_t = int(cut.times[end - 1])
    if int(cut.days[-1]) != today:
        return {"minute_t": minute_t, "contracts": {symbol: 0}, "day_pnl": 0.0,
                "reason": "no prices for today's session yet"}
    run = fb.run(cut, module, params, topstep.costs_of(rules), contracts, topstep.day_rules(rules),
                 first_day=cut.n_days - 1, open_end=True)
    wanted = {s: int(q) for s, q in (run.final or {}).items()}
    wanted.setdefault(symbol, 0)
    q = wanted[symbol]
    side = "flat" if q == 0 else f"long {q}" if q > 0 else f"short {-q}"
    when = datetime.fromtimestamp(minute_t + 60, timezone.utc).astimezone(cme_session.CHICAGO).strftime("%H:%M")
    return {"minute_t": minute_t, "contracts": wanted, "day_pnl": float(run.pnl[-1]),
            "reason": f"After the {when} Chicago minute: {side} {symbol}"}


def run_futures_live(params: dict[str, Any], checkpoint: dict[str, Any] | None, emit: Any, should_stop: Any) -> dict[str, Any]:
    ctx = params["_context"]
    host, token = str(ctx["host_url"]), str(ctx["worker_token"])
    module = module_for(str(params["module"]), params.get("params"))
    model_params = params_with_defaults(module, params.get("params"))
    rules = topstep.Rules.from_dict(params["rules"])
    contracts = max(1, int(params.get("contracts") or 1))
    source = str(params.get("source") or "alpaca")
    job_id = str(params.get("_job_id") or "")
    cache: PriceCache = params.get("_cache") or PriceCache(ctx)
    sent = int((checkpoint or {}).get("sent_minute_t") or 0)
    status = "Starting"
    next_check = last_emit = 0.0
    while True:
        if should_stop():
            raise JobStopped()
        now = time.monotonic()
        if now >= next_check:
            next_check = now + CHECK_SECONDS
            try:
                data = cache.live(source)
                today = cme_session.as_int(datetime.now(cme_session.CHICAGO).date())
                d = decide(data, module, model_params, contracts, rules, today)
                if d["minute_t"] > sent:
                    http.post_json(f"{host}/api/v1/futures/signal",
                                   {"job_id": job_id, "minute_t": d["minute_t"], "contracts": d["contracts"],
                                    "reason": d["reason"]}, token=token, timeout=10.0)
                    sent = d["minute_t"]
                status = d["reason"] + (f" · today {d['day_pnl']:+,.0f} dollars in the replay" if d["day_pnl"] else "")
            except (http.HttpError, http.HttpConnectionError) as exc:
                status = f"Could not reach the coordinator ({str(exc)[:80]}); trying again shortly"
        if now - last_emit >= EMIT_SECONDS:
            emit({"sent_minute_t": sent}, None, status)
            last_emit = now
        time.sleep(0.5)
