"""The model search job: invent new models, backtest them, keep the good ones, repeat
until stopped. The same mechanism as v1 (polymarket-fleet fleet/sim/search.py):

- A new model is one of the model files with new settings. Candidate i of round r draws
  its settings from random.Random(f"{seed}:{r}:{file}:{i}") over the file's
  SEARCH_SPACE, so every run can be repeated exactly. No AI service is involved.
- Each candidate is backtested on the TRAINING period only, and scored by its "shrunk"
  return, roi x trades / (trades + 100) (v1's rule: a result from few trades counts
  for less). The search never looks at the held-out period when it chooses.
- The best candidate of each model file in a round is kept when its training score is
  positive and it made at least 100 trades in training. Only then is it run once on
  the held-out period, for the dashboard to rank it; that result never feeds back into
  what the search keeps. The coordinator stores it (POST /api/v1/models).
- Rounds repeat until the owner presses Stop model search.

params: {"markets": ["stocks", "crypto"], "seed": int, "candidates": int per file per round,
         "limits": {...}, "held_out_fraction": float}
"""
from __future__ import annotations

import random
from typing import Any

from fleet2.common import http
from fleet2.models import REGISTRY
from fleet2.models.base import draw_params, params_with_defaults
from fleet2.sim.backtest import Limits, run_backtest, split_index
from fleet2.sim.control import JobStopped
from fleet2.sim.marketdata import MarketData, load
from fleet2.sim.metrics import MIN_TRADES, summarize
from fleet2.universe import HELD_OUT_FRACTION, MARKETS

DEFAULT_CANDIDATES = 12


def score(metrics: dict[str, Any]) -> float:
    """Shrunk return (v1): roi x trades / (trades + 100)."""
    trades = int(metrics.get("trades") or 0)
    return float(metrics.get("roi") or 0.0) * trades / (trades + 100)


def candidate(seed: int, round_no: int, name: str, index: int) -> dict[str, Any]:
    module = REGISTRY[name]
    return params_with_defaults(module, draw_params(module, random.Random(f"{seed}:{round_no}:{name}:{index}")))


def evaluate(data: MarketData, name: str, params: dict[str, Any], limits: Limits, split: int, held_out_fraction: float,
             should_stop: Any, period: str) -> dict[str, Any]:
    """One backtest of one candidate on the training or the held-out period."""
    module = REGISTRY[name]
    spec = MARKETS[data.market]
    if period == "train":
        start, stop = min(module.warmup(params), split - 1), split
    else:
        start, stop = split, data.n_bars
    run = run_backtest(data, module, params, start, stop, limits, spec["benchmark"], spec["bars_per_year"],
                       None, should_stop)
    return summarize(run)


def run_search(params: dict[str, Any], checkpoint: dict[str, Any] | None, emit: Any, should_stop: Any) -> dict[str, Any]:
    ctx = params["_context"]
    host, token = str(ctx["host_url"]), str(ctx["worker_token"])
    markets = [m for m in (params.get("markets") or ["stocks", "crypto"]) if m in MARKETS]
    seed = int(params.get("seed") or 1)
    per_file = max(1, int(params.get("candidates") or DEFAULT_CANDIDATES))
    raw = params.get("limits") or {}
    limits = Limits(float(raw.get("money", 10_000)), float(raw.get("max_per_position", 1_000)),
                    float(raw.get("max_per_model", 10_000)))
    held_out_fraction = float(params.get("held_out_fraction", HELD_OUT_FRACTION))
    round_no = int((checkpoint or {}).get("round") or 0)
    kept_total = int((checkpoint or {}).get("kept") or 0)
    tried_total = int((checkpoint or {}).get("tried") or 0)
    files = [n for n, m in REGISTRY.items() if m.MARKET in markets]
    while True:
        round_no += 1
        state = {"round": round_no, "kept": kept_total, "tried": tried_total}
        emit(state, 0.0, f"Round {round_no}: loading prices")
        data = {m: load(ctx, m) for m in markets}
        total = len(files) * per_file
        done = 0
        for name in files:
            market = REGISTRY[name].MARKET
            split = split_index(data[market], held_out_fraction)
            best: tuple[float, dict[str, Any], dict[str, Any]] | None = None
            for i in range(per_file):
                if should_stop():
                    raise JobStopped()
                cand = candidate(seed, round_no, name, i)
                train = evaluate(data[market], name, cand, limits, split, held_out_fraction, should_stop, "train")
                tried_total += 1
                done += 1
                s = score(train)
                if train["trades"] >= MIN_TRADES and s > 0 and (best is None or s > best[0]):
                    best = (s, cand, train)
                emit({"round": round_no, "kept": kept_total, "tried": tried_total}, done / total,
                     f"Round {round_no}: trying {REGISTRY[name].NAME} settings {i + 1} of {per_file} "
                     f"· {kept_total} kept so far")
            if best is None:
                continue
            s, cand, train = best
            held = evaluate(data[market], name, cand, limits, split, held_out_fraction, should_stop, "held_out")
            reply = http.post_json(f"{host}/api/v1/models", {
                "job_id": params.get("_job_id"), "module": name, "params": cand, "train_score": s,
                "seed": f"{seed}:{round_no}",
                "metrics": {"train": train, "held_out": held, "split_t": int(data[market].times[split]),
                            "feed": data[market].feed, "params": cand, "market": market},
            }, token=token, timeout=30.0)
            if isinstance(reply, dict) and reply.get("kept"):
                kept_total += 1
