"""The model search job: invent new models, backtest them, keep the good ones, repeat
until stopped.

- A new model is one of the model files with new settings. Each round, for each model
  file, the job first reads that file's search log from the coordinator (GET
  /api/v1/search/log: the settings tried before, by every search, with their TRAINING
  scores and why each was thrown away). Its candidates are then
  fleet2.sim.sampler.suggest(SEARCH_SPACE, that log, candidates per file,
  random.Random(f"{seed}:{round}:{file}")), filled in with the file's defaults. With fewer
  than 20 scored settings in the log that is plain random settings; after that most are
  drawn near the settings that scored well and away from those that failed. Candidate i
  is re-created exactly from its seed text f"{seed}:{round}:{file}:{i}" and the log as it
  was at that moment. When the log cannot be read the search carries on with an empty one.
- Each candidate is backtested on the TRAINING period only and scored by
  fleet2.sim.robust.score: its yearly return minus buy and hold's, which does not grow
  with the number of trades. It must have made at least 100 trades (MIN_TRADES) and
  score above zero. The search never looks at the held-out period when it chooses.
- The best candidate of each file in a round is then checked on the training period by
  robust.report (training split in 4 parts, settings moved 10% either way, double costs,
  coin-flip twins; seeded per round and file). Stop interrupts it like any backtest.
- Only then is it run once on the held-out period, for the dashboard to rank it, and sent
  to the coordinator (POST /api/v1/models) with metrics["robustness"] = the report and
  train_score = the report's score (the WORST of the 4 training parts: harder to reach by
  luck than the whole period's, and the futures search's rule too; the whole-period score
  is in metrics["search_score"]). A winner that fails the checks is still sent: the
  Models screen shows it unranked with its flags, and the coordinator counts it weaker
  than any model that passed. Held-out results never feed back into what is chosen.
- Every candidate goes into the search log (POST /api/v1/search/log, one call per file
  per round): its settings, score, compact training numbers (no curves) and why it was
  thrown away: "too_few_trades", "not_positive", "error" (the backtest failed), or
  "robustness:<flags>" for a winner that failed the checks. A log that cannot be written
  is skipped for the rest of the round, never the search.
- Model files that gave up (params["skip_files"], set by the coordinator from the log)
  are left out. Rounds repeat until the owner presses Stop model search.

params: {"markets": ["stocks", "crypto"], "seed": int, "candidates": int per file per round,
         "limits": {...}, "held_out_fraction": float, "skip_files": [...]}
"""
from __future__ import annotations

import json
import logging
import random
from typing import Any, Callable
from urllib.parse import urlencode

from fleet2.common import http
from fleet2.models import REGISTRY
from fleet2.models.base import params_with_defaults
from fleet2.sim import robust, sampler
from fleet2.sim.backtest import Limits, run_backtest, split_at
from fleet2.sim.control import JobStopped
from fleet2.sim.marketdata import MarketData, load
from fleet2.sim.metrics import MIN_TRADES, summarize
from fleet2.universe import HELD_OUT_FRACTION, MARKETS

log = logging.getLogger(__name__)

DEFAULT_CANDIDATES = 12
COIN_FLIPS = robust.COIN_FLIP_DRAWS  # coin-flip twins per robustness report

# The search log is help, not a must: its calls wait out a short restart, then give up
# for the rest of the round (the other coordinator calls wait several minutes).
LOG_PAUSES = (5.0, 15.0, 30.0)
LOG_BATCH = 200                 # trials per POST (the coordinator takes up to 1000)
LOG_BATCH_BYTES = 256 * 1024    # ... and about this much JSON (its request limit is 1 MiB)
COMPACT_KEYS = ("roi", "benchmark_roi", "vs_buy_and_hold", "max_drawdown", "sharpe", "win_rate", "profit_factor",
                "trades", "avg_hold_s", "years", "t_stat", "costs_paid", "start", "end")


# ------------------------------------------------------------------ the search log (shared with the futures search)


def batches(trials: list[dict[str, Any]], size: int = LOG_BATCH,
            max_bytes: int = LOG_BATCH_BYTES) -> list[list[dict[str, Any]]]:
    """The trials cut into POST-sized pieces: at most `size` each and about `max_bytes` of JSON."""
    out: list[list[dict[str, Any]]] = []
    part: list[dict[str, Any]] = []
    used = 0
    for t in trials:
        n = len(json.dumps(t, default=str))
        if part and (len(part) >= size or used + n > max_bytes):
            out.append(part)
            part, used = [], 0
        part.append(t)
        used += n
    if part:
        out.append(part)
    return out


class SearchLog:
    """The coordinator's search log as one search round uses it. Reading gives a family's
    past trials ([] when there are none or the log cannot be read); writing sends a
    round's trials in batches. After a call fails (once its short retries are spent),
    the log is left alone for the rest of the round, so an outage costs a round of
    learning, never the search. Make a new one each round."""

    def __init__(self, host: str, token: str, job_id: Any, should_stop: Callable[[], bool] | None = None) -> None:
        self.host, self.token, self.job_id, self.should_stop = host, token, job_id, should_stop
        self.down = False

    def _call(self, what: str, call: Callable[[], Any]) -> Any:
        if self.down:
            return None
        try:
            return http.with_retries(call, pauses=LOG_PAUSES, should_stop=self.should_stop)
        except Exception as exc:  # noqa: BLE001 - an outage, an older coordinator: carry on without the log
            self.down = True
            log.warning("search log: %s failed (%s); carrying on without it this round", what, exc)
            return None

    def trials(self, market: str, family: str) -> list[dict[str, Any]]:
        query = urlencode({"market": market, "family": family})
        got = self._call(f"reading {family}", lambda: http.get_json(f"{self.host}/api/v1/search/log?{query}",
                                                                     token=self.token, timeout=30.0))
        rows = got.get("trials") if isinstance(got, dict) else None
        return [t for t in rows or [] if isinstance(t, dict)]

    def post(self, market: str, family: str, trials: list[dict[str, Any]]) -> int:
        """Send the trials; returns how many were stored."""
        stored = 0
        for part in batches(trials):
            body = {"job_id": self.job_id, "market": market, "family": family, "trials": part}
            got = self._call(f"writing {family}", lambda body=body: http.post_json(
                f"{self.host}/api/v1/search/log", body, token=self.token, timeout=30.0))
            if got is None:
                break
            stored += int(got.get("stored") or 0) if isinstance(got, dict) else 0
        return stored


# ------------------------------------------------------------------ candidates and scoring


def candidates(seed: int, round_no: int, name: str, count: int, trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """This round's settings for one model file: the sampler's suggestions given the
    file's log (`trials`), with the file's defaults filled in. Candidate i has the seed
    text f"{seed}:{round_no}:{name}:{i}"; the same seed, round and log give the same list."""
    module = REGISTRY[name]
    rng = random.Random(f"{seed}:{round_no}:{name}")
    return [params_with_defaults(module, p) for p in sampler.suggest(module.SEARCH_SPACE, trials, count, rng)]


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    """A backtest's numbers for the search log: no curve."""
    return {k: metrics.get(k) for k in COMPACT_KEYS}


def rejection(metrics: dict[str, Any], score: float) -> str | None:
    """Why a candidate cannot win its round (None: it can)."""
    if int(metrics.get("trades") or 0) < MIN_TRADES:
        return "too_few_trades"
    if not score > 0:
        return "not_positive"
    return None


def evaluate(data: MarketData, name: str, params: dict[str, Any], limits: Limits, split: int, held_out_fraction: float,
             should_stop: Any, period: str) -> dict[str, Any]:
    """One backtest of one candidate on the training or the held-out period."""
    module = REGISTRY[name]
    spec = MARKETS[data.market]
    if period == "train":
        start, stop = robust.training_bounds(module, params, split)
    else:
        start, stop = split, data.n_bars
    run = run_backtest(data, module, params, start, stop, limits, spec["benchmark"], spec["bars_per_year"],
                       None, should_stop)
    return summarize(run)


def robustness(data: MarketData, name: str, params: dict[str, Any], limits: Limits, split: int, seed: int,
               round_no: int, should_stop: Any, progress: Callable[[float], None] | None) -> dict[str, Any]:
    """robust.report() of a round's winner on its training bars, seeded by the search's
    seed, the round and the file. A check that cannot run (training too short to cut in
    parts, say) fails with the flag "error"; a stop still stops."""
    module = REGISTRY[name]
    start, stop = robust.training_bounds(module, params, split)
    check_seed = random.Random(f"{seed}:{round_no}:{name}:robust").randrange(2**31)
    try:
        return robust.report(data, module, params, start, stop, limits, should_stop, search_space=module.SEARCH_SPACE,
                             n_draws=COIN_FLIPS, seed=check_seed, progress=progress)
    except JobStopped:
        raise
    except Exception as exc:  # noqa: BLE001 - reported on the model, not fatal
        return {"score": None, "flags": ["error"], "passed": False, "error": str(exc)[:300], "seed": check_seed}


# ------------------------------------------------------------------ the job


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
    skip = set(params.get("skip_files") or [])
    files = [n for n, m in REGISTRY.items() if m.MARKET in markets and n not in skip]
    if not files:
        raise RuntimeError("every model file of these markets gave up: new model ideas are needed, or reset the log")
    while True:
        round_no += 1
        state = {"round": round_no, "kept": kept_total, "tried": tried_total}
        emit(state, 0.0, f"Round {round_no}: loading prices")
        data = {m: load(ctx, m) for m in markets}
        search_log = SearchLog(host, token, params.get("_job_id"), should_stop)
        total = len(files) * per_file
        done = 0
        for name in files:
            module = REGISTRY[name]
            market = module.MARKET
            split = split_at(data[market], (params.get("held_out_start_t") or {}).get(market), held_out_fraction)
            cands = candidates(seed, round_no, name, per_file, search_log.trials(market, name))
            trials: list[dict[str, Any]] = []
            best: tuple[float, int, dict[str, Any], dict[str, Any]] | None = None
            for i, cand in enumerate(cands):
                if should_stop():
                    raise JobStopped()
                try:
                    train = evaluate(data[market], name, cand, limits, split, held_out_fraction, should_stop, "train")
                except JobStopped:
                    raise
                except Exception as exc:  # noqa: BLE001 - a candidate that cannot run is logged, not fatal
                    trials.append({"params": cand, "score": None, "rejected": "error",
                                   "metrics": {"error": str(exc)[:200]}})
                else:
                    s = robust.score(train)
                    why = rejection(train, s)
                    trials.append({"params": cand, "score": s, "rejected": why, "metrics": compact(train)})
                    if why is None and (best is None or s > best[0]):
                        best = (s, i, cand, train)
                tried_total += 1
                done += 1
                emit({"round": round_no, "kept": kept_total, "tried": tried_total}, done / total,
                     f"Round {round_no}: trying {module.NAME} settings {i + 1} of {len(cands)} "
                     f"· {kept_total} kept so far")
            report = None
            if best is not None:
                s, i, cand, train = best
                state = {"round": round_no, "kept": kept_total, "tried": tried_total}

                def progress(fraction: float, state: dict[str, Any] = state, title: str = module.NAME) -> None:
                    emit(state, done / total, f"Round {round_no}: checking {title} against coin flips "
                                              f"({fraction:.0%})")

                progress(0.0)
                report = robustness(data[market], name, cand, limits, split, seed, round_no, should_stop, progress)
                if not report.get("passed"):
                    trials[i]["rejected"] = "robustness:" + (",".join(report.get("flags") or []) or "not_positive")
            search_log.post(market, name, trials)
            if best is None or report is None:
                continue
            s, i, cand, train = best
            held = evaluate(data[market], name, cand, limits, split, held_out_fraction, should_stop, "held_out")
            found_body = {
                "job_id": params.get("_job_id"), "module": name, "params": cand,
                "train_score": report["score"] if report.get("score") is not None else s,
                "seed": f"{seed}:{round_no}:{name}:{i}",
                "metrics": {"train": train, "held_out": held, "split_t": int(data[market].times[split]),
                            "feed": data[market].feed, "params": cand, "market": market,
                            "search_score": s, "robustness": report},
            }
            reply = http.with_retries(lambda: http.post_json(f"{host}/api/v1/models", found_body, token=token,
                                                             timeout=30.0), should_stop=should_stop)
            if isinstance(reply, dict) and reply.get("kept"):
                kept_total += 1
