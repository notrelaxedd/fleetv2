"""Futures model search: invent day-trading models, score them on training prices only,
keep the good ones, repeat until stopped.

Each round, for each futures model file, and for recipes (fleet2/models/futures/recipe.py):
- History. For every model file, and every kept recipe being tuned, the job reads that
  family's search log from the coordinator once (GET /api/v1/search/log: every setting
  tried before, by any search, with its TRAINING combine score and why it was thrown
  away). When the log cannot be read the family is searched with an empty history.
- Proposals. Half the candidates come from fleet2.sim.sampler.suggest(SEARCH_SPACE, the
  family's log, n, random.Random(f"{seed}:{round}:{family}:s")), filled in with the
  defaults: random settings while the log has fewer than 20 scores, then mostly settings
  near the ones that scored well and away from those that failed. The other half are
  small changes to the models already kept (fetched from the coordinator at the start of
  the round); with none kept, all come from the sampler. Every candidate can be made
  again from its seed text: "77:3:pullback:s4" is the 5th suggestion of that call (and
  so also depends on the log as it was at that moment), "77:3:pullback:m2
  parent=pullback-s4" a change to that parent. New recipes (random ones and Claude
  Haiku's) have no history: their candidates are draw_params(random.Random(
  f"{seed}:{round}:{family}:{i}")) as before.
- Training numbers (fleet2.worker.futures_jobs.training_numbers), all from training
  prices: the worst of four parts' daily Sharpe ratio at double slippage ("score"); the
  gates: trades on at least 100 different days, profit at normal and at double
  slippage; and, for a candidate that passes the cheap gates, "combine_score": the
  training days run through the same Topstep simulator as the held-out test, at every
  contract size, scored by the expected dollars per Combine attempt at the best size.
  A size that never passes scores combine["floor"], the monthly fees an attempt pays
  when it never passes (about -$147).
- The winner of a family is the candidate that passes every gate with the highest
  combine_score above the floor (the Sharpe score breaks ties). When none beats the
  floor, the family has no winner that round.
- Robustness. The winner is scored again with each setting moved 10% down and 10% up.
  Measured above the floor, the middle (median) neighbour must keep at least half of
  the winner's (combine_score - floor), a neighbour without a combine score counting as
  the floor. Otherwise the winner sat on a sharp peak, which usually means luck, and it
  is dropped.
- Only then is the winner run once on the held-out period, through Topstep's rules
  with coin-flip twins, and sent to the coordinator, which decides whether to keep it
  (train_score is still the Sharpe score; combine is in metrics["train"]).
  Held-out prices are never read before a model is chosen; lockbox prices never.
- Logging. Every candidate of the round goes into its family's search log (POST
  /api/v1/search/log, at most LOG_BATCH per call, once per family per round; a recipe's
  family is its recipe name): score = combine_score (None without one), compact training
  numbers, and why it was thrown away: "gates:<first failed gate>", "error", "no_combine"
  (training too short for an attempt), "floor" (combine_score at the floor), or
  "not_robust" for a winner the robustness check dropped. Robustness neighbours are not
  logged. A log the coordinator cannot take is skipped for the rest of the round.
- The number of settings tried, and their Sharpe ratios, go to the coordinator for the
  "chance this is luck" figure (all recipes are counted together, as "recipe").

Recipes: every round adds a few new recipes (random mixes of building blocks, seeded
like everything else, e.g. "77:3:recipe:1"), each tried with a full set of candidates,
and keeps tuning the recipes of models already kept, with fewer candidates each. When
Claude Haiku has written recipes (coordinator.ai_ideas), the round also takes a few of
those, beside the random ones, and reports every recipe's TRAINING numbers back
(including its best training Combine numbers), so Haiku learns from them. Held-out
numbers are never reported there. Recipes that gave up (params["skip_recipes"]) are
neither tuned nor tried again, whether kept, Haiku's or a random mix.

With several workers, each searches its own share (the coordinator splits them when the
search starts): one takes the recipes, the others split the model files. Model files that
gave up are left out of "files"; a share of no files with "recipes" true searches recipes
only.

Candidates run in a process pool on every core. The training prices are downloaded
once and kept until the coordinator's ETag changes. A stop ends the pool at once.
"""
from __future__ import annotations

import multiprocessing
import os
import random
import statistics
import time
from typing import Any, Callable

import numpy as np

from fleet2.common import http
from fleet2.models.futures import REGISTRY, module_for, recipe, tries_bucket
from fleet2.models.futures.base import (
    draw_params,
    mutate_params,
    neighbours,
    params_with_defaults,
)
from fleet2.sim import sampler, topstep
from fleet2.sim.control import JobStopped
from fleet2.sim.futures_data import FuturesData, PriceCache
from fleet2.worker import futures_jobs
from fleet2.worker.search_job import SearchLog

DEFAULT_CANDIDATES = 16
DEFAULT_NEW_RECIPES = 3  # new random recipes per round
DEFAULT_IDEAS = 3  # Haiku recipes asked for per round (the coordinator may send fewer, or none)
ROBUST_SHARE = 0.5  # the median neighbour must keep at least half the winner's combine score above the floor
LOG_MARKET = "futures"

_STATE: dict[str, Any] = {}


# ------------------------------------------------------------------ proposals


def proposals(seed: int, round_no: int, name: str, count: int, kept: list[dict[str, Any]],
              module: Any = None, trials: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Half new settings, half small changes to kept models (all new when none is kept).
    Each {"name", "params", "seed", "parent"} can be re-created from its seed text (and,
    for a change, the parent's settings). `module` is the recipe, for a recipe.

    New settings: with `trials` (the family's search log, possibly empty) they are
    sampler.suggest(SEARCH_SPACE, trials, n, random.Random(f"{seed}:{round_no}:{name}:s"))
    with the defaults filled in, seed text "...:s<i>" for the i-th suggestion (so they also
    depend on the log as it was). Without (None: a new recipe, no history), each is
    draw_params(random.Random(f"{seed}:{round_no}:{name}:<i>"))."""
    module = module or REGISTRY[name]
    changes = count // 2 if kept else 0
    fresh = count - changes
    out = []
    if trials is None:
        for i in range(fresh):
            text = f"{seed}:{round_no}:{name}:{i}"
            out.append({"name": name, "params": draw_params(module, random.Random(text)), "seed": text, "parent": None})
    else:
        picks = sampler.suggest(module.SEARCH_SPACE, trials, fresh, random.Random(f"{seed}:{round_no}:{name}:s"))
        for i, pick in enumerate(picks):
            out.append({"name": name, "params": params_with_defaults(module, pick),
                        "seed": f"{seed}:{round_no}:{name}:s{i}", "parent": None})
    for j in range(changes):
        parent = kept[j % len(kept)]
        text = f"{seed}:{round_no}:{name}:m{j} parent={parent['id']}"
        out.append({"name": name, "params": mutate_params(module, parent["params"], random.Random(text)),
                    "seed": text, "parent": parent["id"]})
    return out


# ------------------------------------------------------------------ scoring in a process pool


def _init_worker() -> None:
    """Pool workers get a process group of their own: the stop signal the runner sends
    to the job's group reaches only the job, which then ends the pool itself."""
    try:
        os.setpgrp()
    except OSError:
        pass


def score_one(task: dict[str, Any]) -> dict[str, Any]:
    """Training numbers of one candidate (run inside a pool worker; a recipe is rebuilt
    there from the recipe in its settings)."""
    try:
        module = module_for(task["name"], task["params"])
        numbers = futures_jobs.training_numbers(_STATE["data"], module, task["params"], _STATE["rules"])
    except Exception as exc:  # noqa: BLE001 - a bad candidate is reported, not fatal
        return {**task, "score": None, "passes": False, "error": str(exc)[:200]}
    return {**task, **numbers}


class Scorer:
    """Scores candidates on the training prices, in a pool of `processes` (1: in this
    process). The prices reach the pool workers through fork, without copying."""

    def __init__(self, data: FuturesData, rules: topstep.Rules, processes: int) -> None:
        _STATE.update(data=data, rules=rules)
        self.data = data
        self.processes = max(1, int(processes))
        self.pool = None
        if self.processes > 1:
            self.pool = multiprocessing.get_context("fork").Pool(self.processes, initializer=_init_worker)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def score(self, tasks: list[dict[str, Any]], should_stop: Callable[[], bool],
              progress: Callable[[int], None] | None = None) -> list[dict[str, Any]]:
        if self.pool is None:
            out = []
            for task in tasks:
                if should_stop():
                    raise JobStopped()
                out.append(score_one(task))
                if progress:
                    progress(len(out))
            return out
        pending = [self.pool.apply_async(score_one, (t,)) for t in tasks]
        results: list[dict[str, Any] | None] = [None] * len(tasks)
        done = 0
        last = 0.0
        while done < len(tasks):
            if should_stop():
                self.close()
                raise JobStopped()
            for i, p in enumerate(pending):
                if results[i] is None and p.ready():
                    results[i] = p.get()
                    done += 1
            if progress and time.monotonic() - last > 1.0:
                progress(done)
                last = time.monotonic()
            time.sleep(0.02)
        if progress:
            progress(done)
        return [r for r in results if r is not None]


def floor_of(r: dict[str, Any]) -> float | None:
    """The candidate's combine floor (None without combine numbers)."""
    return (r.get("combine") or {}).get("floor")


def above_floor(r: dict[str, Any]) -> bool:
    score, floor = r.get("combine_score"), floor_of(r)
    return score is not None and floor is not None and score > floor


def winner_of(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The round's winner of one family: among candidates that pass every gate and whose
    combine_score beats the floor, the highest combine_score (the Sharpe score breaks
    ties). None when no candidate beats the floor. (A winner needs a Sharpe score too:
    it is the train_score the coordinator keeps models by.)"""
    good = [r for r in rows if r.get("passes") and above_floor(r) and r.get("score") is not None]
    return max(good, key=lambda r: (r["combine_score"], r["score"])) if good else None


def robust(scorer: Scorer, winner: dict[str, Any],
           should_stop: Callable[[], bool]) -> tuple[bool, float | None, float | None]:
    """(kept, median neighbour combine_score, median neighbour Sharpe score): the winner's
    settings moved 10% each way, scored on training. Measured above the fee floor: kept
    when the median neighbour's (combine_score - floor) is at least ROBUST_SHARE x the
    winner's; a neighbour without a combine score (it failed a cheap gate first) counts
    as the floor."""
    module = module_for(winner["name"], winner["params"])
    tasks = [{"name": winner["name"], "params": p, "seed": winner["seed"] + " neighbour", "parent": winner["parent"]}
             for p in neighbours(module, winner["params"])]
    if not tasks:
        return True, None, None
    floor = float(floor_of(winner))
    results = scorer.score(tasks, should_stop)
    above = [r["combine_score"] - floor if r.get("combine_score") is not None else 0.0 for r in results]
    middle = float(statistics.median(above))
    sharpe = float(statistics.median(r["score"] if r.get("score") is not None else -np.inf for r in results))
    return (middle >= ROBUST_SHARE * (winner["combine_score"] - floor), floor + middle,
            sharpe if np.isfinite(sharpe) else None)


def rejection(r: dict[str, Any]) -> str | None:
    """Why a candidate cannot win (None: it could): "error", "gates:<first failed gate>",
    "no_combine" (training too short for one Combine attempt) or "floor"."""
    if r.get("error"):
        return "error"
    failed = [k for k, ok in (r.get("gates") or {}).items() if not ok]
    if failed or not r.get("passes"):
        return "gates:" + (failed[0] if failed else "unknown")
    if r.get("combine_score") is None or floor_of(r) is None:
        return "no_combine"
    if not above_floor(r):
        return "floor"
    return None


def log_trial(r: dict[str, Any], rejected: str | None = None) -> dict[str, Any]:
    """One candidate as a search-log trial: its settings (a recipe's own recipe left out:
    the family names it), score = combine_score, why it was thrown away (`rejected`
    overrides), and compact training numbers (no daily P&L, no table of sizes)."""
    combine = r.get("combine")
    metrics: dict[str, Any] = {k: r.get(k) for k in ("score", "sharpe", "days_traded", "pnl_double", "pnl_normal")}
    metrics["combine"] = {k: v for k, v in combine.items() if k != "sizes"} if isinstance(combine, dict) else None
    if r.get("error"):
        metrics["error"] = str(r["error"])[:200]
    return {"params": {k: v for k, v in r["params"].items() if k != "recipe"}, "score": r.get("combine_score"),
            "rejected": rejected or rejection(r), "metrics": metrics}


# ------------------------------------------------------------------ the job


def _valid(raw: Any) -> bool:
    try:
        recipe.validate(raw)
        return True
    except recipe.BadRecipe:
        return False


def _kept(host: str, token: str, should_stop: Callable[[], bool] | None = None) -> dict[str, list[dict[str, Any]]]:
    rows = http.with_retries(lambda: http.get_json(f"{host}/api/v1/models/kept?market=futures", token=token,
                                                   timeout=30.0), should_stop=should_stop) or []
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row["module"]), []).append({"id": str(row["id"]), "params": dict(row["params"])})
    return out


def tuned(files: list[str], kept: dict[str, list[dict[str, Any]]], skip: set[str] | frozenset[str] = frozenset()) -> list[str]:
    """The families whose search log a round reads: every model file, and every kept
    recipe still being tuned (not given up)."""
    return list(files) + sorted(n for n in kept if recipe.is_recipe(n) and n not in skip)


def families(seed: int, round_no: int, files: list[str], per_file: int, new_recipes: int,
             kept: dict[str, list[dict[str, Any]]],
             ideas: list[dict[str, Any]] | None = None,
             logs: dict[str, list[dict[str, Any]]] | None = None,
             skip: set[str] | frozenset[str] = frozenset()) -> tuple[list[dict[str, Any]], list[str]]:
    """This round's candidates, and the model files and recipes they belong to: every
    file; every recipe that has kept models (fewer candidates each); Haiku's recipes
    (`ideas`, a full set each); and `new_recipes` new random recipes (a full set each,
    none already in the round). `logs` holds each tuned family's search log ([] when it
    could not be read); recipes named in `skip` gave up and are left out."""
    logs = logs or {}
    tasks: list[dict[str, Any]] = []
    names: list[str] = []
    for name in files:
        tasks += proposals(seed, round_no, name, per_file, kept.get(name, []), trials=logs.get(name, []))
        names.append(name)
    for name in sorted(n for n in kept if recipe.is_recipe(n) and n not in skip):
        module = module_for(name, kept[name][0]["params"])
        tasks += proposals(seed, round_no, name, max(2, per_file // 4), kept[name], module, trials=logs.get(name, []))
        names.append(name)
    for idea in ideas or []:
        try:
            module = recipe.family(idea["recipe"])
        except (recipe.BadRecipe, KeyError, TypeError):
            continue
        if module.__name__ in names or module.__name__ in skip:
            continue
        tasks += proposals(seed, round_no, module.__name__, per_file, [], module)
        names.append(module.__name__)
    for i in range(new_recipes):
        mix = recipe.random_recipe(random.Random(f"{seed}:{round_no}:recipe:{i}"))
        module = recipe.family(mix)
        if module.__name__ in names or module.__name__ in skip:
            continue
        tasks += proposals(seed, round_no, module.__name__, per_file, [], module)
        names.append(module.__name__)
    return tasks, names


def _ideas(host: str, token: str, job_id: Any, count: int) -> list[dict[str, Any]]:
    """Haiku's waiting recipes for this round ([] when there are none, or on any problem:
    the search never waits for them)."""
    if count <= 0:
        return []
    try:
        got = http.post_json(f"{host}/api/v1/search/ideas", {"job_id": job_id, "take": count}, token=token, timeout=30.0)
    except Exception:  # noqa: BLE001 - an older coordinator, or a hiccup: carry on with random recipes
        return []
    return [i for i in (got or []) if isinstance(i, dict) and "recipe" in i and "id" in i]


def recipe_results(results: list[dict[str, Any]], names: list[str],
                   idea_of: dict[str, int]) -> list[dict[str, Any]]:
    """Per recipe in the round: how many settings were tried, the best one's TRAINING
    numbers (score, gates, days traded, profit at double and normal costs), and the
    TRAINING Combine numbers of the setting with the best combine_score (its score, best
    contract size, pass and timeout rates on the training days)."""
    out = []
    for name in names:
        if not recipe.is_recipe(name):
            continue
        rows = [r for r in results if r["name"] == name]
        if not rows:
            continue
        scored = [r for r in rows if r.get("score") is not None]
        best = max(scored, key=lambda r: r["score"]) if scored else rows[0]
        combined = [r for r in rows if r.get("combine_score") is not None]
        top = max(combined, key=lambda r: r["combine_score"]) if combined else None
        combine = (top or {}).get("combine") or {}
        out.append({"name": name, "recipe": best["params"]["recipe"], "idea_id": idea_of.get(name),
                    "tried": len(rows), "best_score": best.get("score"),
                    "passes": any(r.get("passes") for r in rows), "days_traded": best.get("days_traded"),
                    "pnl_double": best.get("pnl_double"), "pnl_normal": best.get("pnl_normal"),
                    "combine_score": top["combine_score"] if top else None, "combine_measure": combine.get("measure"),
                    "best_size": combine.get("best_size"),
                    "pass_rate": combine.get("pass_rate"), "timeout_rate": combine.get("timeout_rate")})
    return out


def run_futures_search(params: dict[str, Any], checkpoint: dict[str, Any] | None, emit: Any, should_stop: Any) -> dict[str, Any]:
    ctx = params["_context"]
    host, token = str(ctx["host_url"]), str(ctx["worker_token"])
    rules = topstep.Rules.from_dict(params["rules"])
    if rules.missing_trading_fees():
        raise RuntimeError(topstep.FEE_MESSAGE)
    periods = params["periods"]
    seed = int(params.get("seed") or 1)
    per_file = max(2, int(params.get("candidates") or DEFAULT_CANDIDATES))
    # This worker's share of the search (coordinator.futures_models.shares): some model
    # files, recipes or both. A file the worker no longer knows is skipped; an empty
    # share (after an update) falls back to everything.
    files = [n for n in (params["files"] if "files" in params else sorted(REGISTRY)) if n in REGISTRY]
    with_recipes = bool(params.get("recipes", True))
    skip = frozenset(str(n) for n in params.get("skip_recipes") or [])
    if not files and not with_recipes:
        files, with_recipes = sorted(REGISTRY), True
    new_recipes = max(0, int(params.get("new_recipes", DEFAULT_NEW_RECIPES))) if with_recipes else 0
    ideas_per_round = max(0, int(params.get("ideas", DEFAULT_IDEAS))) if with_recipes else 0
    processes = int(params.get("processes") or os.cpu_count() or 1)
    cache: PriceCache = params.get("_cache") or PriceCache(ctx)
    round_no = int((checkpoint or {}).get("round") or 0)
    kept_total = int((checkpoint or {}).get("kept") or 0)
    tried_total = int((checkpoint or {}).get("tried") or 0)
    scorer: Scorer | None = None
    try:
        while True:
            round_no += 1
            state = {"round": round_no, "kept": kept_total, "tried": tried_total}
            emit(state, 0.0, f"Round {round_no}: loading futures prices")
            train = cache.get("train")
            if train.days.size and train.days[-1] > futures_jobs.day_int(periods["train_end"]):
                raise RuntimeError("the coordinator sent prices past the end of training")
            if scorer is None or scorer.data is not train:
                if scorer is not None:
                    scorer.close()
                scorer = Scorer(train, rules, processes)
            kept = _kept(host, token, should_stop)
            if not with_recipes:  # another worker tunes the kept recipes
                kept = {n: v for n, v in kept.items() if not recipe.is_recipe(n)}
            ideas = _ideas(host, token, params.get("_job_id"), ideas_per_round)
            idea_of = {recipe.name_of(i["recipe"]): int(i["id"]) for i in ideas if _valid(i["recipe"])}
            search_log = SearchLog(host, token, params.get("_job_id"), should_stop)
            logs = {name: search_log.trials(LOG_MARKET, name) for name in tuned(files, kept, skip)}
            tasks, names = families(seed, round_no, files, per_file, new_recipes, kept, ideas, logs, skip)

            def progress(done: int, total: int = len(tasks)) -> None:
                emit({"round": round_no, "kept": kept_total, "tried": tried_total + done}, 0.8 * done / total,
                     f"Round {round_no}: trying settings {done} of {total} · {kept_total} kept so far")

            results = scorer.score(tasks, should_stop, progress)
            tried_total += len(results)
            tries: dict[str, dict[str, float]] = {}
            for r in results:
                t = tries.setdefault(tries_bucket(r["name"]), {"n": 0, "sum": 0.0, "sq": 0.0})
                sr = float(r.get("sr_day") or 0.0)
                t["n"] += 1
                t["sum"] += sr
                t["sq"] += sr * sr
            # Every call to the coordinator is tried again for a few minutes when it does
            # not answer (a restart), so a long search is not lost to a short outage.
            tried_body = {"job_id": params.get("_job_id"), "tries": tries,
                          "recipes": recipe_results(results, names, idea_of)}
            http.with_retries(lambda: http.post_json(f"{host}/api/v1/search/tries", tried_body, token=token,
                                                     timeout=30.0), should_stop=should_stop)
            for name in names:
                rows = [r for r in results if r["name"] == name]
                winner = winner_of(rows)
                ok, middle, middle_sharpe = False, None, None
                if winner is not None:
                    module = module_for(name, winner["params"])
                    emit({"round": round_no, "kept": kept_total, "tried": tried_total}, 0.85,
                         f"Round {round_no}: checking {module.NAME} settings 10% either way")
                    ok, middle, middle_sharpe = robust(scorer, winner, should_stop)
                search_log.post(LOG_MARKET, name, [log_trial(r, "not_robust" if r is winner and not ok else None)
                                                   for r in rows])
                if winner is None or not ok:
                    continue
                emit({"round": round_no, "kept": kept_total, "tried": tried_total}, 0.9,
                     f"Round {round_no}: held-out test of a {module.NAME} find")
                trained = futures_jobs.training_numbers(train, module, winner["params"], rules, keep_daily=True)
                held = cache.get("held_out")
                result = futures_jobs.held_out_numbers(held, module, winner["params"], rules, periods,
                                                       trained["worst_stretch"], should_stop)
                trained["neighbour_median"] = middle_sharpe
                trained["neighbour_combine_median"] = middle
                found_body = {
                    "job_id": params.get("_job_id"), "module": name, "params": winner["params"],
                    "train_score": winner["score"], "seed": winner["seed"], "parent": winner["parent"],
                    "idea_id": idea_of.get(name),
                    "metrics": {"market": "futures", "feed": held.feed, "periods": periods, "params": winner["params"],
                                "train": trained, "held_out": result,
                                "summary": futures_jobs.summary_line(trained, result)},
                }
                reply = http.with_retries(lambda: http.post_json(f"{host}/api/v1/models/futures", found_body,
                                                                 token=token, timeout=60.0), should_stop=should_stop)
                if isinstance(reply, dict) and reply.get("kept"):
                    kept_total += 1
            emit({"round": round_no, "kept": kept_total, "tried": tried_total}, 1.0,
                 f"Round {round_no} done · {tried_total:,} settings tried · {kept_total} kept")
    finally:
        if scorer is not None:
            scorer.close()
