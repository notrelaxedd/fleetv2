"""Model search on the coordinator: start and stop search jobs, store the models they
find, and the status line for the Models screen.

Start model search sends one search job to every idle worker (at least one job, which
waits for a free worker otherwise), each with its own seed so they never try the same
settings. A futures search also gives each worker its own share of the strategies
(futures_models.shares). Stop model search cancels them all.

Found models are kept sparingly so the list stays readable: at most KEEP_PER_FILE
search models per model file stay active. A new find replaces the weakest one only when
it is better: a stock or crypto model that failed the robustness checks is weaker than
any that passed them, and otherwise the higher TRAINING score wins (store_found);
held-out results never decide what is kept (they are for ranking on the Models screen).

The search log (table search_log) holds every setting a search tried, kept or not, with
its TRAINING score and why it was rejected, so the searches can learn from their mistakes
and stop spending hours on a model file or recipe that keeps failing ("gave up", below).
It is shared by the stock/crypto search and the futures search.

Giving up: a family (a model file, or a futures recipe) gave up when
 (a) it has GIVE_UP_TRIES (2,000) logged trials and none of them has a positive score; or
 (b) futures only: at least GIVE_UP_TESTED (5) of its search models reached the held-out
     test and none of them has an expected net per Combine attempt clearly above the
     fee floor (what a model that never passes pays in monthly fees) or beats its
     coin-flip twin. Never decided while the fees are unset.
Rule (b) uses held-out OUTCOMES only to stop spending compute on a whole family. It never
chooses, scores or keeps a setting, and the once-only lockbox Final check is not read or
touched by it: that is what keeps the held-out and lockbox results honest.
Resetting the log (CLI: reset-search-log) also lets a family that gave up by rule (b) try
again: only models tested after the reset count.
"""
from __future__ import annotations

import logging
import math
import secrets
import time
import uuid
from datetime import datetime
from typing import Any, Literal

import psycopg
from psycopg.types.json import Jsonb

from coordinator import queue
from coordinator.errors import BadRequest, Conflict
from coordinator.limits import Limits
from coordinator.models import held_out_start
from fleet2.models import get_module
from fleet2.universe import HELD_OUT_FRACTION

log = logging.getLogger(__name__)

KEEP_PER_FILE = 5
GIVE_UP_TRIES = 2_000   # logged trials without a positive score before a family gives up
GIVE_UP_TESTED = 5      # futures: held-out tested models, none of them promising, before it gives up
FLOOR_MARGIN = 0.10     # futures: beating the fee floor means being this share of it better
LOG_MARKETS = ("stocks", "crypto", "futures")
MAX_TRIALS_PER_CALL = 1000
MAX_ROWS = 50_000       # logged rows kept per (market, family)
TOP_TRIALS = 200        # what GET /api/v1/search/log returns: the best ones ...
RECENT_TRIALS = 300     # ... and the latest ones
PRUNE_EVERY = 600.0     # seconds between prunes in the coordinator loop
RESET_KEY = "search_log_reset"
SEARCH_STATUSES = ("queued", "leased", "cancel_requested")


def running_jobs(conn: psycopg.Connection) -> list[dict[str, Any]]:
    return conn.execute("SELECT * FROM jobs WHERE kind = 'model_search' AND status = ANY(%s)",
                        (list(SEARCH_STATUSES),)).fetchall()


def start(conn: psycopg.Connection, limits: Limits, markets: list[str] | None = None, target: str = "all_idle",
          rules: Any = None) -> dict[str, Any]:
    if running_jobs(conn):
        raise Conflict("Model search is already running")
    markets = markets or ["stocks", "crypto"]
    if markets == ["futures"]:
        from coordinator import futures_models

        return _start(conn, futures_models.search_params(conn, rules), target, "Futures model search")
    skip = skipped_files(conn, markets)
    params = {
        "markets": markets,
        "skip_files": skip,
        "held_out_start_t": {m: held_out_start(conn, m) for m in markets},
        "candidates": 12,
        "limits": {"money": limits.starting_balance_per_model, "max_per_position": limits.max_per_position,
                   "max_per_model": limits.max_per_model},
        "held_out_fraction": HELD_OUT_FRACTION,
    }
    return _start(conn, params, target, "Model search")


def _start(conn: psycopg.Connection, params: dict[str, Any], target: str, label: str) -> dict[str, Any]:
    """One search job per idle worker (each with its own seed), or one job that waits."""
    if target == "all_idle" and not queue.idle_workers(conn):
        target = "auto"
    if target == "all_idle":
        workers = queue.idle_workers(conn)
        futures = params.get("markets") == ["futures"]
        if futures:  # each worker searches its own share of the strategies
            from coordinator import futures_models

            parts = futures_models.shares(len(workers), params.get("files"))
        jobs = []
        for i, w in enumerate(workers):
            job_params = {**params, "seed": secrets.randbelow(10**9), **(parts[i] if futures else {})}
            jobs += queue.create_job(conn, "model_search", job_params, w["id"]).jobs
        message = f"{label} started on {len(jobs)} worker{'s' if len(jobs) != 1 else ''}"
        if futures and len(jobs) > 1:
            message += ", each on different strategies"
        return {"jobs": jobs, "message": message}
    result = queue.create_job(conn, "model_search", {**params, "seed": secrets.randbelow(10**9)}, target)
    where = "waits for a free worker" if result.waiting else "started"
    return {"jobs": result.jobs, "message": f"{label} {where}"}


def stop(conn: psycopg.Connection) -> dict[str, Any]:
    jobs = running_jobs(conn)
    if not jobs:
        raise Conflict("Model search is not running")
    for job in jobs:
        queue.cancel_job(conn, job["id"])
    return {"message": "Model search stopped"}


def search_status(conn: psycopg.Connection) -> dict[str, Any]:
    """{"running": bool, "button": "Start model search" | "Stop model search", "detail": str | None}."""
    jobs = running_jobs(conn)
    found = conn.execute("SELECT count(*) AS n FROM models WHERE origin = 'search' AND status IS DISTINCT FROM 'retired'").fetchone()["n"]
    if not jobs:
        return {"running": False, "button": "Start model search",
                "detail": f"{found} model{'s' if found != 1 else ''} found by search so far" if found else None}
    active = sum(1 for j in jobs if j["status"] == "leased")
    tried = sum(int((j["checkpoint"] or {}).get("tried") or 0) for j in jobs)
    detail = f"Searching on {active} worker{'s' if active != 1 else ''}" if active else "Waiting for a free worker"
    detail += f" · {tried:,} settings tried · {found} kept"
    gave_up = _gave_up_count(jobs)
    if gave_up:
        detail += f" · {gave_up} model file{'s' if gave_up != 1 else ''} gave up"
    return {"running": True, "button": "Stop model search", "detail": detail, "shares": _shares(conn, jobs)}


def _gave_up_count(jobs: list[dict[str, Any]]) -> int:
    """How many model files the running search left out because they gave up (read off
    the jobs' parameters: skip_files for stocks and crypto; for futures, the files no
    job is given)."""
    skipped: set[str] = set()
    futures = [j["params"] or {} for j in jobs if (j["params"] or {}).get("markets") == ["futures"]]
    if futures:
        from fleet2.models.futures import REGISTRY as FUTURES

        if all("files" in p for p in futures):
            skipped |= set(FUTURES) - {f for p in futures for f in p["files"]}
    for j in jobs:
        skipped |= {f"stock:{f}" for f in (j["params"] or {}).get("skip_files") or []}
    return len(skipped)


def _shares(conn: psycopg.Connection, jobs: list[dict[str, Any]]) -> list[str]:
    """"box2: Order block, Pullback" per futures search job split by strategy."""
    from coordinator import futures_models

    out = []
    for j in sorted(jobs, key=lambda j: str(j["id"])):
        p = j["params"] or {}
        if p.get("markets") != ["futures"] or "files" not in p:
            continue
        wid = j["lease_worker_id"] or j["target_worker_id"]
        w = conn.execute("SELECT name FROM workers WHERE id = %s", (wid,)).fetchone() if wid else None
        out.append(f"{w['name'] if w else 'next free worker'}: {futures_models.share_text(p)}")
    return out


def store_found(conn: psycopg.Connection, worker_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """A model a search job found. Kept unless the file already has KEEP_PER_FILE better
    finds; then it replaces the weakest of them or is dropped.

    Better: a model that passed the robustness checks (metrics.robustness.passed; a model
    without the block, from before them, counts as passed) beats every model that failed
    them, whatever their training scores; among models alike in that, the higher
    TRAINING score is better. So a model that failed the checks only ever replaces a
    weaker one that failed them too, and never pushes out one that passed."""
    module_name = str(body["module"])
    module = get_module(module_name)
    score = float(body["train_score"])
    failed = _failed_robustness(body.get("metrics"))
    others = conn.execute(
        """
        SELECT id, (metrics->>'train_score')::float AS score,
               coalesce(metrics->'robustness'->'passed' = 'false'::jsonb, false) AS failed
          FROM models
         WHERE origin = 'search' AND module = %s AND status = 'backtested'
         ORDER BY failed DESC, score ASC NULLS FIRST
        """,
        (module_name,),
    ).fetchall()
    if len(others) >= KEEP_PER_FILE:
        weakest = others[0]
        if failed and not weakest["failed"]:
            return {"kept": False, "reason": "failed the robustness checks, and the models already kept passed them"}
        if failed == weakest["failed"] and weakest["score"] is not None and weakest["score"] >= score:
            return {"kept": False, "reason": "not better than the models already kept"}
        conn.execute("UPDATE models SET status = 'retired' WHERE id = %s", (weakest["id"],))
    n = conn.execute("SELECT count(*) AS n FROM models WHERE module = %s AND origin = 'search'", (module_name,)).fetchone()["n"] + 1
    model_id = f"{module_name}-s{n}"
    while conn.execute("SELECT 1 FROM models WHERE id = %s", (model_id,)).fetchone():
        n += 1
        model_id = f"{module_name}-s{n}"
    metrics = dict(body["metrics"])
    metrics.update(train_score=score, seed=body.get("seed"), found_by=worker_id, job_id=body.get("job_id"))
    conn.execute(
        """
        INSERT INTO models (id, name, module, market, description, how_it_works, params, status, origin, metrics, backtested_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'backtested', 'search', %s, now())
        """,
        (model_id, f"{module.NAME} #{n}", module_name, module.MARKET, module.DESCRIPTION, module.HOW_IT_WORKS,
         Jsonb(body["params"]), Jsonb(metrics)),
    )
    return {"kept": True, "id": model_id}


def _failed_robustness(metrics: Any) -> bool:
    """True when the metrics hold a robustness report that did not pass (as the Models
    screen reads it: no report means no verdict, not a failure)."""
    block = metrics.get("robustness") if isinstance(metrics, dict) else None
    return isinstance(block, dict) and block.get("passed") is False


# ------------------------------------------------------------------ the search log


# Keys that hold curves or per-day arrays; never stored in the log.
_ARRAY_KEYS = {"curve", "daily", "pnl", "daily_pnl", "equity", "twin_curve", "series"}
_MAX_LIST = 32


def clean_metrics(value: Any) -> Any:
    """A trial's metrics with anything array-like removed (keys named curve, daily, pnl,
    equity ... and any list longer than _MAX_LIST): the log keeps compact numbers only."""
    if isinstance(value, dict):
        return {k: clean_metrics(v) for k, v in value.items()
                if str(k).lower() not in _ARRAY_KEYS and "curve" not in str(k).lower()
                and not (isinstance(v, list) and len(v) > _MAX_LIST)}
    if isinstance(value, list):
        return [clean_metrics(v) for v in value]
    return value


def _finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _job_uuid(value: Any) -> str | None:
    try:
        return str(uuid.UUID(str(value))) if value else None
    except ValueError:
        return None


def log_trials(conn: psycopg.Connection, worker_id: str | None, job_id: Any, market: str, family: str,
               trials: list[dict[str, Any]]) -> int:
    """Store a round's tried settings in one executemany; returns how many were stored."""
    if market not in LOG_MARKETS:
        raise BadRequest(f"market must be one of {', '.join(LOG_MARKETS)}")
    if not family:
        raise BadRequest("family is required")
    if len(trials) > MAX_TRIALS_PER_CALL:
        raise BadRequest(f"at most {MAX_TRIALS_PER_CALL} trials per call")
    job = _job_uuid(job_id)
    rows = []
    for t in trials:
        metrics = t.get("metrics")
        rows.append((market, family, job, worker_id, Jsonb(t.get("params") or {}), _finite(t.get("score")),
                     (str(t["rejected"])[:300] if t.get("rejected") else None),
                     Jsonb(clean_metrics(metrics)) if isinstance(metrics, dict) else None))
    if rows:
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO search_log (market, family, job_id, worker_id, params, score, rejected, metrics) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", rows)
    return len(rows)


def family_log(conn: psycopg.Connection, market: str, family: str, rules: Any = None) -> dict[str, Any]:
    """What the log knows about one family: counts, the best score, whether it gave up,
    and its best TOP_TRIALS trials plus its RECENT_TRIALS latest, each once."""
    if market not in LOG_MARKETS:
        raise BadRequest(f"market must be one of {', '.join(LOG_MARKETS)}")
    totals = conn.execute("SELECT count(*) AS tried, count(score) AS scored, max(score) AS best FROM search_log "
                          "WHERE market = %s AND family = %s", (market, family)).fetchone()
    top = conn.execute("SELECT id, params, score, rejected FROM search_log WHERE market = %s AND family = %s "
                       "AND score IS NOT NULL ORDER BY score DESC NULLS LAST LIMIT %s", (market, family, TOP_TRIALS)).fetchall()
    recent = conn.execute("SELECT id, params, score, rejected FROM search_log WHERE market = %s AND family = %s "
                          "ORDER BY created_at DESC, id DESC LIMIT %s", (market, family, RECENT_TRIALS)).fetchall()
    seen: set[int] = set()
    trials = []
    for r in top + recent:
        if r["id"] not in seen:
            seen.add(r["id"])
            trials.append({"params": r["params"], "score": r["score"], "rejected": r["rejected"]})
    return {"market": market, "family": family, "tried": int(totals["tried"]), "scored": int(totals["scored"]),
            "best_score": totals["best"], "gave_up": family in gave_up_families(conn, market, rules, family),
            "trials": trials}


def prune(conn: psycopg.Connection, keep: int = MAX_ROWS) -> int:
    """Keep at most `keep` rows per (market, family), deleting the oldest failed rows
    first (rejected, never scored, or no positive score), then the oldest of the rest.
    Returns how many rows were deleted."""
    deleted = 0
    for r in conn.execute("SELECT market, family, count(*) AS n FROM search_log GROUP BY market, family "
                          "HAVING count(*) > %s", (keep,)).fetchall():
        cur = conn.execute(
            "DELETE FROM search_log WHERE id IN (SELECT id FROM search_log WHERE market = %s AND family = %s "
            "ORDER BY coalesce(rejected IS NULL AND score > 0, false), created_at, id LIMIT %s)",
            (r["market"], r["family"], r["n"] - keep))
        deleted += cur.rowcount
    if deleted:
        log.info("search log: pruned %d old row(s)", deleted)
    return deleted


_last_prune: float | None = None


def prune_if_due(pool: Any) -> int:
    """The coordinator loop's call: prune at most every PRUNE_EVERY seconds."""
    global _last_prune
    now = time.monotonic()
    if _last_prune is not None and now - _last_prune < PRUNE_EVERY:
        return 0
    _last_prune = now
    with pool.connection() as conn:
        return prune(conn)


def reset(conn: psycopg.Connection, market: str | None = None, family: str | None = None) -> int:
    """Forget the log of one family, one market or everything (and let families that gave
    up by their held-out results try again). Returns how many rows were deleted."""
    if market is not None and market not in LOG_MARKETS:
        raise BadRequest(f"market must be one of {', '.join(LOG_MARKETS)}")
    cur = conn.execute("DELETE FROM search_log WHERE (%(m)s::text IS NULL OR market = %(m)s) "
                       "AND (%(f)s::text IS NULL OR family = %(f)s)", {"m": market, "f": family})
    now = conn.execute("SELECT now() AS t").fetchone()["t"]
    row = conn.execute("SELECT value FROM settings WHERE key = %s", (RESET_KEY,)).fetchone()
    marks = dict(row["value"]) if row else {}
    marks[f"{market or '*'}|{family or '*'}"] = now.isoformat()
    conn.execute("INSERT INTO settings (key, value, updated_at) VALUES (%s, %s, now()) "
                 "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()", (RESET_KEY, Jsonb(marks)))
    return cur.rowcount


def _reset_since(conn: psycopg.Connection, market: str, family: str) -> datetime | None:
    row = conn.execute("SELECT value FROM settings WHERE key = %s", (RESET_KEY,)).fetchone()
    marks = row["value"] if row else {}
    times = [datetime.fromisoformat(marks[k]) for k in ("*|*", f"{market}|*", f"*|{family}", f"{market}|{family}")
             if k in marks]
    return max(times) if times else None


def gave_up_families(conn: psycopg.Connection, market: str, rules: Any = None,
                     only: str | None = None) -> dict[str, dict[str, Any]]:
    """{family: {"tried": n, "rule": "tries" | "held_out"}} for the families of `market`
    that gave up (see the module docstring). `rules` (Topstep's) is needed for rule (b)."""
    out: dict[str, dict[str, Any]] = {}
    for r in conn.execute("SELECT family, count(*) AS tried, max(score) AS best FROM search_log "
                          "WHERE market = %s AND (%s::text IS NULL OR family = %s) GROUP BY family",
                          (market, only, only)).fetchall():
        if r["tried"] >= GIVE_UP_TRIES and not (r["best"] is not None and r["best"] > 0):
            out[r["family"]] = {"tried": int(r["tried"]), "rule": "tries"}
    if market == "futures" and rules is not None and rules.account_fees_set() and rules.max_payout is not None:
        for r in conn.execute("SELECT module, count(*) AS n FROM models WHERE market = 'futures' AND origin = 'search' "
                              "AND metrics->'held_out' IS NOT NULL AND (%s::text IS NULL OR module = %s) "
                              "GROUP BY module HAVING count(*) >= %s", (only, only, GIVE_UP_TESTED)).fetchall():
            if r["module"] not in out and _tested_without_an_edge(conn, r["module"], rules):
                tried = conn.execute("SELECT count(*) AS n FROM search_log WHERE market = 'futures' AND family = %s",
                                     (r["module"],)).fetchone()["n"]
                out[r["module"]] = {"tried": int(tried), "rule": "held_out"}
    return out


def _tested_without_an_edge(conn: psycopg.Connection, family: str, rules: Any) -> bool:
    """Rule (b): at least GIVE_UP_TESTED of the family's models were tested on the
    held-out prices under today's rules, and none has an expected net per attempt clearly
    above its fee floor or beats its coin-flip twin. A model whose net cannot be worked
    out counts as not tested, so nothing is decided without the fees."""
    from coordinator import futures_view
    from fleet2.sim import topstep

    since = _reset_since(conn, "futures", family)
    rows = conn.execute("SELECT metrics->'held_out'->>'sim_key' AS sim_key, metrics->'held_out'->'sizes' AS sizes "
                        "FROM models WHERE market = 'futures' AND origin = 'search' AND module = %s "
                        "AND metrics->'held_out' IS NOT NULL AND (%s::timestamptz IS NULL "
                        "OR coalesce(backtested_at, created_at) > %s)", (family, since, since)).fetchall()
    tested = 0
    for row in rows:
        if row["sim_key"] != rules.sim_key():
            continue
        pick = topstep.best_size({"sizes": row["sizes"] or []}, rules)
        if pick is None or pick.get("net") is None:
            continue
        tested += 1
        floor = -rules.combine_monthly * ((pick.get("sim") or {}).get("mean_months") or 0.0)
        if pick["net"] > floor + FLOOR_MARGIN * abs(floor) or futures_view.beats_twin(pick, rules):
            return False
    return tested >= GIVE_UP_TESTED


def skipped_files(conn: psycopg.Connection, markets: list[str]) -> list[str]:
    """The stock and crypto model files left out of a search because they gave up. Raises
    Conflict when that is every file of the markets."""
    from fleet2.models import REGISTRY

    gave_up = {m: gave_up_families(conn, m) for m in markets}
    files = {n: mod.MARKET for n, mod in REGISTRY.items() if mod.MARKET in markets}
    skip = sorted(n for n, m in files.items() if n in gave_up[m])
    if files and len(skip) == len(files):
        raise nothing_left(sum(gave_up[files[n]][n]["tried"] for n in skip))
    return skip


def nothing_left(tries: int) -> Conflict:
    return Conflict(f"No model file has found an edge after {tries:,} tries; new model ideas are needed or the log "
                    "can be reset")
