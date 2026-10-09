"""Model search on the coordinator: start and stop search jobs, store the models they
find, and the status line for the Models screen.

Start model search sends one search job to every idle worker (at least one job, which
waits for a free worker otherwise), each with its own seed so they never try the same
settings. Stop model search cancels them all.

Found models are kept sparingly so the list stays readable: at most KEEP_PER_FILE
search models per model file stay active. A new find replaces the weakest one only when
its TRAINING score is higher; held-out results never decide what is kept (they are for
ranking on the Models screen).
"""
from __future__ import annotations

import secrets
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from coordinator import queue
from coordinator.errors import Conflict
from coordinator.limits import Limits
from coordinator.models import held_out_start
from fleet2.models import get_module
from fleet2.universe import HELD_OUT_FRACTION

KEEP_PER_FILE = 5
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
    params = {
        "markets": markets,
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
        jobs = []
        for w in workers:
            jobs += queue.create_job(conn, "model_search", {**params, "seed": secrets.randbelow(10**9)}, w["id"]).jobs
        return {"jobs": jobs, "message": f"{label} started on {len(jobs)} worker{'s' if len(jobs) != 1 else ''}"}
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
    return {"running": True, "button": "Stop model search", "detail": detail}


def store_found(conn: psycopg.Connection, worker_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """A model a search job found. Kept unless the file already has KEEP_PER_FILE better
    finds (by training score); then it replaces the weakest of them or is dropped."""
    module_name = str(body["module"])
    module = get_module(module_name)
    score = float(body["train_score"])
    others = conn.execute(
        """
        SELECT id, (metrics->>'train_score')::float AS score FROM models
         WHERE origin = 'search' AND module = %s AND status = 'backtested'
         ORDER BY score ASC NULLS FIRST
        """,
        (module_name,),
    ).fetchall()
    if len(others) >= KEEP_PER_FILE:
        weakest = others[0]
        if weakest["score"] is not None and weakest["score"] >= score:
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
