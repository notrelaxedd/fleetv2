"""Futures models on the coordinator: the five starter files, what model search finds,
the count of settings tried, the jobs' parameters and the once-only Final check.

Keeping a varied set (at most KEEP_PER_FILE found models per model file):
- A find whose training days' P&L moves more than 90% like a kept model's is the same
  idea twice: it replaces that model when its training score is higher, and is dropped
  otherwise.
- Otherwise, with fewer than KEEP_PER_FILE kept for its file it is kept; with that many
  it replaces the kept model with the closest settings, when it scores higher.
Training scores decide; held-out results never do (they are for ranking).

No order is ever placed for a futures model: they cannot paper trade (trading.start
refuses), and nothing here talks to a broker.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import psycopg
from psycopg.types.json import Jsonb

from coordinator import futures_data, queue
from coordinator.errors import BadRequest, Conflict, NotFound
from fleet2.models.futures import REGISTRY
from fleet2.models.futures.base import check_module, params_with_defaults
from fleet2.sim import topstep

KEEP_PER_FILE = 5
ALIKE = 0.9  # daily P&L correlation above which two models count as the same idea


def sync_starters(conn: psycopg.Connection) -> int:
    """The five futures model files as starter models with their default settings."""
    for name, module in REGISTRY.items():
        check_module(module)
        conn.execute(
            """
            INSERT INTO models (id, name, module, market, description, how_it_works, params, origin)
            VALUES (%s, %s, %s, 'futures', %s, %s, %s, 'starter')
            ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, description = EXCLUDED.description,
                   how_it_works = EXCLUDED.how_it_works,
                   params = CASE WHEN models.metrics IS NULL THEN EXCLUDED.params ELSE models.params END
            """,
            (name, module.NAME, name, module.DESCRIPTION, module.HOW_IT_WORKS, Jsonb(module.DEFAULT_PARAMS)),
        )
    return len(REGISTRY)


# ------------------------------------------------------------------ jobs


def require_trading_fees(rules: topstep.Rules) -> None:
    missing = rules.missing_trading_fees()
    if missing:
        raise Conflict(f"{topstep.FEE_MESSAGE} ({', '.join(missing)}) before futures backtests and searches can run")


def backtest_params(conn: psycopg.Connection, model: dict[str, Any], rules: topstep.Rules) -> dict[str, Any]:
    require_trading_fees(rules)
    return {"model_id": model["id"], "module": model["module"], "params": model["params"], "market": "futures",
            "rules": rules.to_dict(), "periods": futures_data.futures_periods(conn)}


def search_params(conn: psycopg.Connection, rules: topstep.Rules) -> dict[str, Any]:
    require_trading_fees(rules)
    if futures_data.futures_feed(conn) is None:
        raise Conflict("No futures prices yet: press Load futures prices first")
    return {"markets": ["futures"], "rules": rules.to_dict(), "periods": futures_data.futures_periods(conn),
            "candidates": 16}


def kept_models(conn: psycopg.Connection) -> list[dict[str, Any]]:
    """The futures models search has kept (the parents of next round's small changes)."""
    return conn.execute("SELECT id, module, params FROM models WHERE market = 'futures' AND origin = 'search' "
                        "AND status = 'backtested' ORDER BY id").fetchall()


def add_tries(conn: psycopg.Connection, tries: dict[str, Any]) -> None:
    """Add a round's tried settings per model file: how many, and the sum and sum of
    squares of their daily Sharpe ratios."""
    for module, t in tries.items():
        if module not in REGISTRY:
            raise BadRequest(f"unknown futures model file {module!r}")
        n, total, squares = int(t.get("n") or 0), float(t.get("sum") or 0.0), float(t.get("sq") or 0.0)
        if n < 0 or not all(np.isfinite([total, squares])):
            raise BadRequest("tries must be finite counts")
        conn.execute(
            "INSERT INTO search_tries (module, tries, sharpe_sum, sharpe_sq) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (module) DO UPDATE SET tries = search_tries.tries + EXCLUDED.tries,"
            " sharpe_sum = search_tries.sharpe_sum + EXCLUDED.sharpe_sum,"
            " sharpe_sq = search_tries.sharpe_sq + EXCLUDED.sharpe_sq, updated_at = now()",
            (module, n, total, squares))


def tries(conn: psycopg.Connection) -> dict[str, dict[str, float]]:
    return {r["module"]: {"n": int(r["tries"]), "sum": float(r["sharpe_sum"]), "sq": float(r["sharpe_sq"])}
            for r in conn.execute("SELECT * FROM search_tries").fetchall()}


# ------------------------------------------------------------------ keeping finds


def _daily(metrics: dict[str, Any]) -> dict[int, float]:
    daily = ((metrics or {}).get("train") or {}).get("daily") or {}
    return dict(zip(daily.get("days") or [], daily.get("pnl") or []))


def correlation(a: dict[int, float], b: dict[int, float]) -> float | None:
    """Correlation of two models' daily P&L over the days both have (None if either is flat)."""
    days = sorted(set(a) & set(b))
    if len(days) < 20:
        return None
    x, y = np.array([a[d] for d in days]), np.array([b[d] for d in days])
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def distance(module: Any, a: dict[str, Any], b: dict[str, Any]) -> float:
    """How far apart two settings are: each number as a share of its search range, each
    choice 0 when equal and 1 when not, added up."""
    total = 0.0
    for name, spec in module.SEARCH_SPACE.items():
        if spec[-1] == "choice":
            total += 0.0 if a.get(name) == b.get(name) else 1.0
        else:
            low, high = float(spec[0]), float(spec[1])
            total += abs(float(a.get(name, low)) - float(b.get(name, low))) / ((high - low) or 1.0)
    return total


def store_found(conn: psycopg.Connection, worker_id: str | None, body: dict[str, Any]) -> dict[str, Any]:
    """A futures model a search job found (see the module docstring for what is kept)."""
    name = str(body["module"])
    if name not in REGISTRY:
        raise BadRequest(f"unknown futures model file {name!r}")
    module = REGISTRY[name]
    params = params_with_defaults(module, body["params"])
    score = float(body["train_score"])
    metrics = dict(body["metrics"])
    mine = _daily(metrics)
    kept = conn.execute("SELECT id, name, module, params, metrics, (metrics->>'train_score')::float AS score "
                        "FROM models WHERE market = 'futures' AND origin = 'search' AND status = 'backtested'").fetchall()
    replace = None
    for other in kept:
        alike = correlation(mine, _daily(other["metrics"]))
        if alike is not None and alike > ALIKE:
            if other["score"] is not None and other["score"] >= score:
                return {"kept": False, "reason": f"trades like {other['name']} ({alike:.0%} alike), which scores higher"}
            replace = other
            break
    if replace is None:
        same = [k for k in kept if k["module"] == name]
        if len(same) >= KEEP_PER_FILE:
            closest = min(same, key=lambda k: distance(module, params, k["params"]))
            if closest["score"] is not None and closest["score"] >= score:
                return {"kept": False, "reason": f"not better than the closest kept model, {closest['name']}"}
            replace = closest
    if replace is not None:
        conn.execute("UPDATE models SET status = 'retired' WHERE id = %s", (replace["id"],))
    n = conn.execute("SELECT count(*) AS n FROM models WHERE module = %s AND origin = 'search'", (name,)).fetchone()["n"] + 1
    model_id = f"{name}-s{n}"
    while conn.execute("SELECT 1 FROM models WHERE id = %s", (model_id,)).fetchone():
        n += 1
        model_id = f"{name}-s{n}"
    metrics.update(train_score=score, seed=body.get("seed"), parent=body.get("parent"), found_by=worker_id,
                   job_id=body.get("job_id"))
    conn.execute(
        """
        INSERT INTO models (id, name, module, market, description, how_it_works, params, status, origin, metrics, backtested_at)
        VALUES (%s, %s, %s, 'futures', %s, %s, %s, 'backtested', 'search', %s, now())
        """,
        (model_id, f"{module.NAME} #{n}", name, module.DESCRIPTION, module.HOW_IT_WORKS, Jsonb(params), Jsonb(metrics)),
    )
    return {"kept": True, "id": model_id, "replaced": replace["id"] if replace else None}


# ------------------------------------------------------------------ the Final check (lockbox)


def final_check(conn: psycopg.Connection, model_id: str) -> dict[str, Any] | None:
    return conn.execute("SELECT * FROM final_checks WHERE model_id = %s", (model_id,)).fetchone()


def final_check_running(conn: psycopg.Connection, model_id: str) -> bool:
    return conn.execute("SELECT 1 FROM jobs WHERE kind = 'final_check' AND model_id = %s AND status IN "
                        "('queued', 'leased', 'cancel_requested')", (model_id,)).fetchone() is not None


def start_final_check(conn: psycopg.Connection, model_id: str, rules: topstep.Rules, target: str = "auto") -> dict[str, Any]:
    """Queue the one Final check of a model: the lockbox prices, at the contract size its
    held-out results chose. Refused when it already ran, is running, the model is not a
    tested futures model, or the prices are not real futures prices."""
    model = conn.execute("SELECT * FROM models WHERE id = %s", (model_id,)).fetchone()
    if model is None:
        raise NotFound(f"no model {model_id!r}")
    if model["market"] != "futures":
        raise BadRequest("The Final check is for futures models")
    done = final_check(conn, model_id)
    if done is not None:
        when = done["created_at"]
        raise Conflict(f"The Final check of {model['name']} already ran on {when:%b} {when.day}, {when.year}; "
                       "its result is kept forever and never run again")
    if final_check_running(conn, model_id):
        raise Conflict("The Final check of this model is already running")
    held = (model["metrics"] or {}).get("held_out")
    if not held:
        raise Conflict("Run a backtest first: the Final check uses the contract size the held-out test chose")
    if (model["metrics"] or {}).get("held_out", {}).get("sim_key") != rules.sim_key():
        raise Conflict("Topstep's rules in config/topstep.toml changed since this model was tested: run a backtest first")
    feed = futures_data.futures_feed(conn)
    if feed in (None, "proxy", "mixed"):
        raise Conflict("The Final check needs real futures prices (a Databento key): it runs only once per model, "
                       "and proxy prices can never count")
    require_trading_fees(rules)
    best = topstep.best_size(held, rules)
    contracts = int(best["contracts"]) if best else 1
    params = {"model_id": model_id, "module": model["module"], "params": model["params"], "market": "futures",
              "rules": rules.to_dict(), "periods": futures_data.futures_periods(conn), "contracts": contracts}
    result = queue.create_job(conn, "final_check", params, target, model_id)
    where = "waits for a free worker" if result.waiting else "started"
    return {"jobs": result.jobs, "message": f"Final check of {model['name']} {where}: it runs once and is kept forever"}


def store_final_check(conn: psycopg.Connection, model_id: str, worker_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """The lockbox result of a running Final check job, stored forever (the table
    refuses changes and deletes; a second result is refused here)."""
    job_id = body.get("job_id")
    job = conn.execute("SELECT * FROM jobs WHERE id = %s::uuid AND kind = 'final_check' AND model_id = %s",
                       (job_id, model_id)).fetchone() if job_id else None
    if job is None or job["lease_worker_id"] != worker_id or job["status"] != "leased":
        raise Conflict("This is not a running Final check of that model")
    if final_check(conn, model_id) is not None:
        raise Conflict("A Final check result is already stored for this model")
    conn.execute("INSERT INTO final_checks (model_id, job_id, worker_id, feed, result) VALUES (%s, %s, %s, %s, %s)",
                 (model_id, job_id, worker_id, str(body.get("feed") or ""), Jsonb(body["result"])))
    return {"stored": True}
