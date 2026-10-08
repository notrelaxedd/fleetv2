"""Models on the coordinator: the starter models registered at start, backtest results
stored as workers report them, and the job details a backtest needs.

A model row points at its model file (`module`) plus its own parameters, so a model
found by search later is a row, not new code.
"""
from __future__ import annotations

from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from coordinator.errors import BadRequest, Conflict, NotFound
from coordinator.settings import get_setting, put_setting
from coordinator.limits import Limits
from fleet2.models import REGISTRY
from fleet2.models.base import check_module
from fleet2.universe import HELD_OUT_FRACTION, MARKETS

STATUS_TEXT = {"backtested": "Backtested", "paper_trading": "Paper trading", "retired": "Retired", None: "Not tested yet"}


def sync_starters(conn: psycopg.Connection) -> int:
    """Insert (or refresh the wording of) the starter models; metrics and status are kept."""
    for name, module in REGISTRY.items():
        check_module(module)
        conn.execute(
            """
            INSERT INTO models (id, name, module, market, description, how_it_works, params, origin)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'starter')
            ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, market = EXCLUDED.market,
                   description = EXCLUDED.description, how_it_works = EXCLUDED.how_it_works,
                   params = CASE WHEN models.metrics IS NULL THEN EXCLUDED.params ELSE models.params END
            """,
            (name, module.NAME, name, module.MARKET, module.DESCRIPTION, module.HOW_IT_WORKS, Jsonb(module.DEFAULT_PARAMS)),
        )
    return len(REGISTRY)


def list_models(conn: psycopg.Connection, include_retired: bool = False) -> list[dict[str, Any]]:
    """Every model, starters first then by name."""
    rows = conn.execute(
        "SELECT * FROM models WHERE %s OR status IS DISTINCT FROM 'retired'"
        " ORDER BY (origin = 'starter') DESC, name",
        (include_retired,),
    ).fetchall()
    return rows


def get_model(conn: psycopg.Connection, model_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM models WHERE id = %s", (model_id,)).fetchone()
    if row is None:
        raise NotFound(f"no model {model_id!r}")
    return row


def held_out_start(conn: psycopg.Connection, market: str) -> int:
    """The first moment of the market's held-out period (epoch seconds), fixed once.

    Set the first time it is needed, at the last HELD_OUT_FRACTION of the history then
    cached, and never moved afterwards: every backtest and every search uses the same
    date, so all models are ranked on the same held-out period and model search never
    trains on it, however long the fleet runs. Raises when there are no prices yet."""
    starts = get_setting(conn, "held_out_start", None) or {}
    if market in starts:
        return int(starts[market])
    bench = MARKETS[market]["benchmark"]
    row = conn.execute("SELECT min(ts) AS a, max(ts) AS b FROM bars WHERE symbol = %s", (bench,)).fetchone()
    if row is None or row["a"] is None:
        raise Conflict("No price data yet: run a Data refresh first")
    first, last = row["a"].timestamp(), row["b"].timestamp()
    start = int(first + (1.0 - HELD_OUT_FRACTION) * (last - first))
    starts = {**starts, market: start}
    put_setting(conn, "held_out_start", starts, "system", "held_out_start_set")
    return start


def job_params(conn: psycopg.Connection, kind: str, model_id: str | None, params: dict[str, Any],
               limits: Limits) -> dict[str, Any]:
    """Fill in what a worker needs, so it never has to ask the coordinator for the model."""
    if kind == "data_refresh":
        markets = params.get("markets") or ["stocks", "crypto"]
        if not isinstance(markets, list) or not set(markets) <= {"stocks", "crypto", "futures"}:
            raise BadRequest("markets must be a list of stocks, crypto and/or futures")
        return {"markets": markets}
    if kind == "backtest":
        if not model_id:
            raise BadRequest("Pick a model for this job")
        model = get_model(conn, model_id)
        return {
            "model_id": model["id"],
            "module": model["module"],
            "params": model["params"],
            "market": model["market"],
            "limits": {"money": limits.starting_balance_per_model, "max_per_position": limits.max_per_position,
                       "max_per_model": limits.max_per_model},
            "held_out_fraction": HELD_OUT_FRACTION,
            "held_out_start_t": held_out_start(conn, model["market"]),
        }
    return params


def store_backtest(conn: psycopg.Connection, model_id: str, metrics: dict[str, Any], job_id: str | None) -> dict[str, Any]:
    """Keep the latest backtest of a model; a model never tested becomes Backtested."""
    row = conn.execute(
        """
        UPDATE models SET metrics = %s, backtested_at = now(),
               status = COALESCE(status, 'backtested')
         WHERE id = %s RETURNING id, status
        """,
        (Jsonb({**metrics, "job_id": job_id}), model_id),
    ).fetchone()
    if row is None:
        raise NotFound(f"no model {model_id!r}")
    return row
