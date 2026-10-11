"""Scores Claude's news predictions against what the price really did (docs/NEWS_PLAN.md).

Claude already knows how most old news turned out, so the only evidence that counts is
predictions made before the outcome existed: live ones (forward shadow mode) and
post-cutoff replays. This module turns each such prediction into a number once its
horizon has passed, the same way for both, and decides whether a market's news strategy
has earned paper money.

Why it is built this way:

- No lookahead. The entry is the open of the first bar that *starts* strictly after the
  decision; a bar that was already running when the decision was made is never used,
  even though its open was in the past. Alpaca's daily bars start at midnight New York,
  so a decision during Tuesday's session enters at Wednesday's open (pessimistic on
  purpose), and so does one made at 23:00 Tuesday New York time. The exit is the close
  of the `horizon_days`-th bar counting the entry bar as bar 1 (crypto: `horizon_days *
  24` hourly bars), and it is only read once that bar has finished.
- Real costs. Stocks pay 5 bp slippage each way, and shorts pay the borrow fee
  (`short_borrow_fee_pct` a year) for the calendar days held. Crypto pays 10 bp slippage
  plus Alpaca's 25 bp taker fee each way. These are the backtester's COSTS.
- A coin-flip twin. Every prediction is scored next to a twin with the same entry and
  exit bars but a random direction, seeded by the prediction id so it never changes. A
  forecaster with no skill beats its twin only by luck; the gate asks for the paired
  edge to be more than two standard errors above zero before any money is risked.
- Scores are frozen. Entry and exit prices are written to the prediction, read from one
  snapshot of the bar cache (stock bars are re-downloaded split- and dividend-adjusted,
  so stored prices can shift later; the ratio we keep does not).
"""
from __future__ import annotations

import logging
import math
import random
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import psycopg

from coordinator.data import TIMEFRAME_DELTA
from coordinator.limits import Limits
from fleet2.sim.backtest import COSTS
from fleet2.universe import MARKETS

log = logging.getLogger(__name__)

BATCH = 500  # predictions scored per call at most
SOURCES = ("live", "replay")
# Benchmark bars are taken at the same ts as the symbol's; when one is missing, the
# nearest earlier benchmark bar is used, but only if it is at most this much older.
BENCH_SLACK = {"stocks": timedelta(days=4), "crypto": timedelta(hours=3)}
# Past twice the horizon plus this many days and still unscorable: the bars are probably
# missing, so say so in score_note (it is still retried every pass).
OVERDUE_EXTRA_DAYS = 4
OVERDUE_NOTE = ("Its horizon has passed but the price bars needed to score it (the symbol's or the "
                "benchmark's) are not in the cache; it stays unscored until they arrive.")
BUCKETS = ((0.0, 0.5, "0–0.5"), (0.5, 0.6, "0.5–0.6"), (0.6, 0.7, "0.6–0.7"), (0.7, 0.8, "0.7–0.8"),
           (0.8, 1.0, "0.8–1.0"))

_CANDIDATES = """
SELECT p.id, p.symbol, p.direction,
       e.ts AS entry_ts, e.open AS entry_price, x.ts AS exit_ts, x.close AS exit_price,
       be.open AS bench_entry, bx.close AS bench_exit
FROM news_predictions p
CROSS JOIN LATERAL (
  SELECT b.ts, b.open FROM bars b
  WHERE b.symbol = p.symbol AND b.timeframe = %(tf)s AND b.ts > p.decided_at
  ORDER BY b.ts LIMIT 1) e
CROSS JOIN LATERAL (
  SELECT b.ts, b.close FROM bars b
  WHERE b.symbol = p.symbol AND b.timeframe = %(tf)s AND b.ts >= e.ts
  ORDER BY b.ts OFFSET p.horizon_days * %(per_day)s - 1 LIMIT 1) x
CROSS JOIN LATERAL (
  SELECT b.ts, b.open FROM bars b
  WHERE b.symbol = %(bench)s AND b.timeframe = %(tf)s AND b.ts <= e.ts
  ORDER BY b.ts DESC LIMIT 1) be
CROSS JOIN LATERAL (
  SELECT b.ts, b.close FROM bars b
  WHERE b.symbol = %(bench)s AND b.timeframe = %(tf)s AND b.ts <= x.ts
  ORDER BY b.ts DESC LIMIT 1) bx
WHERE p.market = %(market)s AND p.scored_at IS NULL AND p.direction IN ('up', 'down')
  AND p.decided_at + make_interval(days => p.horizon_days) <= %(now)s
  AND x.ts + %(bar)s <= %(now)s
  AND be.ts >= e.ts - %(slack)s AND bx.ts >= x.ts - %(slack)s
  AND EXISTS (SELECT 1 FROM bars b WHERE b.symbol = %(bench)s AND b.timeframe = %(tf)s AND b.ts >= x.ts)
ORDER BY p.decided_at, p.id
LIMIT %(limit)s
"""

_SAVE = """
UPDATE news_predictions SET entry_ts = %(entry_ts)s, exit_ts = %(exit_ts)s, entry_price = %(entry_price)s,
  exit_price = %(exit_price)s, move_pct = %(move_pct)s, bench_pct = %(bench_pct)s, signed_pct = %(signed_pct)s,
  twin_direction = %(twin_direction)s, twin_signed_pct = %(twin_signed_pct)s, hit = %(hit)s,
  scored_at = %(scored_at)s, score_note = NULL
WHERE id = %(id)s AND scored_at IS NULL
"""


def twin_direction(prediction_id: int) -> str:
    """The coin-flip twin's direction: fixed forever by the prediction id."""
    return "up" if random.Random(f"twin:{prediction_id}").random() < 0.5 else "down"


def days_held(entry_ts: datetime, exit_ts: datetime) -> int:
    """Calendar days from the start of the entry bar to the end of the exit daily bar.

    Counts the exit day as a whole day, so a one-day hold pays one day of borrow fee and
    a Friday-to-Monday hold pays four (conservative: the fee is charged per night held).
    """
    return round((exit_ts - entry_ts).total_seconds() / 86400) + 1


def cost_pct(market: str, direction: str, days: int, limits: Limits) -> float:
    """Round-trip cost in percent of the position for one direction.

    Stocks: 2 x 5 bp slippage, plus `short_borrow_fee_pct` x days / 365 on a short.
    Crypto: 2 x (10 bp slippage + 25 bp fee). Alpaca cannot short crypto, but a crypto
    "down" is scored as if it could be shorted at the same cost (no borrow fee), because
    the score measures the forecaster's skill; the trader only acts on it by staying out.
    """
    c = COSTS[market]
    pct = 2 * (c.slippage_bps + c.fee_bps) / 100.0
    if market == "stocks" and direction == "down":
        pct += limits.short_borrow_fee_pct * days / 365.0
    return pct


def score_row(row: dict[str, Any], market: str, limits: Limits, now: datetime) -> dict[str, Any]:
    """The score columns for one candidate row (entry/exit/benchmark prices already chosen).

    hit compares the move with the benchmark's over the same bars, before costs; for the
    benchmark itself (SPY, BTC/USD) that would always be a miss, so its hit is the raw
    direction being right.
    """
    move = (row["exit_price"] / row["entry_price"] - 1.0) * 100.0
    bench = (row["bench_exit"] / row["bench_entry"] - 1.0) * 100.0
    days = days_held(row["entry_ts"], row["exit_ts"])
    sign = 1.0 if row["direction"] == "up" else -1.0
    twin = twin_direction(row["id"])
    twin_sign = 1.0 if twin == "up" else -1.0
    relative = move if row["symbol"] == MARKETS[market]["benchmark"] else move - bench
    return {
        "id": row["id"], "entry_ts": row["entry_ts"], "exit_ts": row["exit_ts"],
        "entry_price": row["entry_price"], "exit_price": row["exit_price"],
        "move_pct": move, "bench_pct": bench,
        "signed_pct": sign * move - cost_pct(market, row["direction"], days, limits),
        "twin_direction": twin,
        "twin_signed_pct": twin_sign * move - cost_pct(market, twin, days, limits),
        "hit": sign * relative > 0, "scored_at": now,
    }


def score_pending(conn: psycopg.Connection, limits: Limits, now: datetime, batch: int = BATCH) -> int:
    """Score up to `batch` unscored up/down predictions whose exit bar has finished.

    Rows that cannot be scored yet are simply not chosen (so they never crowd out ones
    that can); once well past their horizon they get OVERDUE_NOTE but stay unscored and
    keep being retried. Safe to call any number of times. Returns how many were scored.
    """
    scored: list[dict[str, Any]] = []
    for market, spec in MARKETS.items():
        left = batch - len(scored)
        if left <= 0:
            break
        tf = spec["timeframe"]
        rows = conn.execute(_CANDIDATES, {
            "tf": tf, "per_day": 1 if tf == "1Day" else 24, "bench": spec["benchmark"], "market": market,
            "now": now, "bar": TIMEFRAME_DELTA[tf], "slack": BENCH_SLACK[market], "limit": left,
        }).fetchall()
        for row in rows:
            if row["entry_price"] <= 0 or row["bench_entry"] <= 0:
                log.warning("news prediction %s: non-positive entry price, not scored", row["id"])
                continue
            scored.append(score_row(row, market, limits, now))
    done = 0
    with conn.transaction():
        if scored:
            with conn.cursor() as cur:
                for values in scored:
                    cur.execute(_SAVE, values)
                    done += cur.rowcount
        conn.execute(
            "UPDATE news_predictions SET score_note = %s WHERE scored_at IS NULL AND direction <> 'none'"
            " AND score_note IS NULL AND decided_at + make_interval(days => 2 * horizon_days + %s) < %s",
            (OVERDUE_NOTE, OVERDUE_EXTRA_DAYS, now),
        )
    return done


def _filters(market: str | None, source: str | None, model: str | None,
             min_confidence: float | None) -> tuple[str, list[Any]]:
    where, args = ["TRUE"], []
    for column, value in (("market", market), ("source", source), ("model", model)):
        if value is not None:
            where.append(f"{column} = %s")
            args.append(value)
    if min_confidence is not None:
        where.append("confidence >= %s")
        args.append(min_confidence)
    return " AND ".join(where), args


def _mean(values: np.ndarray) -> float | None:
    return float(values.mean()) if len(values) else None


def stats(conn: psycopg.Connection, market: str | None = None, source: str | None = None,
          model: str | None = None, min_confidence: float | None = None) -> dict[str, Any]:
    """How the scored predictions did (filters optional). Unscored rows never count.

    Percentages are plain numbers (1.5 means 1.5%). edge_pct is the mean of the paired
    difference signed_pct - twin_signed_pct and edge_se_pct its standard error; a value
    that cannot be computed (no rows, or one row for a standard error) is None.
    """
    where, args = _filters(market, source, model, min_confidence)
    counts = conn.execute(
        "SELECT count(*) FILTER (WHERE scored_at IS NULL AND direction <> 'none') AS pending,"
        " count(*) FILTER (WHERE direction = 'none') AS none_count"
        f" FROM news_predictions WHERE {where}", args,
    ).fetchone()
    rows = conn.execute(
        "SELECT direction, confidence, hit, signed_pct, twin_signed_pct FROM news_predictions"
        f" WHERE {where} AND scored_at IS NOT NULL AND direction <> 'none' ORDER BY id", args,
    ).fetchall()
    n = len(rows)
    hit = np.array([bool(r["hit"]) for r in rows], dtype=float)
    signed = np.array([r["signed_pct"] for r in rows], dtype=float)
    twin = np.array([r["twin_signed_pct"] for r in rows], dtype=float)
    conf = np.array([r["confidence"] for r in rows], dtype=float)
    direction = np.array([r["direction"] for r in rows], dtype=object)
    diff = signed - twin
    edge = _mean(diff)
    se = float(diff.std(ddof=1) / math.sqrt(n)) if n >= 2 else None
    calibration = []
    for low, high, label in BUCKETS:
        inside = (conf >= low) & ((conf <= high) if high >= 1.0 else (conf < high))
        calibration.append({"label": label, "n": int(inside.sum()), "hit_rate": _mean(hit[inside])})
    by_direction = {}
    for d in ("up", "down"):
        mask = direction == d
        by_direction[d] = {"n": int(mask.sum()), "hit_rate": _mean(hit[mask]),
                           "mean_signed_pct": _mean(signed[mask])}
    return {
        "n": n,
        "pending": int(counts["pending"]),
        "none_count": int(counts["none_count"]),
        "hit_rate": _mean(hit),
        "mean_signed_pct": _mean(signed),
        "mean_twin_pct": _mean(twin),
        "edge_pct": edge,
        "edge_se_pct": se,
        "beats_twin": bool(n >= 2 and edge is not None and se is not None and edge > 2 * se),
        "calibration": calibration,
        "by_direction": by_direction,
    }


def gate(conn: psycopg.Connection, market: str, limits: Limits) -> tuple[bool, str]:
    """May this market's news strategy start paper trading? (passes, one plain sentence)

    Counts the market's scored predictions from both sources at the confidence it would
    trade (>= limits.min_confidence). Passes only with at least min_scored_predictions of
    them, an edge over the coin-flip twin above two standard errors, and a positive mean
    return after costs.
    """
    if market not in MARKETS:
        raise ValueError(f"unknown market {market!r}")
    s = stats(conn, market=market, min_confidence=limits.min_confidence)
    need = max(int(math.ceil(limits.min_scored_predictions)), 2)
    n = s["n"]
    if n < need:
        return False, f"Needs {need} scored predictions; has {n}."
    edge, se = s["edge_pct"], s["edge_se_pct"]
    if not s["beats_twin"]:
        return False, f"Does not beat the coin-flip twin yet: edge {edge:+.2f}% ± {se:.2f}%."
    if s["mean_signed_pct"] <= 0:
        return False, (f"Beats the coin-flip twin but loses money after costs: "
                       f"{s['mean_signed_pct']:+.2f}% a prediction on average.")
    return True, f"Ready: {n} scored predictions, edge {edge:+.2f}% ± {se:.2f}% over the coin flip."
