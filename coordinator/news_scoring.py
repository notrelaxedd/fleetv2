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
  forecaster with no skill beats its twin only by luck. The edge is measured against the
  market (the move minus the benchmark's, signed by direction, both sides paying the
  call's own costs), so a forecaster that only rides the market's drift, or only saves
  the borrow fee by never shorting, has no edge at all.
- The gate counts what would be traded. Only the calls the news strategy would act on
  (`would_trade`, the very predicate coordinator.news_trading uses for its entries), only
  one observation per symbol, entry day and direction (nine articles making the same call
  on the same day are one call, not nine), and no replay answered by a different model
  than the one asked (a refusal fallback may have a later training cutoff, so it may know
  the outcome). It asks for the edge to be more than two standard errors above zero over
  enough such independent calls, and for those calls to make money after costs.
- Scores are frozen. Entry and exit prices are written to the prediction, read from one
  snapshot of the bar cache (stock bars are re-downloaded split- and dividend-adjusted,
  so stored prices can shift later; the ratio we keep does not).
"""
from __future__ import annotations

import logging
import math
import random
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import psycopg

from coordinator.data import TIMEFRAME_DELTA
from coordinator.limits import Limits
from coordinator.safety import NEWS_EXCLUDED
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
# An edge must also clear this (in percent), so rounding noise in a sample with no
# spread at all is never read as skill.
EDGE_FLOOR = 1e-9
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


def round_trip_cost_pct(market: str) -> float:
    """Costs of buying and selling once, in percent (stocks 0.10, crypto 0.70), from the
    backtester's own cost model (no borrow fee)."""
    return 2.0 * COSTS[market].rate * 100.0


def skip_reason(p: dict[str, Any], market: str, limits: Limits) -> str | None:
    """Why the news strategy would not trade this prediction, or None when it would.

    The one definition of "a prediction the strategy would trade", shared by the gate
    (which counts only these) and coordinator.news_trading's entries (which act only on
    these), so the two can never drift apart. Needs the prediction's symbol, direction,
    confidence, already_priced_in and expected_move_pct.
    """
    if p["symbol"] in NEWS_EXCLUDED:
        return "SPY and QQQ are left to the futures paper books"
    if p["direction"] not in ("up", "down"):
        return "no direction"
    if market == "crypto" and p["direction"] == "down":
        return "Alpaca cannot short coins"
    if p["confidence"] < limits.min_confidence:
        return "not confident enough"
    if p["already_priced_in"]:
        return "already priced in"
    if p["expected_move_pct"] < limits.cost_multiple * round_trip_cost_pct(market):
        return "too small a move to beat the costs"
    return None


def would_trade(p: dict[str, Any], market: str, limits: Limits) -> bool:
    """True when the news strategy would trade this prediction (see skip_reason)."""
    return skip_reason(p, market, limits) is None


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
            where.append(f"p.{column} = %s")
            args.append(value)
    if min_confidence is not None:
        where.append("p.confidence >= %s")
        args.append(min_confidence)
    return " AND ".join(where), args


def _mean(values: np.ndarray) -> float | None:
    return float(values.mean()) if len(values) else None


def _sign(direction: str | None) -> float:
    return {"up": 1.0, "down": -1.0}.get(direction or "", 0.0)


def _observation(r: dict[str, Any]) -> dict[str, float]:
    """One scored prediction as numbers. relative_pct is the call's benchmark-relative
    result after its own costs (the move minus the benchmark's, signed by direction);
    twin_relative_pct is the twin's on the same bars, charged the very same costs, so the
    paired difference is the direction skill alone: neither the market's drift nor the
    borrow fee a short pays can make an edge."""
    sign = _sign(r["direction"])
    twin = r["twin_direction"] or twin_direction(r["id"])
    move, bench = float(r["move_pct"] or 0.0), float(r["bench_pct"] or 0.0)
    cost = sign * move - float(r["signed_pct"])
    return {
        "hit": float(bool(r["hit"])), "signed": float(r["signed_pct"]), "twin": float(r["twin_signed_pct"]),
        "relative": sign * (move - bench) - cost, "twin_relative": _sign(twin) * (move - bench) - cost,
        "confidence": float(r["confidence"]),
    }


def _independent(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, float]]]:
    """(direction, observation) per independent call: the scored predictions of one symbol,
    entry day (UTC date of the entry bar; for stocks, the entry bar itself) and direction
    are averaged into one, because they are the same bet on the same price move."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        day = r["entry_ts"].astimezone(timezone.utc).date() if r["entry_ts"] is not None else ("id", r["id"])
        groups.setdefault((r["symbol"], day, r["direction"]), []).append(r)
    out = []
    for (_, _, direction), members in groups.items():
        obs = [_observation(m) for m in members]
        out.append((direction, {k: sum(o[k] for o in obs) / len(obs) for k in obs[0]}))
    return out


def stats(conn: psycopg.Connection, market: str | None = None, source: str | None = None,
          model: str | None = None, min_confidence: float | None = None,
          trading: Limits | None = None) -> dict[str, Any]:
    """How the scored predictions did (filters optional). Unscored rows never count, and
    neither does a replay whose analysis was answered by a different model than the one
    asked (a refusal fallback may have a later training cutoff): fallback_excluded says
    how many scored ones were left out for that.

    With `trading` (the limits), only the predictions the news strategy would trade count
    (would_trade), and they are collapsed into independent calls (one per symbol, entry
    day and direction, averaged): n then counts those calls and `predictions` the rows
    behind them. Without it every call counts on its own (n == predictions).

    Percentages are plain numbers (1.5 means 1.5%). mean_signed_pct and mean_twin_pct are
    absolute results after costs (what a book would earn). edge_pct is the mean paired
    difference between the call and its twin measured against the benchmark
    (mean_relative_pct - mean_twin_relative_pct, both after the call's own costs), and
    edge_se_pct its standard error; a value that cannot be computed (no rows, or one row
    for a standard error) is None.
    """
    where, args = _filters(market, source, model, min_confidence)
    rows = conn.execute(
        "SELECT p.id, p.market, p.symbol, p.direction, p.confidence, p.already_priced_in, p.expected_move_pct,"
        " p.hit, p.signed_pct, p.twin_signed_pct, p.twin_direction, p.move_pct, p.bench_pct, p.entry_ts,"
        " p.scored_at IS NOT NULL AS scored,"
        # A dated snapshot name of the requested model (claude-opus-5-5-YYYYMMDD) is the same model.
        " (p.source = 'live' OR a.served_model = a.requested_model"
        "  OR a.served_model LIKE a.requested_model || '-%%') AS honest"
        " FROM news_predictions p JOIN news_analyses a ON a.id = p.analysis_id"
        f" WHERE {where} ORDER BY p.id", args,
    ).fetchall()
    if trading is not None:
        rows = [r for r in rows if r["direction"] == "none" or would_trade(r, r["market"], trading)]
    fallback = sum(1 for r in rows if r["scored"] and r["direction"] != "none" and not r["honest"])
    rows = [r for r in rows if r["honest"]]
    pending = sum(1 for r in rows if not r["scored"] and r["direction"] != "none")
    none_count = sum(1 for r in rows if r["direction"] == "none")
    done = [r for r in rows if r["scored"] and r["direction"] != "none"]
    if trading is not None:
        calls = _independent(done)
    else:
        calls = [(r["direction"], _observation(r)) for r in done]
    n = len(calls)

    def column(key: str) -> np.ndarray:
        return np.array([o[key] for _, o in calls], dtype=float)

    hit, signed, twin, conf = column("hit"), column("signed"), column("twin"), column("confidence")
    relative, twin_relative = column("relative"), column("twin_relative")
    direction = np.array([d for d, _ in calls], dtype=object)
    diff = relative - twin_relative
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
    # EDGE_FLOOR keeps rounding noise in a zero-variance sample from counting as an edge.
    beats = bool(n >= 2 and edge is not None and se is not None and edge > max(2 * se, EDGE_FLOOR))
    return {
        "n": n,
        "predictions": len(done),
        "pending": pending,
        "none_count": none_count,
        "fallback_excluded": fallback,
        "hit_rate": _mean(hit),
        "mean_signed_pct": _mean(signed),
        "mean_twin_pct": _mean(twin),
        "mean_relative_pct": _mean(relative),
        "mean_twin_relative_pct": _mean(twin_relative),
        "edge_pct": edge,
        "edge_se_pct": se,
        "beats_twin": beats,
        "calibration": calibration,
        "by_direction": by_direction,
    }


def gate(conn: psycopg.Connection, market: str, limits: Limits) -> tuple[bool, str]:
    """May this market's news strategy start paper trading? (passes, one plain sentence)

    Counts the market's scored predictions from both sources that the strategy would
    trade (would_trade: confident enough, not already priced in, a big enough expected
    move, never SPY or QQQ, rises only for coins), one per symbol, entry day and
    direction, leaving out replays answered by a fallback model. Passes only with at least
    min_scored_predictions such independent calls, an edge over the coin-flip twin against
    the market above two standard errors, and a positive mean absolute return after costs.
    """
    if market not in MARKETS:
        raise ValueError(f"unknown market {market!r}")
    s = stats(conn, market=market, trading=limits)
    need = max(int(math.ceil(limits.min_scored_predictions)), 2)
    n = s["n"]
    calls = f"{n} independent call{'s' if n != 1 else ''}"
    if n < need:
        return False, f"Needs {need} independent scored calls it would trade; has {n}."
    edge, se = s["edge_pct"], s["edge_se_pct"]
    if not s["beats_twin"]:
        return False, (f"Does not beat the coin-flip twin against the market yet: "
                       f"edge {edge:+.2f}% ± {se:.2f}% over {calls}.")
    if s["mean_signed_pct"] <= 0:
        return False, (f"Beats the coin-flip twin but loses money after costs: "
                       f"{s['mean_signed_pct']:+.2f}% a call on average over {calls}.")
    return True, f"Ready: {calls}, edge {edge:+.2f}% ± {se:.2f}% over the coin flip, against the market."
