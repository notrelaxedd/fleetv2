"""Stress tests for a stock or crypto model found by model search, on TRAINING prices only.

Picking the best of many random settings by one training backtest mostly picks luck
(on random-walk prices, 80 random Dip buyer settings had a -0.27 correlation between
training and held-out ROI). The futures search already guards against that; this
brings the same checks to the stock and crypto search. report() runs them all on one
candidate (each round's best) and returns a small JSON-ready dict, stored as
metrics["robustness"]. Nothing here reads held-out prices, talks to the network or
keeps state: the same inputs and seed give the same answer.

- score(metrics): yearly return over buy and hold (the owner's "vs. buy and hold",
  scaled to a year), see score() for why. It does not grow with the number of trades:
  past MIN_TRADES more trades only help if they earn more than they cost. Below it, a
  gain counts for less (trades / MIN_TRADES of it); a loss counts in full.
- Folds (fold_bounds, fold_score): training is cut into 4 back-to-back parts, each
  backtested on its own (the bars before a part are its warm-up, never traded). The
  score is the WORST part's score, as on the futures side, so a setting that only
  worked in one stretch scores low. Each part needs MIN_TRADES / 4 trades for a full
  score. "inconsistent": some parts beat buy and hold and some did not.
- Neighbours (neighbour_check): every setting moved 10% of its search range down and
  up. "fragile": the candidate scores above zero but the middle (median) neighbour
  scores under half of it (ROBUST_SHARE, the futures rule), or below zero: the
  candidate sits on a sharp peak, which usually means luck.
- Double costs (cost_stress): "cost_sensitive" when the score is not above zero with
  every fill's slippage and fees doubled.
- Coin flip (coin_flip_baseline): random-timing twins with the same symbols, decision
  cadence, typical number of positions, share of money invested and about as many
  round trips (so about the same costs), but which symbols and when are seeded coin
  flips. "no_better_than_random" when the candidate scores under the 80th percentile
  of its twins' scores.

report()["passed"] is true when no flag is raised and the score is above zero.

Cost, in passes over the training bars (the 4 fold runs together make one pass): 1 for
the candidate, up to 2 per setting for the neighbours (the models have 3 to 5
settings), 1 at double costs, then 1 + n_draws twins (default 20). A twin costs only
the backtester's own loop, not the model's, so for a slow model the neighbours
dominate: about 2 x settings + 2 model backtests in all.
"""
from __future__ import annotations

import math
import random
import statistics
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import numpy as np

from fleet2.models.base import params_with_defaults
from fleet2.sim.backtest import Limits, Run, run_backtest
from fleet2.sim.control import JobStopped
from fleet2.sim.marketdata import MarketData
from fleet2.sim.metrics import MIN_TRADES, summarize
from fleet2.universe import MARKETS

N_FOLDS = 4
ROBUST_SHARE = 0.5  # the median neighbour must score at least half the candidate (as futures search)
REL_STEP = 0.10  # neighbours: each setting moved by 10% of its search range
COST_MULTIPLIER = 2.0
COIN_FLIP_DRAWS = 20
COIN_FLIP_PERCENTILE = 80.0
FLAGS = ("inconsistent", "fragile", "cost_sensitive", "no_better_than_random")
_ANNUAL_FLOOR = 1.0  # a period shorter than a year is not scaled up to a year

ShouldStop = Callable[[], bool] | None
Progress = Callable[[float], None] | None


# ------------------------------------------------------------------ the score


def _yearly(total: float, years: float) -> float:
    """A total return as a return per year (compounded); never scaled UP from under a year,
    so one lucky month does not become a huge yearly figure."""
    if total <= -1.0:
        return -1.0
    return (1.0 + total) ** (1.0 / max(years, _ANNUAL_FLOOR)) - 1.0


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def score(metrics: dict[str, Any] | None, min_trades: float = MIN_TRADES) -> float:
    """Yearly return over buy and hold, times trades / min_trades (at most 1) when it is a gain.

    Why this: buy and hold is the owner's yardstick ("below zero means the model isn't
    earning its keep"). Measured against it, a bear-market stretch does not fail every
    model that buys and sells (sitting in cash through a crash counts as beating it),
    which matters because fold_score() keeps the worst part. Per year, so parts and
    periods of different lengths compare. Steadiness comes from the worst part across
    time (fold_score) and the neighbour and coin-flip checks. More trades never raise it
    by themselves: costs are already in the return, so churn shows up as less return.
    Too few trades shrink a gain (a few lucky trades count for less) but not a loss.
    Missing numbers: no return known scores 0 (never kept); no buy and hold known
    compares with cash."""
    m = metrics or {}
    roi = _finite(m.get("roi"))
    if roi is None:
        return 0.0
    years = _finite(m.get("years")) or 1.0
    bench = _finite(m.get("benchmark_roi"))
    excess = _yearly(roi, years) - (_yearly(bench, years) if bench is not None else 0.0)
    trades = _finite(m.get("trades")) or 0.0
    if excess > 0 and min_trades > 0:
        excess *= min(1.0, max(trades, 0.0) / float(min_trades))
    return float(excess)


# ------------------------------------------------------------------ folds


def training_bounds(module: ModuleType, params: dict[str, Any] | None, split: int) -> tuple[int, int]:
    """The training period model search backtests on (as fleet2.worker.search_job.evaluate):
    from the model's warm-up to the first held-out bar."""
    return min(int(module.warmup(params_with_defaults(module, params))), split - 1), split


def fold_bounds(start: int, stop: int, n_folds: int = N_FOLDS) -> list[tuple[int, int]]:
    """`n_folds` back-to-back, non-overlapping [a, b) parts covering [start, stop), sizes
    within one bar of each other. Pass the training start AFTER the warm-up (as
    training_bounds gives it): later parts warm up on the bars before them, which the
    backtester allows (they are history, never traded)."""
    n_folds = int(n_folds)
    if n_folds < 1 or stop - start < 2 * n_folds:
        raise ValueError(f"cannot cut bars {start}..{stop} into {n_folds} parts of at least 2 bars")
    edges = np.linspace(start, stop, n_folds + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def _market(data: MarketData) -> tuple[str | None, int]:
    spec = MARKETS[data.market]
    bench = spec["benchmark"] if spec["benchmark"] in data.symbols else None
    return bench, int(spec["bars_per_year"])


def _check(should_stop: ShouldStop) -> None:
    if should_stop is not None and should_stop():
        raise JobStopped()


def _r(x: Any, digits: int = 6) -> float | None:
    v = _finite(x)
    return None if v is None else round(v, digits)


def _compact(metrics: dict[str, Any], s: float) -> dict[str, Any]:
    """One part's numbers for storage: no curve."""
    keys = ("roi", "benchmark_roi", "vs_buy_and_hold", "max_drawdown", "sharpe", "t_stat", "years")
    out: dict[str, Any] = {k: _r(metrics.get(k)) for k in keys}
    out.update(trades=int(metrics.get("trades") or 0), start=metrics.get("start"), end=metrics.get("end"),
               score=_r(s))
    return out


def _fold_runs(data: MarketData, module: Any, params: dict[str, Any] | None, folds: list[tuple[int, int]],
               limits: Limits, should_stop: ShouldStop, cost_multiplier: float = 1.0) -> list[Run]:
    bench, bars_per_year = _market(data)
    runs = []
    for a, b in folds:
        _check(should_stop)
        runs.append(run_backtest(data, module, params, a, b, limits, bench, bars_per_year, None, should_stop,
                                 cost_multiplier))
    return runs


def _aggregate(runs: list[Run], n_folds: int) -> dict[str, Any]:
    per_fold_min = MIN_TRADES / max(1, n_folds)
    folds = []
    for run in runs:
        m = summarize(run)
        folds.append(_compact(m, score(m, per_fold_min)))
    scores = [f["score"] if f["score"] is not None else -1.0 for f in folds]
    return {"folds": folds, "score": _r(min(scores)) if scores else 0.0,
            "trades": sum(f["trades"] for f in folds), "positive_folds": sum(1 for s in scores if s > 0)}


def fold_score(data: MarketData, module: Any, params: dict[str, Any] | None, folds: list[tuple[int, int]],
               limits: Limits = Limits(), should_stop: ShouldStop = None,
               cost_multiplier: float = 1.0) -> dict[str, Any]:
    """Backtest each part on its own and score it (score() with MIN_TRADES / parts trades
    needed per part). {"folds": [compact numbers per part, no curves], "score": the worst
    part's score, "trades": all parts together, "positive_folds": parts scoring above 0}.
    The worst part, not the average: a setting is only as good as its worst stretch
    (the futures search's rule too)."""
    return _aggregate(_fold_runs(data, module, params, folds, limits, should_stop, cost_multiplier), len(folds))


# ------------------------------------------------------------------ neighbours


def neighbours(module: Any, params: dict[str, Any] | None, search_space: dict[str, Any] | None = None,
               rel_step: float = REL_STEP) -> list[tuple[str, int, dict[str, Any]]]:
    """(setting, -1 or +1, settings) for every numeric setting moved by rel_step of its
    search range down and up (whole numbers by at least 1), kept inside the range; a
    move that changes nothing (already at the edge) is left out. At most 2 per setting."""
    base = params_with_defaults(module, params)
    space = getattr(module, "SEARCH_SPACE", {}) if search_space is None else search_space
    out = []
    for name in sorted(space):
        spec = space[name]
        if name not in base or len(spec) != 3 or spec[2] not in ("int", "float"):
            continue
        low, high, kind = spec
        span = float(high) - float(low)
        for sign in (-1, 1):
            if kind == "int":
                moved: Any = int(min(max(int(base[name]) + sign * max(1, int(round(rel_step * span))), int(low)),
                                     int(high)))
            else:
                moved = round(min(max(float(base[name]) + sign * rel_step * span, float(low)), float(high)), 6)
            if moved != base[name]:
                out.append((name, sign, {**base, name: moved}))
    return out


def neighbour_check(data: MarketData, module: Any, params: dict[str, Any] | None, folds: list[tuple[int, int]],
                    limits: Limits = Limits(), should_stop: ShouldStop = None,
                    search_space: dict[str, Any] | None = None, rel_step: float = REL_STEP,
                    base_score: float | None = None, progress: Progress = None) -> dict[str, Any]:
    """Fold scores of every neighbour (see neighbours()). "fragile" when the candidate
    scores above zero and their median is under ROBUST_SHARE x its score (so also when
    the median is below zero). A candidate at or below zero fails anyway and is not
    called fragile. `base_score` saves re-running the candidate when it is known."""
    if base_score is None:
        base_score = fold_score(data, module, params, folds, limits, should_stop)["score"]
    near = neighbours(module, params, search_space, rel_step)
    rows = []
    for k, (name, sign, moved) in enumerate(near):
        _check(should_stop)
        s = fold_score(data, module, moved, folds, limits, should_stop)["score"]
        rows.append({"setting": name, "value": moved[name], "score": s})
        if progress is not None:
            progress((k + 1) / len(near))
    if not rows:
        return {"tried": 0, "median": None, "share": None, "worst": None, "scores": [], "fragile": False}
    middle = float(statistics.median(r["score"] for r in rows))
    worst = min(rows, key=lambda r: r["score"])
    return {"tried": len(rows), "median": _r(middle),
            "share": _r(middle / base_score) if base_score > 0 else None,
            "worst": worst, "scores": rows,
            "fragile": bool(base_score > 0 and middle < ROBUST_SHARE * base_score)}


# ------------------------------------------------------------------ double costs


def cost_stress(data: MarketData, module: Any, params: dict[str, Any] | None, folds: list[tuple[int, int]],
                limits: Limits = Limits(), should_stop: ShouldStop = None,
                multiplier: float = COST_MULTIPLIER) -> dict[str, Any]:
    """The fold score with every fill's slippage and fees x `multiplier`. "cost_sensitive"
    when it is not above zero: the edge is too thin to survive worse fills."""
    got = fold_score(data, module, params, folds, limits, should_stop, cost_multiplier=multiplier)
    return {"multiplier": float(multiplier), "score": got["score"], "trades": got["trades"],
            "positive_folds": got["positive_folds"], "cost_sensitive": bool(got["score"] <= 0)}


# ------------------------------------------------------------------ coin-flip twins


def _proxy(module: Any, **overrides: Any) -> SimpleNamespace:
    """A module-like copy of a model (every public name) with some names replaced."""
    names = {k: v for k, v in vars(module).items() if not k.startswith("_")}
    names.update(overrides)
    return SimpleNamespace(**names)


def _recording(module: Any, log: list[dict[str, float]]) -> SimpleNamespace:
    """The model, unchanged, except that every decision it makes is appended to `log`."""

    def target_positions(history: Any, params: dict[str, Any]) -> Any:
        got = module.target_positions(history, params)
        if isinstance(got, dict):
            log.append({str(s): float(w) for s, w in got.items() if float(w) > 0})
        return got

    return _proxy(module, target_positions=target_positions)


def habits(decisions: list[dict[str, float]], runs: list[Run], every: int = 1) -> dict[str, Any]:
    """What the twin copies from the model's fold runs: the number of positions it holds
    when invested (the median), the typical weight of one position, the average share of
    money invested over all decisions, and how long a position is held, in whole decision
    steps of `every` bars: average positions x bars / round trips, so the twin makes
    about as many round trips (and pays about as much in costs) as the model."""
    counts = [len(d) for d in decisions if d]
    weights = [w for d in decisions for w in d.values()]
    share = min(1.0, float(np.mean([sum(d.values()) for d in decisions]))) if decisions else 0.0
    slots = max(1, int(round(statistics.median(counts)))) if counts else 0
    weight = min(float(np.mean(weights)), 1.0 / slots) if slots and weights else 0.0
    bars = sum(int(r.equity.shape[0]) for r in runs)
    trips = sum(len(r.trades) for r in runs)
    every = max(1, int(every))
    hold = share / weight * bars / trips if trips and weight > 0 else float(bars or every)
    return {"slots": slots, "weight": round(weight, 6), "share": round(share, 6),
            "hold_bars": max(1, int(round(hold / every))) * every, "trips": trips}


def coin_flip_twin(module: Any, habit: dict[str, Any], seed: int, draw: int | str) -> SimpleNamespace:
    """A model the backtester accepts that holds `slots` positions of `weight` each, each
    switched on with probability share / (slots x weight) for blocks of `hold_bars` bars
    and pointed at a symbol picked at random, both by random.Random(seed, draw, slot,
    the bar its block starts at). It decides from the bar index alone (plus whether the
    symbol has a price yet, from the history it is shown), so it never looks ahead and
    the same seed always gives the same twin. Cadence and warm-up are the model's."""
    symbols = tuple(module.SYMBOLS)
    slots, weight = int(habit["slots"]), float(habit["weight"])
    hold = max(1, int(habit["hold_bars"]))
    on = min(1.0, float(habit["share"]) / (slots * weight)) if slots and weight > 0 else 0.0
    phases = [random.Random(f"{seed}:{draw}:{k}:phase").randrange(hold) for k in range(slots)]

    def target_positions(history: Any, params: dict[str, Any]) -> dict[str, float]:
        i = len(history)
        out: dict[str, float] = {}
        if on <= 0.0 or not symbols:
            return out
        for k in range(slots):
            block_start = ((i + phases[k]) // hold) * hold - phases[k]
            rng = random.Random(f"{seed}:{draw}:{k}:{block_start}")
            if rng.random() >= on:
                continue
            symbol = symbols[rng.randrange(len(symbols))]
            last = history.close(symbol, bars=1)
            if last.size and np.isfinite(last[-1]) and last[-1] > 0:
                out[symbol] = min(1.0, out.get(symbol, 0.0) + weight)
        total = sum(out.values())
        return {s: w / total for s, w in out.items()} if total > 1.0 else out

    return _proxy(module, NAME=f"Coin flip {draw}", target_positions=target_positions)


def _percentile(real: float, others: list[float]) -> float:
    """Share of `others` below `real` (ties count half), in percent."""
    if not others:
        return 100.0
    below = sum(1 for o in others if o < real) + 0.5 * sum(1 for o in others if o == real)
    return round(100.0 * below / len(others), 2)


def _calibrated(data: MarketData, module: Any, params: dict[str, Any] | None, folds: list[tuple[int, int]],
                limits: Limits, should_stop: ShouldStop, habit: dict[str, Any], seed: int,
                every: int) -> dict[str, Any]:
    """The habits with the holding time corrected once so a twin makes about as many round
    trips as the model: slots that land on the same symbol, or on the one they held
    before, make fewer trades than the plain formula expects. One extra twin backtest."""
    trips = int(habit.get("trips") or 0)
    if not trips or habit["slots"] == 0 or habit["weight"] <= 0:
        return habit
    probe = fold_score(data, coin_flip_twin(module, habit, seed, "calibrate"), params, folds, limits, should_stop)
    if not probe["trades"]:
        return habit
    every = max(1, int(every))
    hold = habit["hold_bars"] * probe["trades"] / trips
    return {**habit, "hold_bars": max(1, int(round(hold / every))) * every}


def _baseline(data: MarketData, module: Any, params: dict[str, Any] | None, folds: list[tuple[int, int]],
              limits: Limits, should_stop: ShouldStop, habit: dict[str, Any], real: float, n_draws: int,
              seed: int, progress: Progress = None) -> dict[str, Any]:
    habit = _calibrated(data, module, params, folds, limits, should_stop, habit, seed,
                        module.rebalance_every(params_with_defaults(module, params)))
    twins, trades = [], []
    for draw in range(int(n_draws)):
        _check(should_stop)
        got = fold_score(data, coin_flip_twin(module, habit, seed, draw), params, folds, limits, should_stop)
        twins.append(got["score"])
        trades.append(got["trades"])
        if progress is not None:
            progress((draw + 1) / n_draws)
    pct = _percentile(real, twins)
    return {"score": _r(real), "twin_scores": twins, "twin_median": _r(statistics.median(twins)) if twins else None,
            "twin_trades": int(round(statistics.median(trades))) if trades else None,
            "percentile": pct, "draws": len(twins), "seed": int(seed), "habits": habit,
            "no_better_than_random": bool(twins) and pct < COIN_FLIP_PERCENTILE}


def _measured(data: MarketData, module: Any, params: dict[str, Any], folds: list[tuple[int, int]], limits: Limits,
              should_stop: ShouldStop) -> tuple[dict[str, Any], dict[str, Any]]:
    """The candidate's fold score, and its habits for the twins, from the same fold runs."""
    log: list[dict[str, float]] = []
    runs = _fold_runs(data, _recording(module, log), params, folds, limits, should_stop)
    return _aggregate(runs, len(folds)), habits(log, runs, module.rebalance_every(params))


def coin_flip_baseline(data: MarketData, module: Any, params: dict[str, Any] | None,
                       folds: list[tuple[int, int]], limits: Limits = Limits(), should_stop: ShouldStop = None,
                       n_draws: int = COIN_FLIP_DRAWS, seed: int = 0) -> dict[str, Any]:
    """The model's fold score against `n_draws` coin-flip twins' (coin_flip_twin; habits
    measured on the model's own fold runs). {"score", "twin_scores", "percentile" (share
    of twins it beats), "no_better_than_random": under COIN_FLIP_PERCENTILE, ...}.
    Costs 1 backtest of the model and n_draws of cheap twins."""
    params = params_with_defaults(module, params)
    base, habit = _measured(data, module, params, folds, limits, should_stop)
    return _baseline(data, module, params, folds, limits, should_stop, habit, base["score"], n_draws, seed)


# ------------------------------------------------------------------ everything together


def report(data: MarketData, module: Any, params: dict[str, Any] | None, start: int, stop: int,
           limits: Limits = Limits(), should_stop: ShouldStop = None, n_folds: int = N_FOLDS,
           search_space: dict[str, Any] | None = None, n_draws: int = COIN_FLIP_DRAWS, seed: int = 0,
           progress: Progress = None) -> dict[str, Any]:
    """Every check on one candidate over the training bars [start, stop) (training_bounds
    gives them): {"score", "folds", "trades", "positive_folds", "neighbours",
    "cost_stress", "coin_flip", "flags", "passed"}. JSON-ready, no curves. Only ever
    pass training bars: nothing here knows where the held-out period begins."""
    params = params_with_defaults(module, params)
    folds = fold_bounds(start, stop, n_folds)
    n_near = len(neighbours(module, params, search_space))
    steps = 3 + n_near + int(n_draws)  # passes over the training bars: model, neighbours, double, probe, twins

    def part(first: int, size: int) -> Progress:
        """Progress of a stage that covers backtests first .. first + size of `steps`."""
        if progress is None:
            return None
        return lambda fraction: progress(min(1.0, (first + size * fraction) / steps))

    base, habit = _measured(data, module, params, folds, limits, should_stop)
    near = neighbour_check(data, module, params, folds, limits, should_stop, search_space, base_score=base["score"],
                           progress=part(1, n_near))
    costly = cost_stress(data, module, params, folds, limits, should_stop)
    coin = _baseline(data, module, params, folds, limits, should_stop, habit, base["score"], n_draws, seed,
                     part(3 + n_near, int(n_draws)))
    flags = []
    if 0 < base["positive_folds"] < len(folds):
        flags.append("inconsistent")
    if near["fragile"]:
        flags.append("fragile")
    if costly["cost_sensitive"]:
        flags.append("cost_sensitive")
    if coin["no_better_than_random"]:
        flags.append("no_better_than_random")
    return {"score": base["score"], "folds": base["folds"], "trades": base["trades"],
            "positive_folds": base["positive_folds"], "neighbours": near, "cost_stress": costly,
            "coin_flip": coin, "flags": flags, "passed": not flags and base["score"] > 0}
