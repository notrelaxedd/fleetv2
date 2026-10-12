"""Smarter settings for model search: learn from every setting tried so far.

Model search used to draw settings completely at random (stocks, crypto) or at random
plus small changes to kept models (futures), and forgot the settings that failed. This
module suggests new settings from the whole history of trials for one model, so
searches stop walking back into regions that already failed.

suggest(search_space, trials, n, rng, explore) returns n new sets of settings:
- search_space is the model's SEARCH_SPACE: {name: (low, high, "int" | "float")} or
  {name: (options, "choice")}. Only these names are returned (callers add defaults).
- trials is the history, [{"params": {...}, "score": float | None, "rejected": str | None}].
  A trial FAILED when its score is None or "rejected" is set (reasons like "gates",
  "too_few_trades", "not_robust", "negative_score"). Names not in the space are ignored
  and a trial may leave names out.
- Every returned value is valid: whole numbers inside the bounds, decimals inside the
  bounds rounded to 3 places, choices from the options. No set repeats one already tried
  or another in the same batch (unless the space is so small that no new set is left).

How it decides (a small "tree-structured Parzen estimator", numpy only):
1. With fewer than 20 scored trials there is nothing to learn from: every set is drawn
   at random, exactly the way draw_params does (so the first rounds behave as before).
2. Otherwise each suggestion is random with probability `explore` (default 0.3), so the
   search never stops looking elsewhere.
3. The rest are learned. Scored trials are sorted best first. The GOOD group is the top
   20% (at least 5 trials). The BAD group is every other scored trial plus ALL failed
   trials, which is how failures push suggestions away from their region.
4. For each setting a density is built per group. A number is scaled to 0..1 (whole
   numbers use the middle of their step) and each trial adds a bell curve (Gaussian)
   centred on its value. All curves share one width, h = 1.06 * spread * count^(-1/5)
   (the usual rule of thumb), kept between 0.05 and 0.3 of the range, and cut off at the
   edges. One flat "any value" curve is added with the weight of a single trial so the
   density never reaches zero. A choice uses its category counts plus one each.
5. 24 candidates are drawn from the GOOD densities. The one with the highest
   (good density / bad density), multiplied over all settings, is returned. That is the
   setting that looks most like the good trials and least like the bad and failed ones.

Same inputs and same rng state always give the same answer. Standard library and numpy.
"""
from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

MIN_SCORED = 20          # fewer scored trials than this: pure random
GOOD_SHARE = 0.20        # the top share of scored trials is "good"
MIN_GOOD = 5             # ... and never fewer than this many
CANDIDATES = 24          # candidates drawn from the good densities per suggestion
H_MIN, H_MAX = 0.05, 0.30   # kernel width limits, in units of the parameter's range
MAX_TRIES = 50           # attempts to find a new (not yet tried) set before giving up on novelty


# ------------------------------------------------------------------ reading the space and the trials


def _spec(search_space: dict[str, tuple], name: str) -> tuple[str, Any, Any]:
    """(kind, low, high) for a number, ("choice", options, None) for a choice."""
    spec = search_space[name]
    if spec[-1] == "choice":
        return "choice", list(spec[0]), None
    low, high, kind = spec
    if kind == "int":
        return "int", int(low), int(high)
    return "float", float(low), float(high)


def _fix(search_space: dict[str, tuple], name: str, value: Any) -> Any:
    """The value as the search would store it: in bounds, whole or rounded to 3 places."""
    kind, low, high = _spec(search_space, name)
    if kind == "choice":
        return value
    if kind == "int":
        return min(max(int(round(value)), low), high)
    return min(max(round(float(value), 3), low), high)


def _key(search_space: dict[str, tuple], names: list[str], params: dict[str, Any]) -> tuple | None:
    """Identity of a full set of settings (None when the set leaves a name out or holds junk)."""
    out = []
    for name in names:
        if name not in params:
            return None
        kind, low, high = _spec(search_space, name)
        value = params[name]
        if kind == "choice":
            if value not in low:
                return None
            out.append(low.index(value))
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        out.append(_fix(search_space, name, number))
    return tuple(out)


def _is_failed(trial: dict) -> bool:
    return bool(trial.get("rejected")) or _score(trial) is None


def _score(trial: dict) -> float | None:
    value = trial.get("score")
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def summary(trials: list[dict]) -> dict:
    """{"tried", "scored", "best_score", "failed_by_reason": {reason: count}}.

    A trial that failed without a reason is counted under "no_score"."""
    failed: dict[str, int] = {}
    scored = 0
    best: float | None = None
    for trial in trials:
        if _is_failed(trial):
            reason = str(trial.get("rejected") or "no_score")
            failed[reason] = failed.get(reason, 0) + 1
            continue
        scored += 1
        score = _score(trial)
        if best is None or score > best:
            best = score
    return {"tried": len(trials), "scored": scored, "best_score": best, "failed_by_reason": failed}


# ------------------------------------------------------------------ random draws (same as draw_params)


def _random_set(search_space: dict[str, tuple], names: list[str], rng: random.Random) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in names:
        kind, low, high = _spec(search_space, name)
        if kind == "choice":
            out[name] = rng.choice(low)
        elif kind == "int":
            out[name] = rng.randint(low, high)
        else:
            out[name] = round(rng.uniform(low, high), 3)
    return out


# ------------------------------------------------------------------ the two kinds of density


def _unit(kind: str, low: Any, high: Any, value: float) -> float:
    """A number's place in 0..1. Whole numbers use the middle of their step."""
    if kind == "int":
        return (min(max(value, low), high) - low + 0.5) / (high - low + 1)
    return (min(max(value, low), high) - low) / (high - low)


def _from_unit(kind: str, low: Any, high: Any, u: float) -> float:
    if kind == "int":
        return low + min(int(u * (high - low + 1)), high - low)
    return low + u * (high - low)


class _Numbers:
    """Density of a number scaled to 0..1: Gaussian bells on the trial values, cut off
    at 0 and 1, plus one flat component (weight of one trial)."""

    def __init__(self, points: list[float]):
        self.centres = np.asarray(points, dtype=float)
        count = len(points)
        spread = float(self.centres.std()) if count > 1 else 0.3
        self.h = min(max(1.06 * spread * count ** -0.2 if count else H_MAX, H_MIN), H_MAX)
        # share of each bell that falls inside 0..1, so the cut-off density still adds up to 1
        mass = [0.5 * (math.erf((1 - c) / (self.h * math.sqrt(2))) - math.erf(-c / (self.h * math.sqrt(2))))
                for c in self.centres]
        self.mass = np.maximum(np.asarray(mass, dtype=float), 1e-6)

    def pdf(self, u: np.ndarray) -> np.ndarray:
        if len(self.centres) == 0:
            return np.ones_like(u)
        z = (u[:, None] - self.centres[None, :]) / self.h
        bells = np.exp(-0.5 * z * z) / (self.h * math.sqrt(2 * math.pi)) / self.mass[None, :]
        return (1.0 + bells.sum(axis=1)) / (len(self.centres) + 1)

    def sample(self, rng: random.Random) -> float:
        pick = rng.randrange(len(self.centres) + 1)
        if pick == len(self.centres):
            return rng.random()
        for _ in range(20):
            u = rng.gauss(float(self.centres[pick]), self.h)
            if 0.0 <= u <= 1.0:
                return u
        return min(max(rng.gauss(float(self.centres[pick]), self.h), 0.0), 1.0)


class _Choices:
    """Category frequencies, each count raised by one so no option ever gets zero."""

    def __init__(self, indexes: list[int], size: int):
        counts = np.ones(size, dtype=float)
        for i in indexes:
            counts[i] += 1
        self.p = counts / counts.sum()

    def sample(self, rng: random.Random) -> int:
        r = rng.random()
        total = 0.0
        for i, p in enumerate(self.p):
            total += float(p)
            if r < total:
                return i
        return len(self.p) - 1


# ------------------------------------------------------------------ the learned suggestions


def _split(trials: list[dict]) -> tuple[list[dict], list[dict]]:
    """(good, bad): good is the top share of scored trials; bad is the rest plus every failure."""
    scored = [(i, t) for i, t in enumerate(trials) if not _is_failed(t)]
    failed = [t for t in trials if _is_failed(t)]
    scored.sort(key=lambda it: (-_score(it[1]), it[0]))
    n_good = min(len(scored), max(MIN_GOOD, math.ceil(GOOD_SHARE * len(scored))))
    good = [t for _, t in scored[:n_good]]
    bad = [t for _, t in scored[n_good:]] + failed
    return good, bad


def _densities(search_space: dict[str, tuple], names: list[str], group: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in names:
        kind, low, high = _spec(search_space, name)
        if kind == "choice":
            idx = [low.index(t["params"][name]) for t in group
                   if name in (t.get("params") or {}) and t["params"][name] in low]
            out[name] = _Choices(idx, len(low))
            continue
        if high <= low:
            continue
        points = []
        for t in group:
            raw = (t.get("params") or {}).get(name)
            try:
                number = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                points.append(_unit(kind, low, high, number))
        out[name] = _Numbers(points)
    return out


def _learned(search_space: dict[str, tuple], names: list[str], good: dict[str, Any], bad: dict[str, Any],
             rng: random.Random) -> dict[str, Any]:
    """Draw CANDIDATES sets from the good densities, return the one with the best good/bad ratio."""
    cands: list[dict[str, Any]] = []
    raw_u: dict[str, list[float]] = {}
    raw_i: dict[str, list[int]] = {}
    for _ in range(CANDIDATES):
        cand: dict[str, Any] = {}
        for name in names:
            kind, low, high = _spec(search_space, name)
            if name not in good:      # a range with a single value
                cand[name] = low if kind != "choice" else None
                continue
            if kind == "choice":
                i = good[name].sample(rng)
                raw_i.setdefault(name, []).append(i)
                cand[name] = low[i]
            else:
                u = good[name].sample(rng)
                raw_u.setdefault(name, []).append(u)
                cand[name] = _fix(search_space, name, _from_unit(kind, low, high, u))
        cands.append(cand)
    ratio = np.zeros(CANDIDATES)
    for name in names:
        if name not in good:
            continue
        if name in raw_i:
            idx = np.asarray(raw_i[name])
            ratio += np.log(good[name].p[idx]) - np.log(bad[name].p[idx])
        else:
            u = np.asarray(raw_u[name])
            ratio += np.log(np.maximum(good[name].pdf(u), 1e-12)) - np.log(np.maximum(bad[name].pdf(u), 1e-12))
    order = np.argsort(-ratio, kind="stable")
    return {"cands": cands, "order": [int(i) for i in order]}


def suggest(search_space: dict[str, tuple], trials: list[dict], n: int, rng: random.Random,
            explore: float = 0.3) -> list[dict[str, Any]]:
    """n new sets of settings (only the names in search_space); see the module docstring."""
    names = sorted(search_space)
    if n <= 0 or not names:
        return [{} for _ in range(max(n, 0))]
    taken: set[tuple] = set()
    for trial in trials:
        key = _key(search_space, names, trial.get("params") or {})
        if key is not None:
            taken.add(key)

    scored = [t for t in trials if not _is_failed(t)]
    learn = len(scored) >= MIN_SCORED
    if learn:
        good_trials, bad_trials = _split(trials)
        good = _densities(search_space, names, good_trials)
        bad = _densities(search_space, names, bad_trials)

    out: list[dict[str, Any]] = []
    for _ in range(n):
        params: dict[str, Any] | None = None
        key = None
        if learn and rng.random() >= explore:
            for _ in range(3):
                pack = _learned(search_space, names, good, bad, rng)
                for i in pack["order"]:
                    cand = pack["cands"][i]
                    key = _key(search_space, names, cand)
                    if key not in taken:
                        params = cand
                        break
                if params is not None:
                    break
        if params is None:
            for _ in range(MAX_TRIES):
                params = _random_set(search_space, names, rng)
                key = _key(search_space, names, params)
                if key not in taken:
                    break
        taken.add(key)
        out.append(params)
    return out
