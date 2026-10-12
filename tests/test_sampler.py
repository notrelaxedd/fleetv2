"""The model-search sampler: valid settings, no repeats, random at first, learns from failures."""
from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from fleet2.models.base import draw_params as stock_draw
from fleet2.sim import sampler

STOCK_SPACE = {"fast": (3, 30, "int"), "slow": (20, 200, "int"), "band": (0.5, 3.0, "float")}
FUTURES_SPACE = {
    "lookback": (5, 60, "int"),
    "threshold": (0.1, 2.5, "float"),
    "bar_minutes": ((1, 3, 5, 15), "choice"),
    "mode": (("trend", "revert"), "choice"),
}


def _valid(space, params):
    assert set(params) == set(space)
    for name, spec in space.items():
        value = params[name]
        if spec[-1] == "choice":
            assert value in spec[0]
        else:
            low, high, kind = spec
            assert low <= value <= high
            if kind == "int":
                assert isinstance(value, int)
            else:
                assert value == round(value, 3)


def _trial(params, score=None, rejected=None):
    return {"params": params, "score": score, "rejected": rejected}


def _history(space, n, seed=0, fail_every=0):
    rng = random.Random(seed)
    out = []
    for i, p in enumerate(sampler.suggest(space, [], n, rng)):
        if fail_every and i % fail_every == 0:
            out.append(_trial(p, None, "gates"))
        else:
            out.append(_trial(p, rng.uniform(-1, 1)))
    return out


@pytest.mark.parametrize("space", [STOCK_SPACE, FUTURES_SPACE])
def test_every_value_is_valid_over_many_seeds(space):
    for seed in range(25):
        trials = _history(space, 40, seed, fail_every=4)
        for params in sampler.suggest(space, trials, 12, random.Random(seed), explore=0.3):
            _valid(space, params)
        for params in sampler.suggest(space, [], 5, random.Random(seed)):
            _valid(space, params)


def test_odd_trials_are_tolerated():
    trials = _history(FUTURES_SPACE, 30)
    trials += [_trial({"lookback": 10}, 0.5), _trial({}, 0.2), {"params": None, "score": 0.1},
               _trial({"lookback": "x", "mode": "nope", "other": 3}, 0.4), _trial({"bar_minutes": 5}, None, "gates")]
    for params in sampler.suggest(FUTURES_SPACE, trials, 6, random.Random(1)):
        _valid(FUTURES_SPACE, params)
    assert sampler.suggest(FUTURES_SPACE, trials, 0, random.Random(1)) == []


def test_deterministic():
    trials = _history(FUTURES_SPACE, 50, 3, fail_every=3)
    a = sampler.suggest(FUTURES_SPACE, trials, 10, random.Random(42))
    b = sampler.suggest(FUTURES_SPACE, trials, 10, random.Random(42))
    assert a == b
    assert a != sampler.suggest(FUTURES_SPACE, trials, 10, random.Random(43))


def test_no_duplicates_of_tried_or_within_batch():
    space = {"a": (0, 5, "int"), "b": ((1, 2, 3), "choice")}   # 18 combinations, 9 of them tried
    trials = [_trial({"a": a, "b": b}, 0.1 * a + b if (a + b) % 3 else None) for a in range(3) for b in (1, 2, 3)]
    for seed in range(10):
        got = sampler.suggest(space, trials, 4, random.Random(seed), explore=0.2)
        keys = [(p["a"], p["b"]) for p in got]
        assert len(set(keys)) == 4
        assert all(k[0] >= 3 for k in keys)
    for space_, trials_ in ((STOCK_SPACE, _history(STOCK_SPACE, 60, 2)), (FUTURES_SPACE, _history(FUTURES_SPACE, 60, 2))):
        tried = {tuple(sorted(t["params"].items())) for t in trials_}
        got = sampler.suggest(space_, trials_, 40, random.Random(5))
        keys = [tuple(sorted(p.items())) for p in got]
        assert len(set(keys)) == len(keys)
        assert not set(keys) & tried


def test_pure_random_under_threshold_matches_draw_params():
    module = SimpleNamespace(SEARCH_SPACE=STOCK_SPACE)
    few = _history(STOCK_SPACE, sampler.MIN_SCORED - 1, 7)
    got = sampler.suggest(STOCK_SPACE, few, 1, random.Random(11))
    assert got == [stock_draw(module, random.Random(11))]
    # failed trials do not count towards the threshold
    many_failed = few + [_trial({"fast": 4, "slow": 30, "band": 1.0 + i / 100}, None, "gates") for i in range(30)]
    assert sampler.suggest(STOCK_SPACE, many_failed, 1, random.Random(11)) == got


def test_failed_trials_repel():
    space = {"x": (0.0, 1.0, "float"), "k": ((0, 1), "choice")}
    rng = random.Random(0)
    trials = []
    for _ in range(30):                       # all equally good everywhere ...
        trials.append(_trial({"x": round(rng.random(), 3), "k": rng.choice((0, 1))}, 1.0 + rng.random() * 0.01))
    for _ in range(40):                       # ... but x > 0.6 with k == 1 always fails
        trials.append(_trial({"x": round(rng.uniform(0.6, 1.0), 3), "k": 1}, None, "gates"))
    hits = total = 0
    for seed in range(20):
        for p in sampler.suggest(space, trials, 12, random.Random(seed), explore=0.0):
            total += 1
            hits += p["x"] > 0.6 and p["k"] == 1
    assert hits / total < 0.06            # random would land there 20% of the time


# ------------------------------------------------------------------ learning on a synthetic objective

SYNTH = {"x": (0.0, 1.0, "float"), "y": (0.0, 1.0, "float"), "k": (1, 20, "int"), "c": (("a", "b", "c", "d"), "choice")}


def _evaluate(p):
    """(score, rejected). Best near x=0.7, y=0.2, k=12, c='b'; x<0.4 with y>0.5 always fails."""
    if p["x"] < 0.4 and p["y"] > 0.5:
        return None, "gates"
    score = -((p["x"] - 0.7) ** 2 + (p["y"] - 0.2) ** 2) - 0.002 * abs(p["k"] - 12) + (0.3 if p["c"] == "b" else 0.0)
    return score, None


def _run(seed, learned, rounds=15, per_round=12):
    rng = random.Random(seed)
    trials, history = [], []
    for _ in range(rounds):
        batch = sampler.suggest(SYNTH, trials if learned else [], per_round, rng)
        values = []
        for p in batch:
            score, why = _evaluate(p)
            trials.append(_trial(p, score, why))
            values.append(-1.0 if score is None else score)    # a failed setting counts as -1
        history.append(values)
    best = max(t["score"] for t in trials if t["score"] is not None)
    late = sum(sum(v) / len(v) for v in history[-5:]) / 5
    return best, late


def test_learning_beats_random_with_the_same_budget():
    seeds = range(8)
    learned = [_run(s, True) for s in seeds]
    random_ = [_run(s, False) for s in seeds]
    mean = lambda xs: sum(xs) / len(xs)
    assert mean([b for b, _ in learned]) > mean([b for b, _ in random_])
    assert mean([m for _, m in learned]) > mean([m for _, m in random_]) + 0.1
    wins = sum(1 for (lb, lm), (rb, rm) in zip(learned, random_) if lm > rm)
    assert wins >= 7


def test_summary():
    trials = [_trial({}, 0.5), _trial({}, 1.5), _trial({}, None, "gates"), _trial({}, -1.0, "negative_score"),
              _trial({}, None, "gates"), _trial({}, None)]
    assert sampler.summary(trials) == {"tried": 6, "scored": 2, "best_score": 1.5,
                                       "failed_by_reason": {"gates": 2, "negative_score": 1, "no_score": 1}}
    assert sampler.summary([]) == {"tried": 0, "scored": 0, "best_score": None, "failed_by_reason": {}}
