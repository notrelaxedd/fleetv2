"""Futures model search on the worker: the score (worst of four training parts' daily
Sharpe at double slippage), the gates, proposals (half random, half small changes to
kept models with the parent in the seed text), the robustness check, the process pool
and its stop, prices cached by ETag, and the separate periods: choosing reads training
prices only, held-out prices only for a model already chosen, lockbox prices never."""
from __future__ import annotations

import random
from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from coordinator.futures_data import FakeFuturesSource, split_days
from fleet2.models.futures import REGISTRY
from fleet2.models.futures import recipe as R
from fleet2.models.futures.base import params_with_defaults
from fleet2.sim import futures_backtest as fb
from fleet2.sim import futures_data as wfd
from fleet2.sim import futures_stats, topstep
from fleet2.sim.cme_session import as_date
from fleet2.sim.control import JobStopped
from fleet2.worker import futures_jobs, futures_search_job as fs

RULES = topstep.Rules(max_payout=2_000.0, combine_monthly=50.0, activation=100.0,
                      commission_per_side={"MES": 0.62, "MNQ": 0.62}, twin_seeds=2, max_sizes=2)


@pytest.fixture(scope="module")
def full() -> wfd.FuturesData:
    src = FakeFuturesSource(seed=8, first=date(2023, 1, 3), last=date(2024, 12, 31))
    return wfd.build("synthetic", {s: src.series(s) for s in ("MES", "MNQ")})


def planted(data: wfd.FuturesData) -> wfd.FuturesData:
    """The same prices with an edge planted in them: every afternoon (from 300 minutes
    after the open) drifts half a point a minute in the direction of the first hour's
    move. Trend day can find it; a model with no edge would not."""
    close = data.close.copy()
    for k in range(close.shape[0]):
        for d in range(data.n_days):
            a, b = int(data.day_start[d]), int(data.day_end[d])
            if b - a < 390:
                continue
            side = np.sign(close[k, a + 59] - data.open[k, a])
            close[k, a + 300:b] += side * 0.5 * np.arange(1, b - a - 299)
    close = np.round(close * 4) / 4
    open_ = data.open.copy()
    for d in range(data.n_days):
        a, b = int(data.day_start[d]), int(data.day_end[d])
        open_[:, a + 1:b] = close[:, a:b - 1]
    high = np.maximum(open_, close) + 0.25
    low = np.minimum(open_, close) - 0.25
    return wfd.FuturesData(**{**data.__dict__, "open": open_, "high": high, "low": low, "close": close})


@pytest.fixture(scope="module")
def edge(full) -> wfd.FuturesData:
    return planted(full)


@pytest.fixture(scope="module")
def periods(full) -> dict:
    return split_days([as_date(int(d)) for d in full.days])


@pytest.fixture(autouse=True)
def fewer_days(monkeypatch):
    monkeypatch.setattr(futures_jobs, "MIN_DAYS_TRADED", 20)


def cut(data: wfd.FuturesData, iso: str) -> wfd.FuturesData:
    return data.until_day(data.day_index(futures_jobs.day_int(iso), side="right"))


class FakeCache:
    """Stands in for PriceCache: each period cut from one synthetic history, like the
    coordinator does, recording every request and counting downloads per ETag."""

    def __init__(self, data: wfd.FuturesData, periods: dict, held_out: wfd.FuturesData | None = None) -> None:
        self.by = {"train": cut(data, periods["train_end"]),
                   "held_out": held_out if held_out is not None else cut(data, periods["held_out_end"]),
                   "lockbox": cut(data, periods["lockbox_end"])}
        self.requested: list[str] = []
        self.downloads = 0
        self._seen: set[str] = set()

    def get(self, through: str, job_id: str | None = None):
        self.requested.append(through)
        if through not in self._seen:
            self._seen.add(through)
            self.downloads += 1
        return self.by[through]


# ------------------------------------------------------------------ the score and the gates


def test_the_score_is_the_worst_of_four_parts_at_double_slippage(full, periods):
    train = cut(full, periods["train_end"])
    module = REGISTRY["pullback"]
    params = params_with_defaults(module, {"bar_minutes": 5})
    numbers = futures_jobs.training_numbers(train, module, params, RULES)
    costs = topstep.costs_of(RULES)
    double = fb.run(train, module, params, costs.doubled(), 1, topstep.day_rules(RULES))
    parts = [fb.daily_sharpe(p) for p in np.array_split(double.pnl, 4)]
    assert numbers["part_sharpes"] == parts and numbers["score"] == pytest.approx(min(parts))
    normal = fb.run(train, module, params, costs, 1, topstep.day_rules(RULES))
    if numbers["gates"]["days_traded"] and numbers["gates"]["profit_double"]:
        assert numbers["pnl_normal"] == pytest.approx(round(float(normal.pnl.sum()), 2), abs=0.02)
    else:
        assert numbers["pnl_normal"] is None and numbers["passes"] is False  # failed already: no second run
    kept = futures_jobs.training_numbers(train, module, params, RULES, keep_daily=True)
    assert kept["pnl_normal"] == pytest.approx(round(float(normal.pnl.sum()), 2), abs=0.02)
    assert kept["daily"]["pnl"] == [round(float(x), 2) for x in normal.pnl]
    assert numbers["pnl_double"] == pytest.approx(round(float(double.pnl.sum()), 2), abs=0.02)
    assert numbers["days_traded"] == int(np.count_nonzero(double.trades))
    assert numbers["gates"]["days_traded"] == (numbers["days_traded"] >= 20)
    assert numbers["passes"] == all(numbers["gates"].values())
    assert numbers["end"] <= futures_jobs.day_int(periods["train_end"])


def test_the_gates(full, periods, monkeypatch):
    train = cut(full, periods["train_end"])
    module = REGISTRY["trend_day"]
    numbers = futures_jobs.training_numbers(train, module, None, RULES)
    monkeypatch.setattr(futures_jobs, "MIN_DAYS_TRADED", 10_000)
    strict = futures_jobs.training_numbers(train, module, None, RULES)
    assert strict["gates"]["days_traded"] is False and strict["passes"] is False
    assert numbers["gates"]["profit_normal"] == (numbers["pnl_normal"] is not None and numbers["pnl_normal"] > 0)
    assert numbers["gates"]["profit_double"] == (numbers["pnl_double"] > 0)
    with pytest.raises(ValueError, match="Set the fee"):
        futures_jobs.training_numbers(train, module, None, topstep.Rules())


# ------------------------------------------------------------------ proposals


def test_proposals_are_half_random_half_changes_to_kept_models():
    kept = [{"id": "pullback-s1", "params": params_with_defaults(REGISTRY["pullback"], {"trend": 0.5})},
            {"id": "pullback-s2", "params": params_with_defaults(REGISTRY["pullback"], {"dip": 0.3})}]
    props = fs.proposals(77, 3, "pullback", 8, kept)
    random_ones = [p for p in props if p["parent"] is None]
    changes = [p for p in props if p["parent"]]
    assert len(random_ones) == 4 and len(changes) == 4
    assert [p["seed"] for p in random_ones] == [f"77:3:pullback:{i}" for i in range(4)]
    assert changes[0]["seed"] == "77:3:pullback:m0 parent=pullback-s1" and changes[1]["parent"] == "pullback-s2"
    assert fs.proposals(77, 3, "pullback", 8, kept) == props  # the seed text re-creates every one
    for p in changes:  # a small change: every number within 15% (or one step) of the parent's
        parent = next(k for k in kept if k["id"] == p["parent"])["params"]
        for name, spec in REGISTRY["pullback"].SEARCH_SPACE.items():
            if spec[-1] != "choice":
                step = max(abs(parent[name]) * 0.15, (spec[1] - spec[0]) * 0.02, 1 if spec[-1] == "int" else 0)
                assert abs(p["params"][name] - parent[name]) <= step + 1e-9
                assert spec[0] <= p["params"][name] <= spec[1]
    assert all(p["parent"] is None for p in fs.proposals(77, 3, "pullback", 8, []))


# ------------------------------------------------------------------ robustness


class FixedScorer:
    def __init__(self, scores):
        self.scores = scores
        self.seen = []

    def score(self, tasks, should_stop, progress=None):
        self.seen += tasks
        return [{"score": s} for s in self.scores[: len(tasks)]]


def test_a_sharp_peak_is_dropped():
    winner = {"name": "pullback", "params": params_with_defaults(REGISTRY["pullback"], None), "seed": "1:1:pullback:0",
              "parent": None, "score": 1.0}
    n = len(fs.neighbours(REGISTRY["pullback"], winner["params"]))
    assert n >= 8  # two neighbours per numeric setting
    ok, middle = fs.robust(FixedScorer([0.9] * n), winner, lambda: False)
    assert ok and middle == pytest.approx(0.9)
    ok, middle = fs.robust(FixedScorer([0.3] * (n - 2) + [2.0, 2.0]), winner, lambda: False)
    assert not ok and middle == pytest.approx(0.3)  # the median neighbour scores under half of 1.0
    scorer = FixedScorer([0.9] * n)
    fs.robust(scorer, winner, lambda: False)
    moved = [next(k for k in t["params"] if t["params"][k] != winner["params"][k]) for t in scorer.seen]
    assert sorted(set(moved)) == sorted(k for k, s in REGISTRY["pullback"].SEARCH_SPACE.items() if s[-1] != "choice")


# ------------------------------------------------------------------ the process pool


def test_the_pool_scores_like_one_process_and_stops_at_once(full, periods):
    train = cut(full, periods["train_end"])
    tasks = fs.proposals(5, 1, "gap_fade", 6, [])
    one = fs.Scorer(train, RULES, 1)
    many = fs.Scorer(train, RULES, 2)
    try:
        a = one.score(tasks, lambda: False)
        b = many.score(tasks, lambda: False)
        assert [r["score"] for r in a] == [r["score"] for r in b]
        assert [r["seed"] for r in b] == [t["seed"] for t in tasks]
        calls = {"n": 0}

        def stop_soon():
            calls["n"] += 1
            return calls["n"] > 2

        with pytest.raises(JobStopped):
            many.score(fs.proposals(5, 2, "gap_fade", 40, []), stop_soon)
        assert many.pool is None  # ended at once, not left running
    finally:
        one.close()
        many.close()


# ------------------------------------------------------------------ a whole search, with fake HTTP


class Coordinator:
    """The coordinator's search routes, recorded."""

    def __init__(self, kept=None, ideas=None):
        self.kept = kept or []
        self.ideas = list(ideas or [])
        self.found: list[dict] = []
        self.tries: list[dict] = []
        self.recipes: list[list[dict]] = []

    def get_json(self, url, token=None, timeout=None):
        assert url.endswith("/api/v1/models/kept?market=futures")
        return self.kept

    def post_json(self, url, body, token=None, timeout=None):
        if url.endswith("/api/v1/search/tries"):
            self.tries.append(body["tries"])
            self.recipes.append(body.get("recipes") or [])
            return {"ok": True}
        if url.endswith("/api/v1/search/ideas"):
            given, self.ideas = self.ideas[:body["take"]], self.ideas[body["take"]:]
            return given
        assert url.endswith("/api/v1/models/futures")
        self.found.append(body)
        return {"kept": True, "id": f"{body['module']}-s{len(self.found)}"}


def search(monkeypatch, cache, coordinator, periods, rounds=2, rules=RULES, robust_share=None, new_recipes=0, **extra):
    monkeypatch.setattr(fs.http, "get_json", coordinator.get_json)
    monkeypatch.setattr(fs.http, "post_json", coordinator.post_json)
    if robust_share is not None:
        monkeypatch.setattr(fs, "ROBUST_SHARE", robust_share)
    events = []

    def emit(cp, progress, detail):
        events.append((dict(cp), progress, detail))

    params = {"_context": {"host_url": "http://h", "worker_token": "t"}, "markets": ["futures"], "seed": 4,
              "candidates": 4, "files": ["trend_day", "vwap_revert"], "processes": 1, "rules": rules.to_dict(),
              "periods": periods, "_cache": cache, "_job_id": "job-1", "new_recipes": new_recipes, **extra}
    with pytest.raises(JobStopped):
        fs.run_futures_search(params, None, emit, lambda: bool(events) and max(e[0]["round"] for e in events) > rounds)
    return events


def test_a_search_round_end_to_end(monkeypatch, edge, periods):
    cache = FakeCache(edge, periods)
    coordinator = Coordinator()
    events = search(monkeypatch, cache, coordinator, periods, robust_share=-1e9)  # keep every winner, to see the posts
    assert cache.downloads <= 2 and cache.requested.count("train") == 3  # one download per period, then the ETag cache
    assert "lockbox" not in cache.requested
    assert [sum(t["n"] for t in r.values()) for r in coordinator.tries] == [8, 8]
    assert coordinator.found, "nothing passed the gates on the synthetic prices"
    for body in coordinator.found:
        m = body["metrics"]
        assert body["seed"].startswith("4:") and body["train_score"] > 0 and m["train"]["passes"]
        assert m["train"]["end"] <= futures_jobs.day_int(periods["train_end"])
        assert m["held_out"]["numbers"]["start"] > futures_jobs.day_int(periods["train_end"])
        assert m["held_out"]["numbers"]["end"] <= futures_jobs.day_int(periods["held_out_end"])
        assert len(m["train"]["daily"]["pnl"]) == m["train"]["n_days"] and m["held_out"]["sizes"]
    assert events[-1][0]["tried"] >= 16


def test_held_out_prices_are_read_only_after_choosing(monkeypatch, edge, periods):
    cache = FakeCache(edge, periods)
    order = []
    real = futures_jobs.held_out_numbers

    def spy(*a, **k):
        order.append("held_out")
        return real(*a, **k)

    real_score = fs.Scorer.score

    def scoring(self, tasks, should_stop, progress=None):
        order.append("choose")
        assert cache.requested.count("held_out") == order.count("held_out")  # never fetched for choosing
        return real_score(self, tasks, should_stop, progress)

    monkeypatch.setattr(futures_jobs, "held_out_numbers", spy)
    monkeypatch.setattr(fs.Scorer, "score", scoring)
    search(monkeypatch, cache, Coordinator(), periods, rounds=1, robust_share=-1e9)
    assert order and order[0] == "choose" and "lockbox" not in cache.requested


def test_choices_never_depend_on_held_out_or_lockbox_prices(monkeypatch, edge, periods):
    """Two searches with the same seed, one of them shown wildly different held-out and
    lockbox prices: they pick the same settings with the same training scores."""
    full = edge
    first = Coordinator()
    search(monkeypatch, FakeCache(full, periods), first, periods, rounds=1, robust_share=-1e9)
    rng = np.random.default_rng(1)
    start = int(full.day_start[full.day_index(futures_jobs.day_int(periods["train_end"]), side="right")])
    wild = {}
    for name in ("open", "high", "low", "close"):
        arr = getattr(full, name).copy()
        arr[:, start:] = np.round(rng.uniform(2000, 9000, arr[:, start:].shape) * 4) / 4
        wild[name] = arr
    wild["high"] = np.maximum.reduce([wild["high"], wild["open"], wild["close"]])
    wild["low"] = np.minimum.reduce([wild["low"], wild["open"], wild["close"]])
    scrambled = wfd.FuturesData(**{**full.__dict__, **wild})
    second = Coordinator()
    search(monkeypatch, FakeCache(scrambled, periods), second, periods, rounds=1, robust_share=-1e9)
    assert first.found and [(b["module"], b["params"], b["train_score"], b["seed"]) for b in first.found] == \
        [(b["module"], b["params"], b["train_score"], b["seed"]) for b in second.found]
    assert [b["metrics"]["held_out"]["numbers"] for b in first.found] != \
        [b["metrics"]["held_out"]["numbers"] for b in second.found]


def test_the_search_uses_kept_models_as_parents(monkeypatch, full, periods):
    kept = [{"id": "trend_day-s9", "module": "trend_day", "params": params_with_defaults(REGISTRY["trend_day"], None)}]
    coordinator = Coordinator(kept)
    seen = []
    real = fs.proposals

    def spy(seed, round_no, name, count, kept_for_file):
        out = real(seed, round_no, name, count, kept_for_file)
        seen.extend(out)
        return out

    monkeypatch.setattr(fs, "proposals", spy)
    search(monkeypatch, FakeCache(full, periods), coordinator, periods, rounds=1)
    seen = [p for p in seen if p["seed"].startswith("4:1:")]  # round 1
    pull = [p for p in seen if p["name"] == "trend_day"]
    assert sum(p["parent"] == "trend_day-s9" for p in pull) == 2 and sum(p["parent"] is None for p in pull) == 2
    assert all(p["parent"] is None for p in seen if p["name"] == "vwap_revert")


def test_every_round_tries_new_recipes_and_counts_them_together(monkeypatch, edge, periods):
    coordinator = Coordinator()
    seen = []
    real = fs.proposals

    def spy(*a, **k):
        out = real(*a, **k)
        seen.extend(out)
        return out

    monkeypatch.setattr(fs, "proposals", spy)
    search(monkeypatch, FakeCache(edge, periods), coordinator, periods, rounds=2, robust_share=-1e9, new_recipes=2)
    seen = [p for p in seen if p["seed"].startswith(("4:1:", "4:2:"))]  # rounds 1 and 2 (3 was stopped)
    made = sorted({p["name"] for p in seen if R.is_recipe(p["name"])})
    assert len(made) == 4  # two new recipes in each of the two rounds
    for p in seen:
        if R.is_recipe(p["name"]):
            assert R.name_of(p["params"]["recipe"]) == p["name"] and p["name"] in p["seed"]
    # Re-created from the seed: the round's recipes are the same random mixes every time.
    first = R.random_recipe(random.Random("4:1:recipe:0"))
    assert R.name_of(first) in made
    assert [r["recipe"]["n"] for r in coordinator.tries] == [8, 8]  # 2 recipes x 4 candidates, counted together
    assert all(set(r) == {"trend_day", "vwap_revert", "recipe"} for r in coordinator.tries)
    for body in coordinator.found:
        if R.is_recipe(body["module"]):
            assert body["params"]["recipe"] and R.name_of(body["params"]["recipe"]) == body["module"]


def test_a_round_tries_haiku_recipes_beside_random_ones_and_reports_training_numbers(monkeypatch, edge, periods):
    mixes = [R.random_recipe(random.Random(f"haiku idea {i}")) for i in range(2)]
    ideas = [{"id": 70 + i, "family": R.name_of(m), "recipe": m, "note": "an idea"} for i, m in enumerate(mixes)]
    coordinator = Coordinator(ideas=ideas)
    search(monkeypatch, FakeCache(edge, periods), coordinator, periods, rounds=1, robust_share=-1e9, new_recipes=1)
    reported = coordinator.recipes[0]
    by_name = {r["name"]: r for r in reported}
    for idea in ideas:
        entry = by_name[idea["family"]]
        assert entry["idea_id"] == idea["id"] and entry["tried"] == 4  # a full set of candidates each
        assert entry["recipe"] == idea["recipe"]
    randoms = [r for r in reported if r["idea_id"] is None]
    assert len(randoms) == 1  # the round's one random recipe, beside them
    allowed = {"name", "recipe", "idea_id", "tried", "best_score", "passes", "days_traded", "pnl_double", "pnl_normal"}
    assert all(set(r) == allowed for r in reported)  # training numbers only, never held-out ones
    assert coordinator.tries[0]["recipe"]["n"] == 12
    for body in coordinator.found:
        if body["module"] in by_name:
            assert body["idea_id"] == by_name[body["module"]]["idea_id"]


def test_kept_recipes_keep_being_tuned(monkeypatch, full, periods):
    mix = R.random_recipe(random.Random("a kept recipe"))
    module = R.family(mix)
    kept = [{"id": module.__name__ + "-s1", "module": module.__name__,
             "params": params_with_defaults(module, None)}]
    seen = []
    real = fs.proposals

    def spy(*a, **k):
        out = real(*a, **k)
        seen.extend(out)
        return out

    monkeypatch.setattr(fs, "proposals", spy)
    search(monkeypatch, FakeCache(full, periods), Coordinator(kept), periods, rounds=1)
    tuned = [p for p in seen if p["name"] == module.__name__ and p["seed"].startswith("4:1:")]
    assert len(tuned) == 2 and sum(p["parent"] == kept[0]["id"] for p in tuned) == 1  # per_file // 4, at least 2


def test_a_search_without_the_fees_does_not_start(monkeypatch, full, periods):
    with pytest.raises(RuntimeError, match="Set the fee in config/topstep.toml"):
        search(monkeypatch, FakeCache(full, periods), Coordinator(), periods, rules=topstep.Rules())


# ------------------------------------------------------------------ the backtest and Final check jobs


def test_the_futures_backtest_job_reads_training_and_held_out_only(monkeypatch, edge, periods):
    cache = FakeCache(edge, periods)
    monkeypatch.setattr(futures_jobs, "PriceCache", lambda ctx: cache)
    params = {"_context": {"host_url": "h", "worker_token": "t"}, "model_id": "gap_fade", "module": "gap_fade",
              "params": {}, "market": "futures", "rules": RULES.to_dict(), "periods": periods}
    result = futures_jobs.run_futures_backtest(params, None, lambda *a: None, lambda: False)
    assert cache.requested == ["train", "held_out"]
    assert result["summary"].startswith("Held-out: passes ") and result["train"]["daily"]
    assert result["held_out"]["sim_key"] == RULES.sim_key() and result["held_out"]["max_size"] >= 1


def test_a_recipe_model_runs_through_the_same_backtest_job(monkeypatch, edge, periods):
    cache = FakeCache(edge, periods)
    monkeypatch.setattr(futures_jobs, "PriceCache", lambda ctx: cache)
    module = R.family(R.random_recipe(random.Random("backtest me")))
    params = {"_context": {"host_url": "h", "worker_token": "t"}, "model_id": module.__name__ + "-s1",
              "module": module.__name__, "params": params_with_defaults(module, None), "market": "futures",
              "rules": RULES.to_dict(), "periods": periods}
    result = futures_jobs.run_futures_backtest(params, None, lambda *a: None, lambda: False)
    assert cache.requested == ["train", "held_out"] and result["params"]["recipe"] == module.RECIPE
    assert result["held_out"]["sim_key"] == RULES.sim_key()
    with pytest.raises(KeyError, match="hold no recipe"):  # a recipe name without its recipe never runs
        futures_jobs.run_futures_backtest({**params, "params": {}}, None, lambda *a: None, lambda: False)


def test_the_final_check_job_opens_the_lockbox_once_and_sends_the_result(monkeypatch, full, periods):
    cache = FakeCache(full, periods)
    monkeypatch.setattr(futures_jobs, "PriceCache", lambda ctx: cache)
    sent = []
    monkeypatch.setattr(futures_jobs.http, "post_json", lambda url, body, token=None, timeout=None: sent.append((url, body)))
    params = {"_context": {"host_url": "http://h", "worker_token": "t"}, "_job_id": "j9", "model_id": "gap_fade",
              "module": "gap_fade", "params": {}, "rules": RULES.to_dict(), "periods": periods, "contracts": 2}
    out = futures_jobs.run_final_check(params, None, lambda *a: None, lambda: False)
    assert cache.requested == ["lockbox"]
    (url, body), = sent
    assert url == "http://h/api/v1/models/gap_fade/final-check" and body["job_id"] == "j9"
    assert body["result"]["contracts"] == 2 and len(body["result"]["sizes"]) == 2
    assert body["result"]["numbers"]["start"] > futures_jobs.day_int(periods["held_out_end"])
    assert out["summary"].startswith("Final check: lockbox passes ")


# ------------------------------------------------------------------ chance this is luck


def test_chance_of_luck_grows_with_the_number_of_tries():
    rng = np.random.default_rng(2)
    shape = futures_stats.shape(rng.normal(30, 300, 1000))  # a modest edge: Sharpe about 0.1 a day
    one = futures_stats.chance_of_luck(shape["sr_day"], shape["n_days"], shape["skew"], shape["kurt"], 1, 0.0)
    many = futures_stats.chance_of_luck(shape["sr_day"], shape["n_days"], shape["skew"], shape["kurt"], 5000, 0.003)
    assert one < 0.05 < many
    assert futures_stats.expected_best(1, 0.01) == 0.0
    assert futures_stats.expected_best(100, 0.01) < futures_stats.expected_best(10_000, 0.01)
    assert futures_stats.variance_from_sums(3, 6.0, 14.0) == pytest.approx(1.0)  # 1, 2, 3
    assert futures_stats.chance_of_luck(0.1, 2, 0, 3, 10, 0.01) is None


def test_a_worker_searches_only_its_share(monkeypatch, edge, periods):
    """One worker's share is recipes only, another's some files only: neither strays."""
    coordinator = Coordinator()
    search(monkeypatch, FakeCache(edge, periods), coordinator, periods, rounds=1, new_recipes=2,
           files=[], recipes=True)
    assert all(set(r) == {"recipe"} for r in coordinator.tries)
    coordinator = Coordinator()
    search(monkeypatch, FakeCache(edge, periods), coordinator, periods, rounds=1, new_recipes=2,
           files=["trend_day"], recipes=False)
    assert all(set(r) == {"trend_day"} for r in coordinator.tries)
