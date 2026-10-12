"""Both searches learn from the search log: they read each family's log every round,
propose with the sampler once it has enough scores, log every candidate with why it was
thrown away, skip families that gave up, choose futures winners by their training
Combine numbers above the fee floor, check stock winners for robustness, and carry on
when the log is out of reach."""
from __future__ import annotations

import random
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest

from coordinator import search
from fleet2.common import http
from fleet2.models import REGISTRY as STOCK_FILES
from fleet2.models.futures import REGISTRY as FUTURES_FILES
from fleet2.models.futures import recipe as R
from fleet2.models.futures.base import draw_params as futures_draw
from fleet2.models.futures.base import params_with_defaults
from fleet2.sim import robust, sampler
from fleet2.sim.control import JobStopped
from fleet2.worker import futures_jobs, search_job
from fleet2.worker import futures_search_job as fs
from tests.conftest import enroll
from tests.test_futures_search import RULES
from tests.test_lookahead import synthetic

FLOOR = futures_jobs.combine_floor(RULES)
NOT_REVERSAL = sorted(n for n in STOCK_FILES if n != "reversal")


def history(name: str, n: int = 30, good: float = 0.2) -> list[dict]:
    """n scored trials of a futures file: the higher first_minutes, the better."""
    rng = random.Random(f"history {name}")
    space = FUTURES_FILES[name].SEARCH_SPACE
    out = []
    for i in range(n):
        p = {k: v for k, v in futures_draw(FUTURES_FILES[name], rng).items() if k in space}
        out.append({"params": p, "score": FLOOR + p["first_minutes"] * good, "rejected": None})
    return out


class Coordinator:
    """The coordinator's search routes for both searches, recorded. `log` holds what a
    GET of the search log answers; POSTs add to it."""

    def __init__(self, log=None, kept=None, ideas=None, log_down=False):
        self.log: dict[tuple[str, str], list[dict]] = {k: list(v) for k, v in (log or {}).items()}
        self.kept = kept or []
        self.ideas = list(ideas or [])
        self.log_down = log_down
        self.reads: list[tuple[str, str]] = []
        self.posts: list[dict] = []  # bodies POSTed to the search log
        self.found: list[dict] = []
        self.tries: list[dict] = []
        self.recipes: list[list[dict]] = []
        self.log_calls = 0
        self.attempts: list[str] = []  # families whose log was asked for, answered or not

    def get_json(self, url, token=None, timeout=None):
        if "/api/v1/search/log?" in url:
            self.log_calls += 1
            q = parse_qs(urlsplit(url).query)
            self.attempts.append(q["family"][0])
            if self.log_down:
                raise http.HttpError(502, "bad gateway", url)
            key = (q["market"][0], q["family"][0])
            self.reads.append(key)
            trials = [{k: t[k] for k in ("params", "score", "rejected")} for t in self.log.get(key, [])]
            return {"market": key[0], "family": key[1], "tried": len(trials), "gave_up": False, "trials": trials}
        assert url.endswith("/api/v1/models/kept?market=futures")
        return self.kept

    def post_json(self, url, body, token=None, timeout=None):
        if url.endswith("/api/v1/search/log"):
            self.log_calls += 1
            if self.log_down:
                raise http.HttpConnectionError("refused")
            assert set(body) == {"job_id", "market", "family", "trials"} and 0 < len(body["trials"]) <= 1000
            for t in body["trials"]:
                assert set(t) == {"params", "score", "rejected", "metrics"}
            self.posts.append(body)
            self.log.setdefault((body["market"], body["family"]), []).extend(body["trials"])
            return {"stored": len(body["trials"])}
        if url.endswith("/api/v1/search/tries"):
            self.tries.append(body["tries"])
            self.recipes.append(body.get("recipes") or [])
            return {"ok": True}
        if url.endswith("/api/v1/search/ideas"):
            given, self.ideas = self.ideas[:body["take"]], self.ideas[body["take"]:]
            return given
        assert url.endswith(("/api/v1/models", "/api/v1/models/futures"))
        self.found.append(body)
        return {"kept": True}

    def posted(self, market: str, family: str) -> list[dict]:
        return [t for b in self.posts if (b["market"], b["family"]) == (market, family) for t in b["trials"]]


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(http, "_sleep", lambda s: None)


# ------------------------------------------------------------------ the sampler takes over from random settings


def test_after_20_scored_trials_the_sampler_proposes_near_the_good_ones():
    module = FUTURES_FILES["trend_day"]
    few, many = history("trend_day", 10), history("trend_day", 40)
    plain = fs.proposals(5, 1, "trend_day", 12, [], trials=[])
    assert fs.proposals(5, 1, "trend_day", 12, [], trials=few) == plain  # under 20 scores: plain random settings
    learned = fs.proposals(5, 1, "trend_day", 12, [], trials=many)
    assert learned == fs.proposals(5, 1, "trend_day", 12, [], trials=many)  # the same log gives the same proposals
    assert [p["params"] for p in learned] != [p["params"] for p in plain]
    assert [p["params"] for p in learned] != [futures_draw(module, random.Random(f"5:1:trend_day:{i}"))
                                               for i in range(12)]
    mean = np.mean([p["params"]["first_minutes"] for p in learned])
    assert mean > np.mean([p["params"]["first_minutes"] for p in plain]) and mean > 90  # towards the good region
    tried = {tuple(sorted(t["params"].items())) for t in many}
    assert not any(tuple(sorted((k, p["params"][k]) for k in module.SEARCH_SPACE)) in tried for p in learned)
    # Stocks: the same sampler, seeded by seed, round and file.
    space = STOCK_FILES["momentum"].SEARCH_SPACE
    rng = random.Random("stock history")
    stock_log = []
    for _ in range(30):
        p = {k: (rng.randint(lo, hi) if kind == "int" else round(rng.uniform(lo, hi), 3))
             for k, (lo, hi, kind) in space.items()}
        stock_log.append({"params": p, "score": float(p[sorted(space)[0]]), "rejected": None})
    learned = search_job.candidates(7, 3, "momentum", 6, stock_log)
    expected = sampler.suggest(space, stock_log, 6, random.Random("7:3:momentum"))
    assert [{k: c[k] for k in space} for c in learned] == expected
    assert learned != search_job.candidates(7, 3, "momentum", 6, [])


# ------------------------------------------------------------------ futures: the whole round, with fake numbers


def fake_numbers(value):
    """futures_jobs.training_numbers stand-in: passes every gate, combine_score = value(params)."""

    def numbers(train, module, params, rules, keep_daily=False, should_stop=None):
        params = params_with_defaults(module, params)
        score = value(params)
        combine = None if score is None else {
            "measure": "net_per_attempt", "floor": FLOOR, "best_size": 2, "pass_rate": 0.3, "fail_rate": 0.2,
            "timeout_rate": 0.5, "net_per_attempt": score, "score": score,
            "sizes": [{"contracts": k, "score": score} for k in (1, 2)]}
        out = {"score": 0.4, "sharpe": 0.5, "sr_day": 0.03, "days_traded": 150, "pnl_double": 900.0,
               "pnl_normal": 1500.0, "worst_stretch": 400.0, "combine_score": score, "combine": combine,
               "gates": {"days_traded": True, "profit_double": True, "within_the_day": True, "profit_normal": True},
               "passes": True}
        if keep_daily:
            out["daily"] = {"days": [20200102], "pnl": [1.0]}
        return out

    return numbers


class Cache:
    def __init__(self):
        self.requested = []

    def get(self, through, job_id=None):
        self.requested.append(through)
        return SimpleNamespace(days=np.array([20200102]), feed="synthetic")


PERIODS = {"train_end": "2020-12-31", "held_out_end": "2021-12-31", "lockbox_end": "2022-12-31"}


def futures_search(monkeypatch, coordinator, value, rounds=1, robust_share=None, numbers=None, **extra):
    monkeypatch.setattr(futures_jobs, "training_numbers", numbers or fake_numbers(value))
    monkeypatch.setattr(futures_jobs, "held_out_numbers", lambda *a, **k: {"numbers": {"trades": 9}, "sizes": []})
    monkeypatch.setattr(fs.http, "get_json", coordinator.get_json)
    monkeypatch.setattr(fs.http, "post_json", coordinator.post_json)
    if robust_share is not None:
        monkeypatch.setattr(fs, "ROBUST_SHARE", robust_share)
    events = []
    params = {"_context": {"host_url": "http://h", "worker_token": "t"}, "markets": ["futures"], "seed": 4,
              "candidates": 6, "files": ["trend_day"], "processes": 1, "rules": RULES.to_dict(), "periods": PERIODS,
              "_cache": Cache(), "_job_id": "job-1", "new_recipes": 0, "ideas": 0, **extra}
    with pytest.raises(JobStopped):
        fs.run_futures_search(params, None, lambda cp, p, d: events.append((cp, d)),
                              lambda: bool(events) and max(e[0]["round"] for e in events) > rounds)
    return events


def by_first_minutes(params):
    return FLOOR + 1.0 + float(params.get("first_minutes", 0))  # above the floor; the higher the better


def test_futures_reads_the_log_logs_every_candidate_and_picks_the_best_combine_score(monkeypatch):
    served = history("trend_day", 25)
    coordinator = Coordinator({("futures", "trend_day"): served})
    events = futures_search(monkeypatch, coordinator, by_first_minutes, rounds=2)
    assert coordinator.reads[:2] == [("futures", "trend_day")] * 2  # once per family per round
    first = coordinator.posts[0]
    assert first["family"] == "trend_day" and first["job_id"] == "job-1" and len(first["trials"]) == 6
    # Round 1's proposals are exactly the sampler's, given the log as it was.
    expected = fs.proposals(4, 1, "trend_day", 6, [], trials=served)
    assert [t["params"] for t in first["trials"]] == [p["params"] for p in expected]
    for t in first["trials"]:
        assert t["score"] == by_first_minutes(t["params"])
        assert set(t["metrics"]) == {"score", "sharpe", "days_traded", "pnl_double", "pnl_normal", "combine"}
        assert "sizes" not in t["metrics"]["combine"] and t["metrics"]["combine"]["best_size"] == 2
    assert all(t["rejected"] is None for t in first["trials"])  # all beat the floor; the winner held up
    # Round 2 read round 1's trials too (the log grows), and is reproducible from that log.
    assert len(coordinator.posted("futures", "trend_day")) == 12
    winner = max(first["trials"], key=lambda t: t["score"])
    found = coordinator.found[0]
    assert {k: found["params"][k] for k in winner["params"]} == winner["params"]
    assert found["train_score"] == 0.4 and found["seed"].startswith("4:1:trend_day:s")
    assert found["metrics"]["train"]["neighbour_combine_median"] is not None
    assert any("checking Trend day settings" in d for _, d in events)


def test_futures_has_no_winner_when_everything_sits_at_the_floor(monkeypatch):
    coordinator = Coordinator()
    calls = []
    monkeypatch.setattr(fs, "robust", lambda *a: calls.append(a) or (True, None, None))
    futures_search(monkeypatch, coordinator, lambda p: FLOOR)
    assert not coordinator.found and not calls  # nothing beats the floor: no robustness check, no held-out test
    trials = coordinator.posted("futures", "trend_day")
    assert len(trials) == 6 and all(t["rejected"] == "floor" and t["score"] == FLOOR for t in trials)


def test_a_winner_that_is_not_robust_is_logged_so(monkeypatch):
    coordinator = Coordinator()
    futures_search(monkeypatch, coordinator, by_first_minutes, robust_share=1e9)
    trials = coordinator.posted("futures", "trend_day")
    assert not coordinator.found
    assert sorted(t["rejected"] or "" for t in trials) == [""] * 5 + ["not_robust"]
    dropped = next(t for t in trials if t["rejected"] == "not_robust")
    assert dropped["score"] == max(t["score"] for t in trials)


def test_futures_gate_failures_name_the_gate(monkeypatch):
    def numbers(train, module, params, rules, keep_daily=False, should_stop=None):
        return {"score": -0.1, "sr_day": 0.0, "days_traded": 3, "pnl_double": -5.0, "pnl_normal": None,
                "combine_score": None, "combine": None, "passes": False,
                "gates": {"days_traded": False, "profit_double": False, "within_the_day": True,
                          "profit_normal": False}}

    coordinator = Coordinator()
    futures_search(monkeypatch, coordinator, None, numbers=numbers)
    trials = coordinator.posted("futures", "trend_day")
    assert len(trials) == 6 and all(t["rejected"] == "gates:days_traded" and t["score"] is None for t in trials)
    assert not coordinator.found


def test_skip_recipes_and_a_share_of_recipes_only(monkeypatch):
    kept_mix, other_mix, idea_mix, random_mix = (R.random_recipe(random.Random(f"mix {i}")) for i in range(4))
    names = [R.name_of(m) for m in (kept_mix, other_mix, idea_mix, random_mix)]
    assert len(set(names)) == 4
    kept = [{"id": names[0] + "-s1", "module": names[0], "params": params_with_defaults(R.family(kept_mix), None)},
            {"id": names[1] + "-s1", "module": names[1], "params": params_with_defaults(R.family(other_mix), None)}]
    ideas = [{"id": 5, "family": names[2], "recipe": idea_mix, "note": "x"}]
    served = {("futures", names[1]): [{"params": {"bar_minutes": 5}, "score": FLOOR, "rejected": "floor"}]}
    coordinator = Coordinator(served, kept=kept, ideas=ideas)
    monkeypatch.setattr(R, "random_recipe", lambda rng: random_mix)
    futures_search(monkeypatch, coordinator, by_first_minutes, files=[], recipes=True, new_recipes=1, ideas=3,
                   skip_recipes=[names[0], names[2], names[3]])
    # Only the kept recipe that did not give up is tuned, with its log read; no model file is searched.
    assert set(coordinator.reads) == {("futures", names[1])}
    families = {b["family"] for b in coordinator.posts}
    assert families == {names[1]}
    assert all(set(t) == {"recipe"} for t in coordinator.tries)
    for t in coordinator.posted("futures", names[1]):
        assert "recipe" not in t["params"]  # the family names the recipe; its settings hold only numbers
    assert [r["name"] for r in coordinator.recipes[0]] == [names[1]]
    entry = coordinator.recipes[0][0]
    assert entry["combine_score"] is not None and entry["best_size"] == 2 and entry["timeout_rate"] == 0.5
    assert entry["pass_rate"] == 0.3 and entry["combine_measure"] == "net_per_attempt"


def test_an_empty_share_of_files_with_recipes_searches_recipes(monkeypatch):
    coordinator = Coordinator()
    futures_search(monkeypatch, coordinator, by_first_minutes, files=[], recipes=True, new_recipes=2)
    assert coordinator.reads == []  # new recipes have no history to read
    assert coordinator.posts and all(R.is_recipe(b["family"]) for b in coordinator.posts)
    assert len({b["family"] for b in coordinator.posts}) == 2
    assert coordinator.found and all(R.is_recipe(b["module"]) for b in coordinator.found)


def test_futures_search_carries_on_when_the_log_is_down(monkeypatch):
    coordinator = Coordinator(log_down=True)
    futures_search(monkeypatch, coordinator, by_first_minutes, rounds=2, files=["trend_day", "gap_fade"])
    assert len(coordinator.found) == 4  # two families, two rounds: found as usual
    # Each round tries the log once (with its short retries), then leaves it alone.
    assert set(coordinator.attempts) == {"trend_day"} and not coordinator.posts
    assert coordinator.attempts.count("trend_day") <= 3 * (len(search_job.LOG_PAUSES) + 1)


# ------------------------------------------------------------------ stocks


def stock_search(monkeypatch, coordinator, rounds=1, skip=NOT_REVERSAL, stop=None, **extra):
    data = {"stocks": synthetic("stocks", 900, seed=5), "crypto": synthetic("crypto", 2400, seed=6)}
    monkeypatch.setattr(search_job, "load", lambda ctx, market: data[market])
    monkeypatch.setattr(search_job.http, "get_json", coordinator.get_json)
    monkeypatch.setattr(search_job.http, "post_json", coordinator.post_json)
    monkeypatch.setattr(search_job, "COIN_FLIPS", 3)
    events = []
    params = {"_context": {"host_url": "http://h", "worker_token": "t"}, "markets": ["stocks"], "seed": 7,
              "candidates": 4, "skip_files": skip, **extra}
    with pytest.raises(JobStopped):
        search_job.run_search(params, None, lambda cp, p, d: events.append((cp, d)),
                              stop or (lambda: bool(events) and max(e[0]["round"] for e in events) > rounds))
    return events, data


def test_stocks_read_and_write_the_log_and_check_the_winner(monkeypatch):
    coordinator = Coordinator()
    events, _ = stock_search(monkeypatch, coordinator)
    assert set(coordinator.attempts) == {"reversal"}  # skip_files: the rest gave up
    trials = coordinator.posted("stocks", "reversal")
    assert len(trials) == 4 and len(coordinator.posts) == 1
    for t in trials:
        assert "curve" not in t["metrics"] and t["metrics"]["trades"] is not None
        if t["rejected"] in (None, "not_positive") or str(t["rejected"]).startswith("robustness:"):
            assert t["score"] == pytest.approx(robust.score(t["metrics"]))
    (found,) = coordinator.found
    report = found["metrics"]["robustness"]
    assert set(report) >= {"score", "folds", "neighbours", "cost_stress", "coin_flip", "flags", "passed"}
    assert report["coin_flip"]["draws"] == 3 and found["train_score"] == report["score"]
    assert found["seed"].startswith("7:1:reversal:")
    i = int(found["seed"].rsplit(":", 1)[1])
    winner = trials[i]
    assert winner["params"] == found["params"] and found["metrics"]["search_score"] == winner["score"]
    if report["passed"]:
        assert winner["rejected"] is None
    else:
        assert winner["rejected"] == "robustness:" + (",".join(report["flags"]) or "not_positive")
    title = STOCK_FILES["reversal"].NAME
    assert any(d == f"Round 1: checking {title} against coin flips (0%)" for _, d in events)
    # The report covers the training bars only (it never sees where held-out starts).
    folds = report["folds"]
    assert folds and max(f["end"] for f in folds) < found["metrics"]["split_t"]


def test_a_failed_robustness_report_is_still_sent_and_logged(monkeypatch):
    monkeypatch.setattr(search_job.robust, "report", lambda *a, **k: {
        "score": 0.01, "flags": ["fragile", "cost_sensitive"], "passed": False})
    coordinator = Coordinator()
    stock_search(monkeypatch, coordinator)
    (found,) = coordinator.found
    assert found["metrics"]["robustness"]["passed"] is False and found["train_score"] == 0.01
    rejected = [t["rejected"] for t in coordinator.posted("stocks", "reversal")]
    assert "robustness:fragile,cost_sensitive" in rejected


def test_stop_interrupts_a_long_robustness_check(monkeypatch):
    coordinator = Coordinator()
    started = {"n": 0}
    real = robust.report

    def report(*a, **k):
        started["n"] += 1
        out = real(*a, **k)
        started["finished"] = True
        return out

    monkeypatch.setattr(search_job.robust, "report", report)
    calls = {"n": 0}

    def stop():
        if started["n"]:
            calls["n"] += 1
        return calls["n"] > 5  # a few checks into the report

    stock_search(monkeypatch, coordinator, stop=stop)
    assert started["n"] == 1 and "finished" not in started and not coordinator.found


def test_skip_files_every_file_gave_up(monkeypatch):
    with pytest.raises(RuntimeError, match="gave up"):
        stock_search(monkeypatch, Coordinator(), skip=sorted(STOCK_FILES))


def test_stock_search_carries_on_when_the_log_is_down(monkeypatch):
    coordinator = Coordinator(log_down=True)
    stock_search(monkeypatch, coordinator)
    assert len(coordinator.found) == 1 and coordinator.posts == []


def test_log_batches_stay_small():
    trials = [{"params": {"i": i}, "score": 1.0, "rejected": None, "metrics": {"x": "y" * 100}} for i in range(450)]
    parts = search_job.batches(trials)
    assert [len(p) for p in parts] == [200, 200, 50] and [t for p in parts for t in p] == trials
    parts = search_job.batches(trials, max_bytes=2_000)
    assert all(len(p) <= 16 for p in parts) and [t for p in parts for t in p] == trials


def test_the_real_coordinator_takes_what_the_stock_search_sends(monkeypatch, client, conn):
    worker = enroll(client, conn, "w1")
    headers = {"Authorization": "Bearer " + worker["worker_token"]}

    def get_json(url, token=None, timeout=None):
        r = client.get(url.replace("http://h", ""), headers=headers)
        assert r.status_code == 200, r.text
        return r.json()

    def post_json(url, body, token=None, timeout=None):
        r = client.post(url.replace("http://h", ""), json=body, headers=headers)
        assert r.status_code == 200, r.text
        return r.json()

    data = {"stocks": synthetic("stocks", 900, seed=5)}
    monkeypatch.setattr(search_job, "load", lambda ctx, market: data[market])
    monkeypatch.setattr(search_job.http, "get_json", get_json)
    monkeypatch.setattr(search_job.http, "post_json", post_json)
    monkeypatch.setattr(search_job.robust, "report", lambda *a, **k: {"score": 0.02, "flags": [], "passed": True})
    events = []
    with pytest.raises(JobStopped):
        search_job.run_search({"_context": {"host_url": "http://h", "worker_token": "t"}, "markets": ["stocks"],
                               "seed": 7, "candidates": 4, "skip_files": NOT_REVERSAL},
                              None, lambda cp, p, d: events.append(cp), lambda: bool(events) and events[-1]["round"] > 2)
    log = get_json("http://h/api/v1/search/log?market=stocks&family=reversal")
    assert log["tried"] == 8 and len(log["trials"]) == 8
    row = conn.execute("SELECT metrics FROM models WHERE module = 'reversal' AND origin = 'search' LIMIT 1").fetchone()
    assert row["metrics"]["robustness"]["passed"] is True


# ------------------------------------------------------------------ the coordinator keeps models that passed


def found(score, passed=None):
    metrics = {"train": {"roi": score}, "held_out": {"roi": 0.0, "trades": 120}}
    if passed is not None:
        metrics["robustness"] = {"score": score, "flags": [] if passed else ["fragile"], "passed": passed}
    return {"module": "momentum", "params": {"lookback": 60}, "train_score": score, "seed": "1:1", "metrics": metrics}


def active(conn):
    return conn.execute("SELECT id, (metrics->>'train_score')::float AS score, metrics->'robustness'->>'passed' AS ok "
                        "FROM models WHERE origin = 'search' AND status = 'backtested' ORDER BY id").fetchall()


def test_a_model_that_failed_robustness_never_pushes_out_one_that_passed(conn):
    for s in (0.01, 0.02, 0.03, 0.04):
        assert search.store_found(conn, None, found(s, passed=True))["kept"]
    assert search.store_found(conn, None, found(0.5, passed=False))["kept"]  # room left: kept, ranked last
    reply = search.store_found(conn, None, found(0.9, passed=False))  # full: higher score, but failed
    assert reply["kept"]  # it replaces the weaker failed one ...
    assert sorted(r["score"] for r in active(conn) if r["ok"] == "false") == [0.9]
    reply = search.store_found(conn, None, found(0.005, passed=True))  # ... and a passed one beats any failed one
    assert reply["kept"] and all(r["ok"] == "true" for r in active(conn))
    reply = search.store_found(conn, None, found(5.0, passed=False))
    assert not reply["kept"] and "robustness" in reply["reason"]  # never pushes out a passed model
    assert len(active(conn)) == search.KEEP_PER_FILE
    # Among passed ones (and models from before the checks, which count as passed) the usual order holds.
    assert not search.store_found(conn, None, found(0.001, passed=True))["kept"]
    assert search.store_found(conn, None, found(0.06))["kept"]
    assert min(r["score"] for r in active(conn)) == 0.01


def test_kept_futures_models_compare_by_training_combine_score_first():
    from coordinator.futures_models import keep_rank

    def m(combine, floor=-296.0):
        return {"train": {"combine_score": combine, "combine": {"floor": floor}}}

    old = keep_rank({"train": {}}, 2.5)            # found before the Combine score: high Sharpe only
    floor_only = keep_rank(m(-296.0), 3.0)         # never passes in training
    passes = keep_rank(m(150.0), 0.1)
    better = keep_rank(m(900.0), -0.2)
    assert old < passes and floor_only < passes < better
    assert keep_rank(m(150.0), 0.5) > passes       # Sharpe breaks a tie
