"""Model search: v1's seeded random search, choosing on the training period only,
keeping a few good models per file, start and stop from the dashboard."""
from __future__ import annotations

import pytest

from coordinator import search
from fleet2.sim.control import JobStopped
from fleet2.worker import search_job
from tests.conftest import enroll, heartbeat
from tests.test_lookahead import synthetic


def test_candidates_are_repeatable_and_inside_the_search_space():
    a = search_job.candidate(42, 1, "momentum", 3)
    assert a == search_job.candidate(42, 1, "momentum", 3)
    assert a != search_job.candidate(42, 1, "momentum", 4)
    assert a != search_job.candidate(43, 1, "momentum", 3)
    from fleet2.models import momentum
    for key, (low, high, _) in momentum.SEARCH_SPACE.items():
        assert low <= a[key] <= high


def test_score_shrinks_results_from_few_trades():
    assert search_job.score({"roi": 0.2, "trades": 100}) == pytest.approx(0.1)
    assert search_job.score({"roi": 0.2, "trades": 10}) < search_job.score({"roi": 0.1, "trades": 1000})


def test_a_search_round_chooses_on_training_only_and_reports_held_out(monkeypatch):
    data = {"stocks": synthetic("stocks", 900, seed=5), "crypto": synthetic("crypto", 2400, seed=6)}
    monkeypatch.setattr(search_job, "load", lambda ctx, market: data[market])
    seen_periods = []
    real_evaluate = search_job.evaluate

    def spy(d, name, params, limits, split, frac, stop, period):
        seen_periods.append(period)
        return real_evaluate(d, name, params, limits, split, frac, stop, period)

    monkeypatch.setattr(search_job, "evaluate", spy)
    posted = []
    monkeypatch.setattr(search_job.http, "post_json", lambda url, body, token=None, timeout=None: posted.append(body) or {"kept": True})
    rounds = []

    def emit(cp, progress, detail):
        rounds.append(cp["round"])

    with pytest.raises(JobStopped):
        search_job.run_search({"_context": {"host_url": "h", "worker_token": "t"}, "markets": ["stocks", "crypto"],
                               "seed": 7, "candidates": 3},
                              None, emit, lambda: len(rounds) > 0 and max(rounds) > 1)
    # every held-out run comes after its file's training runs, one per kept model
    assert seen_periods.count("held_out") == len(posted)
    for body in posted:
        train, held = body["metrics"]["train"], body["metrics"]["held_out"]
        assert train["trades"] >= 100 and body["train_score"] > 0
        assert held["start"] >= body["metrics"]["split_t"] > train["end"]


def test_the_coordinator_keeps_at_most_five_per_file_and_only_better_ones(client, conn):
    def found(score):
        return {"module": "momentum", "params": {"lookback": 60}, "train_score": score, "seed": "1:1",
                "metrics": {"train": {"roi": score}, "held_out": {"roi": 0.0, "trades": 120, "enough_trades": True}}}

    for s in (0.01, 0.02, 0.03, 0.04, 0.05):
        assert search.store_found(conn, None, found(s))["kept"]
    assert not search.store_found(conn, None, found(0.005))["kept"]
    reply = search.store_found(conn, None, found(0.06))
    assert reply["kept"] and reply["id"] == "momentum-s6"
    statuses = {r["id"]: r["status"] for r in conn.execute("SELECT id, status FROM models WHERE origin = 'search'")}
    assert list(statuses.values()).count("retired") == 1 and statuses["momentum-s1"] == "retired"
    row = conn.execute("SELECT name, description FROM models WHERE id = 'momentum-s6'").fetchone()
    assert row["name"] == "Momentum #6"


def test_start_and_stop_from_the_dashboard(client, conn):
    for name in ("w1", "w2"):
        heartbeat(client, enroll(client, conn, name))
    resp = client.post("/api/search/start", json={})
    assert resp.status_code == 201 and resp.json()["message"] == "Model search started on 2 workers"
    seeds = {j["params"]["seed"] for j in resp.json()["jobs"]}
    assert len(seeds) == 2
    assert client.post("/api/search/start", json={}).status_code == 409
    status = search.search_status(conn)
    assert status["running"] and status["button"] == "Stop model search"
    assert client.post("/api/search/stop").json() == {"message": "Model search stopped"}
    assert not search.search_status(conn)["running"]


def test_paper_job_decides_at_once_then_on_its_cadence():
    from fleet2.models import momentum
    from fleet2.worker.paper_job import decision

    params = dict(momentum.DEFAULT_PARAMS)
    data = synthetic("stocks", 401)  # 401 is not a multiple of rebalance_every (5)
    assert decision(data, momentum, params) is None
    n, weights = decision(data, momentum, params, first=True)
    assert n == 401 and isinstance(weights, dict)
    assert decision(data.slice(0, 400), momentum, params)[0] == 400
