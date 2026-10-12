"""The search log shared by both searches: storing and reading it, giving up on families
that keep failing, skipping them in the jobs' parameters, pruning and resetting it, the
fee-floor note on the futures screen and the stock robustness flags."""
from __future__ import annotations

import re
from dataclasses import replace

import pytest
from psycopg.types.json import Jsonb

from coordinator import cli, futures_data, futures_models, models_view, search
from coordinator.config import Config
from fleet2.models import REGISTRY as STOCK_FILES
from fleet2.models.futures import REGISTRY as FUTURES_FILES
from tests.conftest import enroll, heartbeat, seed_history
from tests.test_dashboard import attr_tags, element, text
from tests.test_futures_screen import FEES, fees, load, shaped, store  # noqa: F401 (fees is a fixture)
from tests.test_models_screen import metrics_with


def auth(worker: dict) -> dict:
    return {"Authorization": "Bearer " + worker["worker_token"]}


def post(client, worker, market="futures", family="gap_fade", trials=None, **extra):
    body = {"job_id": None, "market": market, "family": family, "trials": trials if trials is not None else []}
    return client.post("/api/v1/search/log", json={**body, **extra}, headers=auth(worker))


def fill(conn, market: str, family: str, n: int, score: float | None = None, rejected: str | None = "too few trades"):
    """n logged trials of one family, straight into the table."""
    conn.execute("INSERT INTO search_log (market, family, params, score, rejected) "
                 "SELECT %s, %s, jsonb_build_object('i', g), %s, %s FROM generate_series(1, %s) g",
                 (market, family, score, rejected, n))


def search_model(conn, family: str, n: int, metrics: dict) -> None:
    """A futures model found by search, with its held-out results."""
    conn.execute(
        "INSERT INTO models (id, name, module, market, description, how_it_works, params, status, origin, metrics, backtested_at) "
        "VALUES (%s, %s, %s, 'futures', 'd', 'h', '{}', 'backtested', 'search', %s, now())",
        (f"{family}-s{n}", f"{family} #{n}", family, Jsonb(metrics)))


def held_out_models(conn, family: str, how_many: int, paid: float = 0.0) -> None:
    for i in range(how_many):
        search_model(conn, family, i + 1, shaped(0.0, 0.0, paid, months=3.0))


# ------------------------------------------------------------------ POST and GET


def test_trials_round_trip_with_the_exact_shapes(client, conn):
    w = enroll(client, conn, "w1")
    job = conn.execute("SELECT gen_random_uuid() AS u").fetchone()["u"]
    trials = [{"params": {"lookback": 20}, "score": 1.5, "rejected": None, "metrics": {"trades": 120, "sharpe": 0.8}},
              {"params": {"lookback": 5}, "score": None, "rejected": "too few trades", "metrics": None}]
    r = post(client, w, "stocks", "momentum", trials, job_id=str(job))
    assert r.status_code == 200 and r.json() == {"stored": 2}
    row = conn.execute("SELECT * FROM search_log WHERE score IS NOT NULL").fetchone()
    assert (row["market"], row["family"], str(row["job_id"]), row["worker_id"]) == ("stocks", "momentum", str(job), w["worker_id"])
    r = client.get("/api/v1/search/log", params={"market": "stocks", "family": "momentum"}, headers=auth(w))
    assert r.status_code == 200
    assert r.json() == {"market": "stocks", "family": "momentum", "tried": 2, "scored": 1, "best_score": 1.5,
                        "gave_up": False,
                        "trials": [{"params": {"lookback": 20}, "score": 1.5, "rejected": None},
                                   {"params": {"lookback": 5}, "score": None, "rejected": "too few trades"}]}
    empty = client.get("/api/v1/search/log", params={"market": "crypto", "family": "pairs"}, headers=auth(w)).json()
    assert empty["tried"] == 0 and empty["best_score"] is None and empty["trials"] == [] and empty["gave_up"] is False


def test_the_log_needs_a_worker_token_and_a_valid_market(client, conn):
    w = enroll(client, conn, "w1")
    body = {"market": "futures", "family": "gap_fade", "trials": []}
    assert client.post("/api/v1/search/log", json=body).status_code == 401
    assert client.post("/api/v1/search/log", json=body, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/api/v1/search/log", params={"market": "futures", "family": "x"}).status_code == 401
    assert post(client, w, market="options").status_code == 400
    assert client.get("/api/v1/search/log", params={"market": "options", "family": "x"}, headers=auth(w)).status_code == 400
    assert post(client, w, trials=[{"params": {}}] * 1001).status_code == 400
    assert post(client, w, trials=[{"params": {}}] * 1000).json() == {"stored": 1000}


def test_curves_and_daily_arrays_are_stripped_from_metrics(client, conn):
    w = enroll(client, conn, "w1")
    metrics = {"trades": 90, "sharpe": 0.4, "curve": [1.0] * 50, "twin_curve": {"pnl": [1, 2]},
               "daily": {"days": [1, 2], "pnl": [3.0, 4.0]}, "pnl": [1.0, 2.0],
               "nested": {"curve": [1, 2, 3], "pnl_double": -3.5, "equity": [1, 2]}, "long": list(range(100)),
               "band": [1.0, 2.0, 3.0]}
    assert post(client, w, trials=[{"params": {}, "score": 0.1, "metrics": metrics}]).json() == {"stored": 1}
    stored = conn.execute("SELECT metrics FROM search_log").fetchone()["metrics"]
    assert stored == {"trades": 90, "sharpe": 0.4, "nested": {"pnl_double": -3.5}, "band": [1.0, 2.0, 3.0]}


def test_get_returns_the_best_200_and_the_latest_300_once_each(client, conn):
    w = enroll(client, conn, "w1")
    # scores fall as time goes on: the best are the oldest, so the two groups are apart
    for start in range(0, 700, 350):
        trials = [{"params": {"i": i}, "score": 1000.0 - i} for i in range(start, start + 350)]
        assert post(client, w, trials=trials).json() == {"stored": 350}
    got = client.get("/api/v1/search/log", params={"market": "futures", "family": "gap_fade"}, headers=auth(w)).json()
    ids = [t["params"]["i"] for t in got["trials"]]
    assert got["tried"] == 700 and got["scored"] == 700 and got["best_score"] == 1000.0
    assert len(ids) == 500 and len(set(ids)) == 500 and set(ids) == set(range(200)) | set(range(400, 700))
    # scores rise as time goes on: the best are the latest, so the groups overlap and are not repeated
    trials = [{"params": {"i": i}, "score": float(i)} for i in range(400)]
    post(client, w, family="pullback", trials=trials)
    got = client.get("/api/v1/search/log", params={"market": "futures", "family": "pullback"}, headers=auth(w)).json()
    ids = [t["params"]["i"] for t in got["trials"]]
    assert len(ids) == len(set(ids)) == 300 and set(ids) == set(range(100, 400))   # the two groups overlap: each once


def test_the_log_stays_fast_with_many_rows(conn):
    fill(conn, "futures", "gap_fade", 100_000, score=0.5, rejected=None)
    conn.execute("ANALYZE search_log")
    conn.execute("SET enable_seqscan = off")
    conn.execute("SET enable_bitmapscan = off")
    plan = "\n".join(r["QUERY PLAN"] for r in conn.execute(
        "EXPLAIN SELECT id FROM search_log WHERE market = 'futures' AND family = 'gap_fade' AND score IS NOT NULL "
        "ORDER BY score DESC NULLS LAST LIMIT 200").fetchall())
    assert "search_log_best" in plan and "Sort" not in plan      # the index hands over the best 200 in order
    plan = "\n".join(r["QUERY PLAN"] for r in conn.execute(
        "EXPLAIN SELECT id FROM search_log WHERE market = 'futures' AND family = 'gap_fade' "
        "ORDER BY created_at DESC, id DESC LIMIT 300").fetchall())
    assert "search_log_recent" in plan and "Sort" not in plan


# ------------------------------------------------------------------ giving up


def test_rule_a_a_family_gives_up_at_2000_trials_without_a_positive_score(client, conn):
    w = enroll(client, conn, "w1")
    fill(conn, "futures", "gap_fade", search.GIVE_UP_TRIES - 1, score=-1.0, rejected=None)
    assert search.gave_up_families(conn, "futures") == {}
    fill(conn, "futures", "gap_fade", 1, score=None)
    assert search.gave_up_families(conn, "futures") == {"gap_fade": {"tried": 2000, "rule": "tries"}}
    got = client.get("/api/v1/search/log", params={"market": "futures", "family": "gap_fade"}, headers=auth(w)).json()
    assert got["gave_up"] is True and got["tried"] == 2000
    assert search.gave_up_families(conn, "stocks") == {}                 # other markets are separate
    fill(conn, "futures", "gap_fade", 1, score=0.01, rejected=None)         # one positive score is enough hope
    assert search.gave_up_families(conn, "futures") == {}
    got = client.get("/api/v1/search/log", params={"market": "futures", "family": "gap_fade"}, headers=auth(w)).json()
    assert got["gave_up"] is False


def test_rule_b_needs_five_held_out_models_and_none_beating_the_floor_or_the_twin(conn):
    held_out_models(conn, "gap_fade", 4)
    assert search.gave_up_families(conn, "futures", FEES) == {}                         # only four reached held-out
    search_model(conn, "gap_fade", 5, shaped(0.0, 0.0, 0.0, months=3.0))
    gone = search.gave_up_families(conn, "futures", FEES)
    assert gone == {"gap_fade": {"tried": 0, "rule": "held_out"}}
    assert search.gave_up_families(conn, "futures", FEES, "pullback") == {}
    assert search.gave_up_families(conn, "futures") == {}                              # no rules, nothing decided


def test_rule_b_does_not_fire_when_one_model_beats_the_floor_or_its_twin(conn):
    held_out_models(conn, "gap_fade", 4)
    search_model(conn, "gap_fade", 5, shaped(0.0, 0.0, 300.0, months=3.0))              # net +120: well above the floor
    assert search.gave_up_families(conn, "futures", FEES) == {}
    held_out_models(conn, "pullback", 4)
    search_model(conn, "pullback", 5, shaped(0.15, 0.05, 20.0, twin_paid=0.0, months=3.0))   # passes more than its twin
    assert search.gave_up_families(conn, "futures", FEES) == {}
    # a model that only loses less than the floor because it blows up early is not an edge
    held_out_models(conn, "trend_day", 5)
    for i in range(5):
        conn.execute("UPDATE models SET metrics = %s WHERE id = %s", (Jsonb(shaped(0.0, 0.0, 0.0, months=1.0)), f"trend_day-s{i + 1}"))
    assert "trend_day" in search.gave_up_families(conn, "futures", FEES)


def test_rule_b_is_never_decided_without_the_fees(conn):
    held_out_models(conn, "gap_fade", 6)
    no_fees = replace(FEES, combine_monthly=None, activation=None)
    assert search.gave_up_families(conn, "futures", no_fees) == {}
    no_cap = replace(FEES, max_payout=None)
    assert search.gave_up_families(conn, "futures", no_cap) == {}
    assert "gap_fade" in search.gave_up_families(conn, "futures", FEES)
    # tested under other rules than today's: they do not count
    assert search.gave_up_families(conn, "futures", replace(FEES, profit_target=9_000.0)) == {}


def test_resetting_the_log_lets_a_family_that_gave_up_by_its_results_try_again(conn):
    held_out_models(conn, "gap_fade", 5)
    fill(conn, "futures", "gap_fade", 10)
    assert "gap_fade" in search.gave_up_families(conn, "futures", FEES)
    assert search.reset(conn, "futures", "pullback") == 0
    assert "gap_fade" in search.gave_up_families(conn, "futures", FEES)
    assert search.reset(conn, "futures", "gap_fade") == 10
    assert search.gave_up_families(conn, "futures", FEES) == {}
    for i in range(5):                                  # models tested after the reset count again
        search_model(conn, "gap_fade", 10 + i, shaped(0.0, 0.0, 0.0, months=3.0))
    assert "gap_fade" in search.gave_up_families(conn, "futures", FEES)


# ------------------------------------------------------------------ the jobs skip what gave up


def test_shares_never_hand_a_worker_an_empty_share():
    names = sorted(FUTURES_FILES)
    for left in (names, names[:4], names[:1], []):
        for workers in (1, 2, 3, 4, 9, 11, 30):
            parts = futures_models.shares(workers, left)
            assert len(parts) == workers and all(p["files"] or p["recipes"] for p in parts)
            assert {f for p in parts for f in p["files"]} <= set(left)
            if workers > 1 and left:
                assert {f for p in parts for f in p["files"]} == set(left)      # every live file is searched
                assert [p["recipes"] for p in parts].count(True) == 1
    assert futures_models.shares(3, ["pullback"]) == [{"files": [], "recipes": True},
                                                      {"files": ["pullback"], "recipes": False},
                                                      {"files": ["pullback"], "recipes": False}]


def futures_ready(client, conn, monkeypatch, workers: int):
    monkeypatch.setattr(futures_data, "MIN_DAYS_TO_FIX_PERIODS", 20)
    for i in range(workers):
        heartbeat(client, enroll(client, conn, f"w{i + 1}"))
    client.app.state.topstep = FEES
    load(conn)


def test_a_futures_search_leaves_out_the_files_and_recipes_that_gave_up(client, conn, monkeypatch):
    futures_ready(client, conn, monkeypatch, 3)
    fill(conn, "futures", "gap_fade", 2000, rejected="no trades")
    fill(conn, "futures", "pullback", 2000, score=-0.5, rejected=None)
    fill(conn, "futures", "recipe_abc", 2000, rejected="no trades")
    fill(conn, "futures", "recipe_ok", 2000, score=0.2, rejected=None)
    r = client.post("/api/search/start", json={"markets": ["futures"]})
    assert r.status_code == 201, r.text
    jobs = conn.execute("SELECT params FROM jobs WHERE kind = 'model_search'").fetchall()
    assert len(jobs) == 3
    live = sorted(set(FUTURES_FILES) - {"gap_fade", "pullback"})
    for j in jobs:
        assert j["params"]["skip_recipes"] == ["recipe_abc"]
        assert j["params"]["files"] or j["params"]["recipes"]
        assert not {"gap_fade", "pullback"} & set(j["params"]["files"])
    assert sorted(f for j in jobs for f in j["params"]["files"]) == live
    status = search.search_status(conn)
    assert status["detail"].endswith("· 2 model files gave up")


def test_one_job_gets_the_live_files_in_its_params(client, conn, monkeypatch):
    futures_ready(client, conn, monkeypatch, 0)
    assert client.post("/api/search/start", json={"markets": ["futures"], "target": "auto"}).status_code == 201
    params = conn.execute("SELECT params FROM jobs WHERE kind = 'model_search'").fetchone()["params"]
    assert "files" not in params and params["skip_recipes"] == []       # nothing gave up: a worker searches everything
    assert "gave up" not in (search.search_status(conn)["detail"] or "")
    client.post("/api/search/stop")
    fill(conn, "futures", "gap_fade", 2000)
    client.post("/api/search/start", json={"markets": ["futures"], "target": "auto"})
    params = conn.execute("SELECT params FROM jobs WHERE kind = 'model_search' AND status <> 'cancelled' "
                          "ORDER BY created_at DESC").fetchone()["params"]
    assert params["files"] == sorted(set(FUTURES_FILES) - {"gap_fade"})


def test_a_futures_search_skips_a_file_that_gave_up_on_its_held_out_results(client, conn, monkeypatch):
    futures_ready(client, conn, monkeypatch, 2)
    held_out_models(conn, "pullback", 5)
    params = futures_models.search_params(conn, FEES)
    assert "pullback" not in params["files"] and len(params["files"]) == len(FUTURES_FILES) - 1


def test_a_futures_search_keeps_going_on_recipes_when_every_file_gave_up(client, conn, monkeypatch):
    futures_ready(client, conn, monkeypatch, 2)
    for name in FUTURES_FILES:
        fill(conn, "futures", name, 2000)
    r = client.post("/api/search/start", json={"markets": ["futures"]})
    assert r.status_code == 201
    assert all(j["params"]["files"] == [] and j["params"]["recipes"] for j in r.json()["jobs"])


def test_a_stock_search_leaves_out_the_files_that_gave_up(client, conn):
    seed_history(conn)
    for name in ("w1", "w2"):
        heartbeat(client, enroll(client, conn, name))
    fill(conn, "stocks", "momentum", 2000)
    fill(conn, "crypto", "crypto_trend", 2000)
    r = client.post("/api/search/start", json={})
    assert r.status_code == 201
    assert [j["params"]["skip_files"] for j in r.json()["jobs"]] == [["crypto_trend", "momentum"]] * 2
    assert search.search_status(conn)["detail"].endswith(" · 2 model files gave up")
    client.post("/api/search/stop")
    r = client.post("/api/search/start", json={"markets": ["stocks"]})
    assert r.status_code == 201 and r.json()["jobs"][0]["params"]["skip_files"] == ["momentum"]


def test_a_stock_search_refuses_to_start_when_every_file_gave_up(client, conn):
    seed_history(conn)
    heartbeat(client, enroll(client, conn, "w1"))
    for name, module in STOCK_FILES.items():
        fill(conn, module.MARKET, name, 2000)
    r = client.post("/api/search/start", json={})
    assert r.status_code == 409 and r.json()["detail"] == (
        f"No model file has found an edge after {2000 * len(STOCK_FILES):,} tries; "
        "new model ideas are needed or the log can be reset")
    # only the files of the chosen market matter
    for name, module in STOCK_FILES.items():
        if module.MARKET == "stocks":
            conn.execute("DELETE FROM search_log WHERE family = %s", (name,))
    r = client.post("/api/search/start", json={"markets": ["stocks"]})
    assert r.status_code == 201 and r.json()["jobs"][0]["params"]["skip_files"] == []


# ------------------------------------------------------------------ reset, prune


def test_the_reset_command_forgets_the_log(conn, test_db_url, monkeypatch):
    fill(conn, "futures", "gap_fade", 5)
    fill(conn, "futures", "pullback", 3)
    fill(conn, "stocks", "momentum", 2)
    monkeypatch.setattr(cli.Config, "from_env", classmethod(lambda cls, env=None: Config(database_url=test_db_url)))
    count = lambda: conn.execute("SELECT count(*) AS n FROM search_log").fetchone()["n"]  # noqa: E731
    assert cli.main(["reset-search-log", "--market", "futures", "--family", "gap_fade"]) == 0 and count() == 5
    assert cli.main(["reset-search-log", "--market", "futures"]) == 0 and count() == 2
    assert cli.main(["reset-search-log"]) == 0 and count() == 0


def test_prune_keeps_the_cap_and_deletes_the_oldest_failed_rows_first(conn):
    assert search.MAX_ROWS == 50_000
    # oldest first: 10 good, then 30 failed (rejected), then 10 never scored
    conn.execute("INSERT INTO search_log (market, family, params, score, rejected, created_at) "
                 "SELECT 'futures', 'gap_fade', jsonb_build_object('i', g), 1.0, NULL, now() - interval '1 day' * (100 - g) "
                 "FROM generate_series(1, 10) g")
    conn.execute("INSERT INTO search_log (market, family, params, score, rejected, created_at) "
                 "SELECT 'futures', 'gap_fade', jsonb_build_object('i', g), -1.0, 'bad', now() - interval '1 day' * (100 - g) "
                 "FROM generate_series(11, 40) g")
    conn.execute("INSERT INTO search_log (market, family, params, score, rejected, created_at) "
                 "SELECT 'futures', 'gap_fade', jsonb_build_object('i', g), NULL, NULL, now() - interval '1 day' * (100 - g) "
                 "FROM generate_series(41, 50) g")
    fill(conn, "futures", "pullback", 5)                                   # under the cap: untouched
    assert search.prune(conn, keep=50) == 0
    assert search.prune(conn, keep=35) == 15
    left = conn.execute("SELECT (params->>'i')::int AS i FROM search_log WHERE family = 'gap_fade' ORDER BY i").fetchall()
    ids = [r["i"] for r in left]
    assert len(ids) == 35 and ids[:10] == list(range(1, 11))              # the good ones survive
    assert 11 not in ids and 25 not in ids and 26 in ids                  # the 15 oldest failed rows went
    assert conn.execute("SELECT count(*) AS n FROM search_log WHERE family = 'pullback'").fetchone()["n"] == 5
    assert search.prune(conn, keep=8) == 27 and conn.execute(
        "SELECT count(*) AS n FROM search_log WHERE family = 'gap_fade'").fetchone()["n"] == 8


def test_the_loop_prunes_at_low_frequency(conn, monkeypatch):
    calls = []

    class Pool:
        def connection(self):
            calls.append(1)
            return _Conn()

    class _Conn:
        def __enter__(self):
            return conn

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(search, "_last_prune", None)
    monkeypatch.setattr(search, "MAX_ROWS", 3)
    monkeypatch.setattr(search, "prune", lambda c, keep=3: 7)
    assert search.prune_if_due(Pool()) == 7 and search.prune_if_due(Pool()) == 0 and len(calls) == 1
    monkeypatch.setattr(search, "_last_prune", search.time.monotonic() - search.PRUNE_EVERY - 1)
    assert search.prune_if_due(Pool()) == 7 and len(calls) == 2


# ------------------------------------------------------------------ the fee floor on the futures screen


def floor_metrics(pass_rate: float, gave_up: int, failed: int) -> dict:
    m = shaped(pass_rate, 0.0, 0.0, months=3.0)
    for entry in m["held_out"]["sizes"]:
        entry["sim"].update(attempts=40, gave_up=gave_up, failed=failed, fail_rate=failed / 40)
    return m


def test_a_model_that_never_reaches_the_target_is_told_so(client, conn, fees):
    store(conn, "gap_fade", floor_metrics(0.0, 36, 4))
    store(conn, "pullback", floor_metrics(0.0, 6, 34))      # it fails fast instead: not stuck at the floor
    store(conn, "trend_day", floor_metrics(0.3, 10, 18))    # passes often enough
    html = client.get("/models?market=futures&id=gap_fade").text
    row = html[html.index('data-model="gap_fade"'):html.index('data-model="gap_fade"') + 1500]
    assert element(row.split("</li>")[0], 'data-tag="floor"') == "Never reaches the target"
    card = element(html, 'data-metric="net"')
    assert "Never reaches the target: it pays the monthly fee until time runs out." in card
    assert "The fees alone cost about $150 an attempt." in card
    assert element(html, "data-floor-note").startswith("Never reaches the target")
    assert len(attr_tags(html, 'data-tag="floor"')) == 1                 # only the one model
    other = client.get("/models?market=futures&id=pullback").text
    assert "data-floor-note" not in other
    pullback_row = other[other.index('data-model="pullback"'):other.index('data-model="trend_day"')]
    assert 'data-tag="floor"' not in pullback_row
    assert "data-floor-note" not in client.get("/models?market=futures&id=trend_day").text


def test_without_a_stored_timeout_count_it_comes_from_the_pass_and_fail_rates(client, conn, fees):
    m = floor_metrics(0.0, 0, 4)
    for entry in m["held_out"]["sizes"]:
        entry["sim"].pop("gave_up")
    store(conn, "gap_fade", m)
    assert "data-floor-note" in client.get("/models?market=futures&id=gap_fade").text


# ------------------------------------------------------------------ stock robustness flags


def robust(conn, model_id: str, roi: float, flags: list[str], passed: bool | None, trades: int = 150) -> None:
    from coordinator import models
    m = metrics_with(roi, trades)
    if passed is not None:
        m["robustness"] = {"score": 0.3, "flags": flags, "passed": passed}
    models.store_backtest(conn, model_id, m, None)


def test_robustness_flags_show_as_tags_with_a_plain_sentence(client, conn):
    robust(conn, "momentum", 0.20, ["inconsistent", "fragile"], True)
    robust(conn, "dip_buy", 0.10, ["cost_sensitive", "no_better_than_random"], True)
    html = client.get("/models?id=momentum").text
    detail = html[html.index("data-model-tags"):html.index("data-how-it-works")]
    assert "Inconsistent" in text(detail) and "Fragile settings" in text(detail)
    assert "Breaks at higher costs" not in text(detail)
    notes = {k: element(detail, f'data-robustness="{k}"') for k in ("inconsistent", "fragile")}
    assert notes["fragile"].startswith("Fragile settings.") and "settings" in notes["fragile"]
    assert all(n.count(".") <= 2 for n in notes.values())                 # one sentence beside the tag's name
    row = html[html.index('data-model="momentum"'):html.index('data-model="dip_buy"')]
    assert element(row, 'data-tag="robust-inconsistent"') == "Inconsistent"
    assert element(row, 'data-tag="robust-fragile"') == "Fragile settings"
    other = html[html.index('data-model="dip_buy"'):]
    assert element(other, 'data-tag="robust-cost_sensitive"') == "Breaks at higher costs"
    assert element(other, 'data-tag="robust-no_better_than_random"') == "No better than random"
    assert len(attr_tags(html, 'data-tag="robust-fragile"')) == 1
    assert models_view.ROBUSTNESS.keys() == {"inconsistent", "fragile", "cost_sensitive", "no_better_than_random"}


def test_a_model_that_failed_the_robustness_checks_is_shown_but_not_ranked(client, conn):
    robust(conn, "momentum", 0.50, ["fragile"], False)         # best ROI, but not robust
    robust(conn, "dip_buy", 0.05, [], True)                    # robust, ranked first despite the lower ROI
    robust(conn, "pairs", 0.20, [], None)                      # no robustness block: behaves as before
    html = client.get("/models").text
    assert re.findall(r'data-model="([^"]+)"', html)[:3] == ["pairs", "dip_buy", "momentum"]
    ranks = dict(re.findall(r'data-model="([^"]+)" data-rank="(\d*)"', html))
    assert ranks["pairs"] == "1" and ranks["dip_buy"] == "2" and ranks["momentum"] == ""
    detail = client.get("/models?id=momentum").text
    assert "Shown but not ranked" in text(detail[detail.index("data-model-tags"):detail.index("data-how-it-works")])


def test_all_models_failing_robustness_leaves_none_ranked(client, conn):
    robust(conn, "momentum", 0.50, ["inconsistent"], False)
    html = client.get("/models").text
    assert re.findall(r'data-rank="(\d*)"', html)[0] == ""
    assert "passes the robustness checks" in text(element_html(html))


def element_html(html: str) -> str:
    return html[html.index('data-notice="unranked"') - 50:html.index('data-notice="unranked"') + 400]
