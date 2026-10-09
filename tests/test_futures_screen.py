"""Futures on the coordinator and the dashboard: keeping a varied set of finds, the count
of tries, starting a futures search and a futures backtest, the Models screen's
Futures view (ranked by held-out expected net per attempt, coin-flip twin beside every
number, proxy prices labelled, "Set the fee" while fees are unset, never "ready" on
proxy prices), the once-only Final check, and no orders for futures models."""
from __future__ import annotations

import copy
import random
import re
from dataclasses import replace
from datetime import date

import numpy as np
import pytest
from psycopg.types.json import Jsonb

from coordinator import futures_data, futures_models, futures_view, models
from coordinator.futures_data import FakeFuturesSource, split_days
from fleet2.models.futures import REGISTRY
from fleet2.models.futures import recipe as R
from fleet2.models.futures.base import params_with_defaults
from fleet2.sim import futures_data as wfd
from fleet2.sim import topstep
from fleet2.sim.cme_session import as_date
from fleet2.worker import futures_jobs
from tests.conftest import enroll, heartbeat
from tests.test_dashboard import attr_tags, element, text

FEES = topstep.Rules(max_payout=2_000.0, combine_monthly=50.0, activation=100.0,
                     commission_per_side={"MES": 0.62, "MNQ": 0.62}, twin_seeds=2, max_sizes=2)
_METRICS: dict = {}


def real_metrics() -> dict:
    """A futures backtest result from the worker's own code, on synthetic prices."""
    if not _METRICS:
        src = FakeFuturesSource(seed=8, first=date(2024, 1, 2), last=date(2024, 12, 31))
        data = wfd.build("synthetic", {s: src.series(s) for s in ("MES", "MNQ")})
        periods = split_days([as_date(int(d)) for d in data.days])
        cut = lambda iso: data.until_day(data.day_index(futures_jobs.day_int(iso), side="right"))  # noqa: E731
        module = REGISTRY["gap_fade"]
        params = params_with_defaults(module, None)
        train = futures_jobs.training_numbers(cut(periods["train_end"]), module, params, FEES, keep_daily=True)
        held = futures_jobs.held_out_numbers(cut(periods["held_out_end"]), module, params, FEES, periods,
                                             train["worst_stretch"])
        _METRICS.update({"market": "futures", "feed": "synthetic", "periods": periods, "params": params,
                         "train": train, "held_out": held, "summary": futures_jobs.summary_line(train, held)})
    return copy.deepcopy(_METRICS)


def shaped(pass_rate: float, twin_rate: float, paid: float, twin_paid: float = 0.0, feed: str = "synthetic",
           months: float = 1.0) -> dict:
    """Metrics whose chosen size has these pass rates and payouts (net = 0.9 x paid -
    50 x months - 100 x pass rate)."""
    m = real_metrics()
    m["feed"] = m["held_out"]["feed"] = feed
    for entry in m["held_out"]["sizes"]:
        entry["sim"].update(pass_rate=pass_rate, mean_paid=paid, mean_months=months, attempts=40)
        entry["twin"].update(pass_rate=twin_rate, mean_paid=twin_paid, mean_months=months)
    m["held_out"]["sizes"] = m["held_out"]["sizes"][:1]
    m["held_out"]["sim_key"] = FEES.sim_key()
    m["held_out"]["double_slippage_pnl"] = 500.0
    return m


@pytest.fixture
def fees(client):
    client.app.state.topstep = FEES
    return FEES


def store(conn, model_id: str, metrics: dict) -> None:
    models.store_backtest(conn, model_id, metrics, None)


def futures_ids(html: str) -> list[str]:
    return re.findall(r'data-model="([^"]+)"', html)


# ------------------------------------------------------------------ the default view is unchanged


def test_futures_models_stay_out_of_the_stock_and_crypto_view(client, conn):
    html = client.get("/models").text
    assert not set(futures_ids(html)) & set(REGISTRY)
    assert 'data-tab="futures"' in html and 'href="/models?market=futures"' in html
    fleet = client.get("/fleet").text
    assert "Opening range" not in fleet  # not offered for paper trading in the Assign panel


def test_futures_models_never_use_the_stock_books(client, conn):
    store(conn, "gap_fade", real_metrics())
    r = client.post("/api/models/gap_fade/paper/start", json={})
    assert r.status_code == 409 and "Futures view" in r.json()["detail"]
    assert conn.execute("SELECT count(*) AS n FROM books").fetchone()["n"] == 0


# ------------------------------------------------------------------ the futures view


def test_the_futures_view_lists_the_five_starters_and_asks_for_the_fees(client, conn):
    html = client.get("/models?market=futures").text
    assert sorted(futures_ids(html)) == sorted(REGISTRY)
    assert 'data-tab="futures" class="current"' in html and 'data-market="futures"' in html
    assert "Set the fee in config/topstep.toml" in element(html, "data-fee-note")
    assert "disabled" in attr_tags(html, 'data-action="search-start"')[0]
    assert element(html, "data-futures-prices") == "No futures prices yet: run a Futures prices job"
    assert element(html, 'data-action="futures-prices"') == "Load futures prices"
    assert "Not tested yet" in element(html, "data-untested")
    assert "first paper trades on Alpaca" in element(html, "data-paper-note")
    assert "paper-start" not in html


def test_proxy_prices_are_labelled_on_screen(client, conn):
    conn.execute("INSERT INTO bar_status (symbol, timeframe, first_ts, last_ts, bars, feed, refreshed_at) VALUES "
                 "('MES', '1Min', '2019-05-06T13:30:00Z', '2026-10-07T19:59:00Z', 100, 'proxy', now())")
    m = real_metrics()
    m["feed"] = m["held_out"]["feed"] = "proxy"
    store(conn, "gap_fade", m)
    html = client.get("/models?market=futures&id=gap_fade").text
    prices = element(html, "data-futures-prices")
    assert prices.startswith("Futures prices: proxy: SPY and QQQ standing in for MES and MNQ, May 6, 2019 to Oct 7, 2026")
    assert "never count" in element(html, "data-proxy-note")
    assert element(html, 'data-tag="feed"') == "Proxy prices"
    verdict = element(html, "data-verdict-text")
    assert verdict == "Not ready for a Combine. First missing: Real futures prices (Databento)."
    assert "Proxy prices never count toward a verdict" in html
    assert 'data-ready="false"' in html


def test_unset_fees_show_a_message_instead_of_money(client, conn):
    store(conn, "gap_fade", shaped(0.5, 0.3, 1_000.0))
    html = client.get("/models?market=futures&id=gap_fade").text
    assert element(html, 'data-metric="net"').startswith("Expected net per attempt")
    net = attr_tags(html, 'data-metric="net"')
    card = html[html.index('data-metric="net"'):]
    assert "Set the fee in config/topstep.toml" in text(card[:card.index("</article>")])
    assert "Set the fee in config/topstep.toml to rank futures models" in element(html, 'data-notice="unranked"')
    assert re.findall(r'data-rank="(\d+)"', html) == []
    assert element(html, 'data-money') == "Set the fee" and net


def test_ranked_by_held_out_net_and_only_when_beating_the_twin(client, conn, fees):
    store(conn, "gap_fade", shaped(0.50, 0.30, 1_000.0))        # net 900 - 50 - 50 = $800
    store(conn, "pullback", shaped(0.45, 0.30, 2_000.0))        # net 1,800 - 50 - 45 = $1,705
    store(conn, "trend_day", shaped(0.30, 0.40, 5_000.0))       # passes less often than its coin flip
    store(conn, "vwap_revert", shaped(0.50, 0.30, 100.0, twin_paid=1_000.0))  # its coin flip makes more money
    html = client.get("/models?market=futures").text
    ranks = dict(re.findall(r'data-model="([^"]+)" data-rank="(\d*)"', html))
    assert ranks == {"pullback": "1", "gap_fade": "2", "trend_day": "", "vwap_revert": "", "opening_range": "", "fair_value_gap": ""}
    assert futures_ids(html)[:2] == ["pullback", "gap_fade"]
    rows = html.split("<li>")
    pull = next(r for r in rows if 'data-model="pullback"' in r)
    assert element(pull, "data-money") == "+$1,705" and element(pull, "data-pass-line") == "Passes 45% · coin flip 30% · −$0.59 a day of $50.00 needed"
    trend = next(r for r in rows if 'data-model="trend_day"' in r)
    assert element(trend, 'data-tag="twin"') == "No better than a coin flip"
    detail = client.get("/models?market=futures&id=pullback").text
    assert element(detail, 'data-metric="pass_rate"').startswith("Pass rate 45%")
    assert "Coin-flip twin: 30%" in element(detail, 'data-metric="pass_rate"')
    assert "+$1,705" in element(detail, 'data-metric="net"') and "Coin-flip twin: −$80" in element(detail, 'data-metric="net"')
    assert "Chance this is luck" in detail and "Held-out profit and loss, added up" in detail
    assert 'data-ref="0"' in detail and 'data-label="Coin-flip twin"' in detail


def test_a_model_tested_under_other_rules_is_not_ranked(client, conn, fees):
    m = shaped(0.5, 0.3, 1_000.0)
    m["held_out"]["sim_key"] = "older"
    store(conn, "gap_fade", m)
    html = client.get("/models?market=futures").text
    assert re.findall(r'data-rank="(\d+)"', html) == []
    assert "Rules changed: backtest again" in html


def test_never_ready_without_the_lockbox_and_shadow_trading(client, conn, fees):
    store(conn, "gap_fade", shaped(0.6, 0.2, 3_000.0, feed="databento"))
    html = client.get("/models?market=futures&id=gap_fade").text
    checks = re.findall(r'<li data-check="(\w+)">', html)
    assert checks == ["ok", "ok", "ok", "ok", "pending", "pending"]
    assert element(html, "data-verdict-text").startswith("Not ready for a Combine. First missing: Lockbox Final check")
    v = futures_view.verdict({"metrics": shaped(0.6, 0.2, 3_000.0, feed="databento")}, FEES,
                             futures_view.chosen(shaped(0.6, 0.2, 3_000.0)["held_out"], FEES), None)
    assert not v["ready"]


# ------------------------------------------------------------------ keeping finds


def found(name: str, score: float, pnl: list[float], **params) -> dict:
    days = list(range(20240101, 20240101 + len(pnl)))
    m = real_metrics()
    m["train"]["daily"] = {"days": days, "pnl": pnl}
    return {"module": name, "params": params_with_defaults(REGISTRY[name], params), "train_score": score,
            "seed": f"1:1:{name}:0", "parent": None, "metrics": m}


def noise(seed: int, n: int = 60) -> list[float]:
    return list(np.round(np.random.default_rng(seed).normal(0, 100, n), 2))


def test_a_new_find_replaces_the_closest_kept_model_when_it_scores_higher(conn):
    for i in range(5):
        r = futures_models.store_found(conn, None, found("gap_fade", 1.0 + i, noise(i), min_gap=0.1 + 0.2 * i))
        assert r["kept"]
    near_fourth = found("gap_fade", 3.5, noise(10), min_gap=0.72)  # closest to the 4th (0.7), which scores 4.0
    assert not futures_models.store_found(conn, None, near_fourth)["kept"]
    better = found("gap_fade", 4.5, noise(11), min_gap=0.72)
    reply = futures_models.store_found(conn, None, better)
    assert reply["kept"] and reply["replaced"] == "gap_fade-s4"  # not the weakest (gap_fade-s1)
    status = {r["id"]: r["status"] for r in conn.execute("SELECT id, status FROM models WHERE origin = 'search'")}
    assert status["gap_fade-s4"] == "retired" and status["gap_fade-s1"] == "backtested"
    seed = conn.execute("SELECT metrics->>'seed' AS s FROM models WHERE id = %s", (reply["id"],)).fetchone()["s"]
    assert seed == "1:1:gap_fade:0"


def test_a_find_that_trades_like_a_kept_model_is_dropped(conn):
    base = noise(1)
    assert futures_models.store_found(conn, None, found("trend_day", 2.0, base))["kept"]
    twin = [round(x * 1.1 + 1, 2) for x in base]  # 100% alike, from another file
    reply = futures_models.store_found(conn, None, found("pullback", 1.5, twin))
    assert not reply["kept"] and "alike" in reply["reason"]
    better = futures_models.store_found(conn, None, found("pullback", 3.0, twin))
    assert better["kept"] and better["replaced"] == "trend_day-s1"
    assert futures_models.correlation({1: 1.0, 2: 2.0}, {1: 1.0}) is None


def recipe_found(i: int, score: float, pnl: list[float]) -> dict:
    module = R.family(R.random_recipe(random.Random(f"kept recipe {i}")))
    days = list(range(20240101, 20240101 + len(pnl)))
    m = real_metrics()
    m["train"]["daily"] = {"days": days, "pnl": pnl}
    return {"module": module.__name__, "params": params_with_defaults(module, None), "train_score": score,
            "seed": f"1:1:{module.__name__}:0", "parent": None, "metrics": m}


def test_recipe_finds_share_one_pool_and_keep_their_recipe(conn, monkeypatch):
    monkeypatch.setattr(futures_models, "KEEP_RECIPES", 3)
    finds = [recipe_found(i, 1.0 + i, noise(100 + i)) for i in range(3)]
    for body in finds:
        assert futures_models.store_found(conn, None, body)["kept"]
    weak = futures_models.store_found(conn, None, recipe_found(7, 0.5, noise(107)))
    assert not weak["kept"] and "weakest of 3 kept recipe models" in weak["reason"]
    strong = futures_models.store_found(conn, None, recipe_found(8, 9.0, noise(108)))
    assert strong["kept"] and strong["replaced"] == finds[0]["module"] + "-s1"  # the lowest score goes
    row = conn.execute("SELECT * FROM models WHERE id = %s", (strong["id"],)).fetchone()
    assert R.name_of(row["params"]["recipe"]) == row["module"] and row["metrics"]["recipe_by"] == "random"
    assert row["description"].startswith("A recipe model: ") and row["name"].endswith(" #1")
    detail = futures_view.detail(row, FEES, {}, None, False, "synthetic")
    assert "recipe put together by a random mix of building blocks" in detail["origin"]
    assert "recipe" not in detail["settings"] and "stop ticks" in detail["settings"]
    # The five files keep their own limits beside the recipes.
    assert futures_models.store_found(conn, None, found("gap_fade", 1.0, noise(200)))["kept"]


def same_recipe_found(score: float, seed: int, **params) -> dict:
    body = recipe_found(1, score, noise(300 + seed))
    module = R.family(body["params"]["recipe"])
    body["params"] = params_with_defaults(module, params)
    return body


def test_one_recipe_cannot_fill_every_place(conn):
    finds = [same_recipe_found(1.0 + i, i, start_minute=30 + 60 * i) for i in range(3)]
    for body in finds:
        assert futures_models.store_found(conn, None, body)["kept"]
    name = finds[0]["module"]
    near = same_recipe_found(1.5, 9, start_minute=95)  # closest to the 2nd (90), which scores 2.0
    reply = futures_models.store_found(conn, None, near)
    assert not reply["kept"] and "closest kept model of this recipe" in reply["reason"]
    better = same_recipe_found(2.5, 10, start_minute=95)
    reply = futures_models.store_found(conn, None, better)
    assert reply["kept"] and reply["replaced"] == name + "-s2"
    kept = conn.execute("SELECT count(*) AS n FROM models WHERE module = %s AND status = 'backtested'",
                        (name,)).fetchone()["n"]
    assert kept == futures_models.KEEP_PER_RECIPE
    # Another recipe still gets a place.
    assert futures_models.store_found(conn, None, recipe_found(5, 0.5, noise(400)))["kept"]


def test_extra_models_of_one_recipe_are_retired_but_never_a_trading_or_checked_one(conn):
    models.sync_starters(conn)
    module = R.family(R.random_recipe(random.Random("crowded")))
    for i in range(6):  # as box1 had them before the limit: one recipe, many places
        conn.execute("INSERT INTO models (id, name, module, market, description, how_it_works, params, status, origin, "
                     "metrics) VALUES (%s, %s, %s, 'futures', '', '', %s, 'backtested', 'search', %s)",
                     (f"{module.__name__}-s{i + 1}", f"{module.NAME} #{i + 1}", module.__name__,
                      Jsonb(params_with_defaults(module, None)), Jsonb({"train_score": float(i)})))
    conn.execute("INSERT INTO futures_books (model_id, venue, contracts) VALUES (%s, 'alpaca_paper', 1)",
                 (module.__name__ + "-s1",))  # the weakest, but trading on paper
    retired = futures_models.trim_recipes(conn)
    assert sorted(retired) == sorted(f"{module.__name__}-s{i}" for i in (2, 3, 4))
    left = {r["id"] for r in conn.execute("SELECT id FROM models WHERE module = %s AND status = 'backtested'",
                                          (module.__name__,))}
    assert left == {f"{module.__name__}-s{i}" for i in (1, 5, 6)}  # the trading one and the two best
    assert futures_models.trim_recipes(conn) == []


def test_a_find_whose_recipe_does_not_match_its_name_is_refused(conn):
    body = recipe_found(1, 2.0, noise(5))
    body["params"]["recipe"] = R.random_recipe(random.Random("another"))
    with pytest.raises(Exception, match="not " + body["module"]):
        futures_models.store_found(conn, None, body)


def test_recipe_tries_are_counted_together(client, conn, fees):
    w = enroll(client, conn, "w1")
    hdr = {"Authorization": "Bearer " + w["worker_token"]}
    r = client.post("/api/v1/search/tries", json={"tries": {"recipe": {"n": 40, "sum": 1.0, "sq": 0.5}}}, headers=hdr)
    assert r.status_code == 200 and futures_models.tries(conn)["recipe"]["n"] == 40
    m = {"module": R.name_of(R.random_recipe(random.Random(1))), "origin": "search",
         "metrics": {"train": {"sr_day": 0.08, "n_days": 150, "skew": 0.0, "kurt": 3.0}}}
    assert futures_view.luck(m, futures_models.tries(conn))["trials"] == 40


def test_tries_are_counted_and_feed_the_luck_figure(client, conn, fees):
    w = enroll(client, conn, "w1")
    hdr = {"Authorization": "Bearer " + w["worker_token"]}
    for _ in range(2):
        r = client.post("/api/v1/search/tries", json={"tries": {"gap_fade": {"n": 50, "sum": 1.0, "sq": 0.5}}}, headers=hdr)
        assert r.status_code == 200
    assert futures_models.tries(conn)["gap_fade"] == {"n": 100, "sum": 2.0, "sq": 1.0}
    assert client.post("/api/v1/search/tries", json={"tries": {"nope": {"n": 1}}}, headers=hdr).status_code == 400
    m = {"module": "gap_fade", "origin": "search", "metrics": {"train": {"sr_day": 0.08, "n_days": 150, "skew": 0.0, "kurt": 3.0}}}
    few = futures_view.luck(m, {"gap_fade": {"n": 2, "sum": 0.0, "sq": 0.0002}})
    many = futures_view.luck(m, futures_models.tries(conn))
    assert many["trials"] == 100 and many["value"] > few["value"]
    store(conn, "gap_fade", real_metrics())
    html = client.get("/models?market=futures&id=gap_fade").text
    assert "100 futures settings tried so far" in html


# ------------------------------------------------------------------ starting jobs from the dashboard


def load(conn, days: int = 40) -> None:
    """Synthetic futures bars for `days` trading days, straight into the table."""
    src = FakeFuturesSource(seed=3, first=date(2024, 1, 2), last=date(2024, 3, 31))
    s = src.series("MES")
    n = days * 390
    rows = [("MES", "1Min", int(s["t"][i]), s["o"][i], s["h"][i], s["l"][i], s["c"][i], s["v"][i], "synthetic",
             int(s["iid"][i])) for i in range(n)]
    conn.cursor().executemany("INSERT INTO bars (symbol, timeframe, ts, open, high, low, close, volume, feed, "
                              "instrument_id) VALUES (%s, %s, to_timestamp(%s), %s, %s, %s, %s, %s, %s, %s)", rows)
    conn.execute("INSERT INTO bar_status (symbol, timeframe, first_ts, last_ts, bars, feed, refreshed_at) "
                 "SELECT 'MES', '1Min', min(ts), max(ts), count(*), 'synthetic', now() FROM bars WHERE timeframe = '1Min'")


def test_a_futures_search_starts_from_the_dashboard_once_the_fees_are_set(client, conn, monkeypatch):
    monkeypatch.setattr(futures_data, "MIN_DAYS_TO_FIX_PERIODS", 20)
    heartbeat(client, enroll(client, conn, "w1"))
    r = client.post("/api/search/start", json={"markets": ["futures"]})
    assert r.status_code == 409 and r.json()["detail"].startswith("Set the fee in config/topstep.toml")
    client.app.state.topstep = FEES
    r = client.post("/api/search/start", json={"markets": ["futures"]})
    assert r.status_code == 409 and "Load futures prices" in r.json()["detail"]
    load(conn)
    r = client.post("/api/search/start", json={"markets": ["futures"]})
    assert r.status_code == 201 and r.json()["message"] == "Futures model search started on 1 worker"
    params = conn.execute("SELECT params FROM jobs WHERE kind = 'model_search'").fetchone()["params"]
    assert params["markets"] == ["futures"] and params["rules"]["combine_monthly"] == 50.0
    assert set(params["periods"]) >= {"train_end", "held_out_end", "lockbox_end"}
    assert client.post("/api/search/start", json={"markets": ["futures", "stocks"]}).status_code == 400
    html = client.get("/models?market=futures").text
    assert element(html, 'data-action="search-stop"') == "Stop model search"


def test_a_futures_backtest_needs_the_fees_and_keeps_how_search_found_it(client, conn, monkeypatch):
    monkeypatch.setattr(futures_data, "MIN_DAYS_TO_FIX_PERIODS", 20)
    load(conn)
    r = client.post("/api/jobs", json={"kind": "backtest", "model_id": "gap_fade", "target": "auto"})
    assert r.status_code == 409 and "Set the fee" in r.json()["detail"]
    client.app.state.topstep = FEES
    r = client.post("/api/jobs", json={"kind": "backtest", "model_id": "gap_fade", "target": "auto"})
    assert r.status_code == 201, r.text
    params = conn.execute("SELECT params FROM jobs WHERE kind = 'backtest'").fetchone()["params"]
    assert params["market"] == "futures" and params["module"] == "gap_fade" and "train_end" in params["periods"]
    reply = futures_models.store_found(conn, None, found("gap_fade", 2.0, noise(3)))
    store(conn, reply["id"], real_metrics())  # backtested again later
    kept = conn.execute("SELECT metrics FROM models WHERE id = %s", (reply["id"],)).fetchone()["metrics"]
    assert kept["seed"] == "1:1:gap_fade:0" and kept["train_score"] == 2.0 and "held_out" in kept


# ------------------------------------------------------------------ the Final check


def test_the_final_check_runs_once_and_is_kept_forever(client, conn, monkeypatch):
    monkeypatch.setattr(futures_data, "MIN_DAYS_TO_FIX_PERIODS", 20)
    client.app.state.topstep = FEES
    w = enroll(client, conn, "w1")
    heartbeat(client, w)
    assert client.post("/api/models/gap_fade/final-check").status_code == 409  # not tested
    store(conn, "gap_fade", shaped(0.5, 0.3, 1_000.0))
    conn.execute("INSERT INTO bar_status (symbol, timeframe, bars, feed, refreshed_at) VALUES ('MES', '1Min', 1, 'proxy', now())")
    r = client.post("/api/models/gap_fade/final-check")
    assert r.status_code == 409 and "proxy prices can never count" in r.json()["detail"]
    html = client.get("/models?market=futures&id=gap_fade").text
    assert "disabled" in attr_tags(html, 'data-action="final-check"')[0]
    assert "Proxy prices never count" in element(html, "data-final-reason")
    conn.execute("DELETE FROM bar_status")
    load(conn)
    r = client.post("/api/models/gap_fade/final-check")
    assert r.status_code == 201, r.text
    job = conn.execute("SELECT * FROM jobs WHERE kind = 'final_check'").fetchone()
    assert job["params"]["contracts"] >= 1 and job["model_id"] == "gap_fade"
    assert client.post("/api/models/gap_fade/final-check").status_code == 409  # already running
    hdr = {"Authorization": "Bearer " + w["worker_token"]}
    result = copy.deepcopy(real_metrics()["held_out"])
    result["contracts"] = 1
    body = {"job_id": str(job["id"]), "feed": "synthetic", "result": result}
    assert client.post("/api/v1/models/gap_fade/final-check", json=body, headers=hdr).status_code == 409  # not leased yet
    heartbeat(client, w, want_job=True)
    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job["id"],)).fetchone()["status"] == "leased"
    assert client.post("/api/v1/models/gap_fade/final-check", json=body, headers=hdr).status_code == 200
    assert client.post("/api/v1/models/gap_fade/final-check", json=body, headers=hdr).status_code == 409  # once
    conn.execute("UPDATE jobs SET status = 'succeeded' WHERE id = %s", (job["id"],))
    again = client.post("/api/models/gap_fade/final-check")
    assert again.status_code == 409 and "kept forever" in again.json()["detail"]
    html = client.get("/models?market=futures&id=gap_fade").text
    assert element(html, 'data-final-check="stored"').startswith("Final check, kept forever: the lockbox passes")
    assert 'data-action="final-check"' not in html
    assert "Lockbox Final check" in html and "Lockbox passes" in html


# ------------------------------------------------------------------ profit per day


def with_daily(total: float, twin_total: float, days: int = 470) -> dict:
    m = shaped(0.0, 0.0, 0.0, months=3.0)
    m["held_out"]["numbers"]["days"] = days
    m["held_out"]["sizes"][0].update(net_pnl=total, twin_pnl=twin_total)
    return m


@pytest.mark.parametrize("total, value, tone", [
    (203.0, "$0.43", "warn"),        # Gap fade #1 on box1: $203 over 470 days, far from $50 a day
    (-575.0, "−$1.22", "loss"),
    (470 * 60.0, "$60.00", "gain"),  # enough to reach $3,000 in 60 days on average
])
def test_profit_per_day_says_how_far_a_model_is_from_a_combine(client, conn, fees, total, value, tone):
    store(conn, "gap_fade", with_daily(total, -40.0))
    html = client.get("/models?market=futures&id=gap_fade").text
    card = element(html, 'data-metric="per_day"')
    assert card.startswith(f"Profit per day {value}")
    assert "needs about $50.00 a day" in card and "coin-flip twin: −$0.09" in card
    assert f'data-tone="{tone}"' in attr_tags(html, 'data-metric="per_day"')[0]
    row = element(html, 'data-model="gap_fade"')
    assert f"{value} a day of $50.00 needed" in row


def test_profit_per_day_follows_the_rules(conn):
    m = {"id": "gap_fade", "module": "gap_fade", "metrics": with_daily(600.0, 0.0, days=100)}
    pick = futures_view.chosen(m["metrics"]["held_out"], FEES)
    daily = futures_view.per_day(m, replace(FEES, profit_target=6000.0, combine_max_days=40), pick)
    assert daily == {"value": 6.0, "twin": 0.0, "needed": 150.0, "days": 100, "contracts": 1}
