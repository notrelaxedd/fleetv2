"""The Combine measure of model search: training days through Topstep's rules at every
contract size the held-out test would try (fleet2.worker.futures_jobs.training_combine),
and its one number, combine_score. A steady small profit only runs out the clock and
scores the floor; a solid daily edge passes at a larger size; a wild model mostly fails;
the many-sizes-at-once simulation gives simulate()'s own numbers; only training prices
are read; the output is small plain JSON; without fees it falls back to passes a month."""
from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from coordinator.futures_data import FakeFuturesSource, split_days
from fleet2.models.futures import REGISTRY
from fleet2.models.futures.base import params_with_defaults
from fleet2.sim import futures_backtest as fb
from fleet2.sim import futures_data as wfd
from fleet2.sim import topstep
from fleet2.sim.cme_session import as_date
from fleet2.worker import futures_jobs
from tests.test_futures_search import cut, planted

FEES = {"MES": 0.62, "MNQ": 0.62}
RULES = topstep.Rules(max_payout=2_000.0, combine_monthly=49.0, activation=149.0, commission_per_side=FEES)
NO_FEES = topstep.Rules(commission_per_side=FEES)  # account fees and payout cap unset, as in config/topstep.toml
N = 600


def steady() -> tuple[np.ndarray, np.ndarray]:
    """$3 a day on average at one contract: even at 10 contracts, $1,800 in 60 days."""
    pnl = np.tile([8.0, -2.0], N // 2)
    return pnl, np.minimum(pnl, 0) - 5


def solid() -> tuple[np.ndarray, np.ndarray]:
    """$15 a day with a $25 spread at one contract: too slow at one, plenty at five."""
    rng = np.random.default_rng(3)
    pnl = rng.normal(15, 25, N)
    return pnl, np.minimum(pnl, 0) - np.abs(rng.normal(0, 10, N))


def wild() -> tuple[np.ndarray, np.ndarray]:
    """$400 swings and deep dips on a $5 edge: one contract is already too many."""
    rng = np.random.default_rng(4)
    pnl = rng.normal(5, 400, N)
    return pnl, np.minimum(pnl, 0) - np.abs(rng.normal(0, 300, N))


# ------------------------------------------------------------------ the three kinds of model


def test_a_steady_small_profit_only_runs_out_the_clock_and_scores_the_floor():
    c = futures_jobs.combine_numbers(*steady(), RULES)
    assert c["max_size"] == 10 and not c["too_risky"]
    assert all(row["timeout_rate"] == 1.0 and row["score"] == c["floor"] for row in c["sizes"])
    assert c["timeout_rate"] == 1.0 and c["pass_rate"] == 0.0 and c["mean_days_to_pass"] is None
    assert c["score"] == c["floor"] == -(49.0 * 3 + 149.0)
    assert c["net_per_attempt"] == pytest.approx(-147.0)  # three months of fees: the -$147 of the 40-hour search
    assert c["measure"] == "net_per_attempt" and c["note"] is None and c["slippage"] == "normal"


def test_a_solid_daily_edge_passes_at_a_larger_size():
    c = futures_jobs.combine_numbers(*solid(), RULES)
    one = c["sizes"][0]
    assert one["pass_rate"] == 0.0 and one["score"] == c["floor"]  # $15 a day cannot make $3,000 in 60 days
    assert c["best_size"] > 2 and c["pass_rate"] > 0.9 and c["fail_rate"] == 0.0
    assert c["score"] == c["sizes"][c["best_size"] - 1]["score"] == max(r["score"] for r in c["sizes"])
    assert c["score"] > 0 and c["net_per_attempt"] == pytest.approx(c["score"], abs=0.01)
    assert 1 <= c["mean_days_to_pass"] <= RULES.combine_max_days
    assert c["timeout_rate"] == pytest.approx(1 - c["pass_rate"] - c["fail_rate"], abs=1e-3)


def test_a_wild_model_mostly_fails_at_its_only_size():
    c = futures_jobs.combine_numbers(*wild(), RULES)
    assert c["too_risky"] and c["max_size"] == 1 and c["best_size"] == 1
    assert c["fail_rate"] > 0.5 and c["fail_rate"] > c["pass_rate"]
    assert c["score"] < futures_jobs.combine_numbers(*solid(), RULES)["score"]


def test_a_size_with_no_pass_scores_the_floor_even_when_it_fails_fast():
    """Failing fast costs fewer monthly fees than running out the clock, but neither ever
    earns: both sit at the floor, below any size that passes at all."""
    pnl = np.full(N, -100.0)
    c = futures_jobs.combine_numbers(pnl, pnl - 50, RULES)
    assert c["fail_rate"] == 1.0 and c["net_per_attempt"] > c["floor"] and c["score"] == c["floor"]
    rare = futures_jobs.combine_numbers(*solid(), replace(RULES, max_sizes=3))
    assert 0 < rare["pass_rate"] < 1 and rare["score"] > c["score"]


# ------------------------------------------------------------------ without fees


def test_fees_unset_falls_back_to_combines_passed_per_month_of_fees():
    for rules, note in ((NO_FEES, topstep.FEE_MESSAGE), (replace(RULES, max_payout=None), topstep.PAYOUT_CAP_MESSAGE)):
        assert futures_jobs.combine_floor(rules) == 0.0
        s = futures_jobs.combine_numbers(*steady(), rules)
        assert s["measure"] == "passes_per_month" and s["note"] == note
        assert s["net_per_attempt"] is None and s["score"] == s["floor"] == 0.0
        g = futures_jobs.combine_numbers(*solid(), rules)
        assert g["score"] == pytest.approx(g["pass_rate"] / g["mean_months"], abs=1e-3) and g["score"] > 0
        assert g["best_size"] > 2 and g["passes_per_month"] == g["score"]
        assert all(row["mean_paid"] is None and row["net"] is None for row in g["sizes"])


def test_too_few_training_days_for_an_attempt_give_no_score():
    pnl, dip = solid()
    c = futures_jobs.combine_numbers(pnl[:30], dip[:30], RULES)
    assert c["attempts"] == 0 and c["score"] is None and c["best_size"] is None


# ------------------------------------------------------------------ the simulator, many sizes at once


@pytest.mark.parametrize("rules", [
    RULES,
    replace(RULES, daily_loss_limit=300.0),
    replace(RULES, consistency_kind="total_profit", best_day_share=0.4, floor_to_start_after_payout=False),
    replace(RULES, express_days=30, combine_max_days=40),
])
def test_all_sizes_at_once_give_simulates_own_numbers(rules):
    rng = np.random.default_rng(11)
    pnl = rng.normal(20, 60, 400)
    dip = np.minimum(pnl, 0) - np.abs(rng.normal(0, 30, 400))
    days = [(pnl * k, dip * k) for k in range(1, 8)]
    batched = futures_jobs.simulate_sizes(days, rules, express_cap=None)
    for (p, d), mine in zip(days, batched):
        sim = topstep.simulate(p, d, rules)
        for key in ("attempts", "passed", "failed", "gave_up", "pass_rate", "fail_rate", "median_days_to_pass",
                    "mean_months", "mean_paid"):
            assert mine[key] == pytest.approx(sim[key], abs=1e-9), key
        assert mine["express_every"] == 1


def test_payouts_on_any_day_are_simulated_size_by_size():
    rules = replace(RULES, payout_min_day_profit=0.0)
    pnl, dip = solid()
    days = [(pnl * k, dip * k) for k in (1, 4)]
    for (p, d), mine in zip(days, futures_jobs.simulate_sizes(days, rules)):
        assert mine["mean_paid"] == pytest.approx(topstep.simulate(p, d, rules)["mean_paid"])


def test_a_capped_express_sample_keeps_the_passes_exact_and_the_money_close():
    pnl, dip = solid()
    days = [(pnl * k, dip * k) for k in range(1, 8)]
    exact = futures_jobs.simulate_sizes(days, RULES, express_cap=None)
    sampled = futures_jobs.simulate_sizes(days, RULES, express_cap=800)
    assert sampled[0]["express_every"] > 1
    for e, s in zip(exact, sampled):
        assert (s["passed"], s["failed"], s["gave_up"], s["mean_months"]) == (e["passed"], e["failed"], e["gave_up"],
                                                                             e["mean_months"])
        assert s["mean_paid"] == pytest.approx(e["mean_paid"], rel=0.05, abs=1.0)


# ------------------------------------------------------------------ in model search, on training prices


@pytest.fixture(scope="module")
def full() -> wfd.FuturesData:
    src = FakeFuturesSource(seed=8, first=date(2023, 1, 3), last=date(2024, 12, 31))
    return planted(wfd.build("synthetic", {s: src.series(s) for s in ("MES", "MNQ")}))


@pytest.fixture(scope="module")
def periods(full) -> dict:
    return split_days([as_date(int(d)) for d in full.days])


@pytest.fixture(autouse=True)
def fewer_days(monkeypatch):
    monkeypatch.setattr(futures_jobs, "MIN_DAYS_TRADED", 20)


def trend_day():
    module = REGISTRY["trend_day"]
    return module, params_with_defaults(module, None)


def test_a_gate_passer_gets_the_combine_numbers_and_the_score_is_unchanged(full, periods):
    train = cut(full, periods["train_end"])
    module, params = trend_day()
    numbers = futures_jobs.training_numbers(train, module, params, RULES)
    assert numbers["passes"]
    double = fb.run(train, module, params, topstep.costs_of(RULES).doubled(), 1, topstep.day_rules(RULES))
    assert numbers["score"] == pytest.approx(min(fb.daily_sharpe(p) for p in np.array_split(double.pnl, 4)))
    c = numbers["combine"]
    assert numbers["combine_score"] == c["score"] is not None
    assert c["attempts"] == train.n_days - RULES.combine_max_days + 1
    normal = fb.run(train, module, params, topstep.costs_of(RULES), 1, topstep.day_rules(RULES))
    assert c == futures_jobs.combine_numbers(normal.pnl, normal.dip, RULES)  # normal slippage, one contract scaled
    assert c["max_size"] == topstep.size_limit(numbers["worst_stretch"], RULES)[0]  # the sizes held-out will try
    kept = futures_jobs.training_numbers(train, module, params, RULES, keep_daily=True)
    assert kept["combine"] == c


def test_a_candidate_that_fails_a_gate_is_not_simulated(full, periods, monkeypatch):
    train = cut(full, periods["train_end"])
    module, params = trend_day()
    monkeypatch.setattr(futures_jobs, "MIN_DAYS_TRADED", 10_000)
    called = []
    monkeypatch.setattr(futures_jobs, "training_combine", lambda *a, **k: called.append(1))
    numbers = futures_jobs.training_numbers(train, module, params, RULES)
    assert not numbers["passes"] and numbers["combine"] is None and numbers["combine_score"] is None and not called


def test_sizes_scaled_from_one_contract_match_backtests_at_each_size(full, periods, monkeypatch):
    train = cut(full, periods["train_end"])
    module, params = trend_day()
    rules = replace(RULES, sizing_share=1.0, max_sizes=4)
    scaled = futures_jobs.training_combine(train, module, params, rules)
    assert scaled["sized_by"] == "scaled"
    monkeypatch.setattr(futures_jobs, "whole_targets", lambda targets: False)
    rerun = futures_jobs.training_combine(train, module, params, rules)
    assert rerun["sized_by"] == "rerun" and rerun["max_size"] == scaled["max_size"] > 1
    for a, b in zip(scaled["sizes"], rerun["sizes"]):
        for key, value in a.items():
            assert b[key] == pytest.approx(value, abs=0.02), key


def test_only_training_prices_are_read(full, periods, monkeypatch):
    end = futures_jobs.day_int(periods["train_end"])
    train = cut(full, periods["train_end"])
    seen = []
    real_run, real_targets = fb.run, fb.model_targets

    def run(data, *a, **k):
        seen.append(int(data.days[-1]))
        return real_run(data, *a, **k)

    def targets(data, *a, **k):
        seen.append(int(data.days[-1]))
        return real_targets(data, *a, **k)

    monkeypatch.setattr(fb, "run", run)
    monkeypatch.setattr(fb, "model_targets", targets)
    module, params = trend_day()
    numbers = futures_jobs.training_numbers(train, module, params, RULES)
    futures_jobs.training_combine(train, module, params, RULES)
    assert seen and max(seen) <= end and numbers["combine"]["attempts"] <= train.n_days
    # Prices after the end of training changed beyond recognition: the same numbers.
    later = full.day_index(end, side="right")
    a = int(full.day_start[later])
    shifted = wfd.FuturesData(**{**full.__dict__, **{k: np.r_["-1", getattr(full, k)[:, :a], getattr(full, k)[:, a:] * 3]
                                                     for k in ("open", "high", "low", "close")}})
    again = futures_jobs.training_numbers(cut(shifted, periods["train_end"]), module, params, RULES)
    assert again["combine"] == numbers["combine"] and again["combine_score"] == numbers["combine_score"]


def test_the_output_is_small_plain_json(full, periods):
    train = cut(full, periods["train_end"])
    module, params = trend_day()
    numbers = futures_jobs.training_numbers(train, module, params, replace(RULES, sizing_share=1.0))
    c = numbers["combine"]
    assert json.loads(json.dumps(numbers)) == numbers
    assert len(c["sizes"]) == c["max_size"] <= RULES.max_sizes
    assert len(json.dumps(c)) < 3_000  # no daily arrays: one short row per size
    for row in c["sizes"]:
        assert all(v is None or isinstance(v, (int, float, str, bool)) for v in row.values())
    assert all(not isinstance(v, (list, dict)) for k, v in c.items() if k != "sizes")
    assert not any(isinstance(v, float) and not math.isfinite(v) for v in c.values())
