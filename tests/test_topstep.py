"""The Topstep rules simulator against hand-worked examples, the rules file, and the
model evaluation (contract sizes and coin-flip twins). Every dollar figure below is
worked out by hand in the comments."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from coordinator.config import REPO_ROOT
from fleet2.sim import topstep as ts
from tests.test_futures_backtest import COSTS, every_20_minutes, scripted, wiggly

RULES = ts.Rules(max_payout=2_000.0, combine_monthly=50.0, activation=100.0,
                 commission_per_side={"MES": 0.5, "MNQ": 0.4})


def floors(pnl, dip=None, rules=RULES):
    rows = ts.trace(pnl, dip if dip is not None else [min(0.0, p) for p in pnl], rules)
    return rows


# ------------------------------------------------------------------ the trailing loss floor


def test_the_floor_follows_the_end_of_day_high():
    # $50,000 account, $2,000 limit: the floor starts at $48,000. The day ends at $50,500,
    # so the floor moves to $48,500.
    rows = floors([500.0])
    assert rows[0]["floor"] == 48_000 and rows[0]["balance"] == 50_500 and rows[0]["floor_after"] == 48_500


def test_a_loss_the_next_day_does_not_lower_the_floor():
    rows = floors([500.0, -400.0, 0.0])
    assert [r["floor"] for r in rows] == [48_000, 48_500, 48_500]  # still $48,500 after the $400 loss
    assert rows[1]["balance"] == 50_100
    # So a $1,600 dip on day 3 (50,100 - 1,600 = 48,500) reaches the floor and fails.
    assert floors([500.0, -400.0, 0.0], [0.0, -400.0, -1_600.0])[-1]["outcome"] == "failed"
    assert floors([500.0, -400.0, 0.0], [0.0, -400.0, -1_599.0])[-1]["outcome"] == "running"


def test_the_floor_stops_at_the_starting_balance():
    # Ending a day at $52,900 would put the floor at $50,900, but it stops at $50,000.
    rows = floors([2_900.0, -2_899.0, -1.0], rules=replace(RULES, consistency_kind="total_profit", best_day_share=1.0))
    assert rows[0]["floor_after"] == 50_000
    assert rows[1]["outcome"] == "running" and rows[1]["balance"] == 50_001  # $50,001 is still above it
    assert rows[2]["outcome"] == "failed"  # $50,000 reaches it


def test_an_intraday_dip_through_the_floor_fails_even_when_the_day_closes_green():
    rows = floors([100.0], [-2_000.0])  # down $2,000 at the worst moment, up $100 at the close
    assert rows == [{"floor": 48_000, "low": 48_000, "balance": 48_000, "outcome": "failed"}]
    assert floors([100.0], [-1_999.0])[0]["outcome"] == "running"


# ------------------------------------------------------------------ consistency, both kinds


def test_consistency_measured_against_the_target():
    # Best day may be at most 50% of the $3,000 target: $1,500.
    rule = replace(RULES, consistency_kind="target", best_day_share=0.5)
    assert floors([1_500.0, 1_500.0], rules=rule)[-1]["outcome"] == "passed"
    rows = floors([1_600.0, 1_500.0, 2_000.0], rules=rule)
    assert [r["outcome"] for r in rows] == ["running", "running", "running"]  # a $1,600 day can never pass


def test_consistency_measured_against_total_profit():
    # Best day may be at most 55% of the total profit.
    rule = replace(RULES, consistency_kind="total_profit", best_day_share=0.55)
    assert floors([1_600.0, 1_500.0], rules=rule)[-1]["outcome"] == "passed"  # 1,600 <= 0.55 x 3,100 = 1,705
    rows = floors([2_000.0, 1_100.0, 600.0], rules=rule)
    # day 2: profit $3,100 but 2,000 > 1,705; day 3: profit $3,700 and 2,000 <= 0.55 x 3,700 = 2,035
    assert [r["outcome"] for r in rows] == ["running", "running", "passed"]


def test_the_fast_combine_agrees_with_the_trace_from_every_start_day():
    rng = np.random.default_rng(3)
    pnl = np.round(rng.normal(60, 600, 300))
    dip = np.minimum(0.0, pnl) - np.abs(np.round(rng.normal(0, 300, 300)))
    for rule in (RULES, replace(RULES, consistency_kind="total_profit", best_day_share=0.55),
                 replace(RULES, daily_loss_limit=800.0)):
        fast = ts.combine(pnl, dip, rule)
        for s in range(0, 300, 7):
            if s + rule.combine_max_days > 300:
                assert fast["state"][s] == 0  # too late in the period to start
                continue
            rows = ts.trace(pnl[s:].tolist(), dip[s:].tolist(), rule)
            outcome = rows[-1]["outcome"]
            assert {"passed": 1, "failed": 2, "gave up": 3}[outcome] == fast["state"][s]
            assert fast["days"][s] == len(rows) and fast["end"][s] == s + len(rows) - 1


# ------------------------------------------------------------------ the Express Funded account


def test_a_payout_and_the_floor_reset():
    # Five days of +$200 (each a winning day of $150 or more): balance $51,000, the floor
    # has trailed to $49,000. Payout: 50% of the $1,000 profit = $500, leaving $50,500,
    # and the floor moves to the starting balance, $50,000.
    pnl = np.array([200.0] * 5 + [0.0])
    dip = np.array([0.0] * 5 + [-500.0])  # day 6 dips $500: 50,500 - 500 = 50,000
    e = ts.express(pnl, dip, np.array([0]), RULES)
    assert e["paid"].tolist() == [500.0] and e["payouts"].tolist() == [1]
    assert e["breached"].tolist() == [True] and e["days"].tolist() == [6]
    kept = ts.express(pnl, dip, np.array([0]), replace(RULES, floor_to_start_after_payout=False))
    assert kept["breached"].tolist() == [False]  # without the reset the floor was still $49,000


def test_payout_rules_winning_days_and_the_cap():
    small = np.array([100.0] * 10)  # $100 days are not winning days of $150+
    assert ts.express(small, np.zeros(10), np.array([0]), RULES)["payouts"].tolist() == [0]
    big = np.array([1_000.0] * 5)  # $5,000 profit: 50% is $2,500, capped at $300
    e = ts.express(big, np.zeros(5), np.array([0]), replace(RULES, max_payout=300.0))
    assert e["paid"].tolist() == [300.0]
    four = np.array([200.0] * 4 + [100.0] * 6)
    assert ts.express(four, np.zeros(10), np.array([0]), RULES)["payouts"].tolist() == [0]
    with pytest.raises(ValueError, match="max_payout"):
        ts.express(big, np.zeros(5), np.array([0]), replace(RULES, max_payout=None))


def test_simulate_and_the_money_per_attempt():
    # Attempts give up after 2 days here, so they start on days 0 to 5 (6 attempts).
    # Day 0 and 1: +$1,500 each, so the attempt from day 0 passes on day 1 (2 days, 1 month);
    # every other attempt gives up after 2 days (1 month). Express from day 2: five +$200
    # days, then a payout of 50% of $1,000 = $500.
    rule = replace(RULES, combine_max_days=2)
    pnl = np.array([1_500.0, 1_500.0, 200, 200, 200, 200, 200])
    sim = ts.simulate(pnl, np.zeros(7), rule)
    assert sim["attempts"] == 6 and sim["late_starts"] == 1 and sim["passed"] == 1 and sim["gave_up"] == 5
    assert sim["pass_rate"] == pytest.approx(1 / 6) and sim["median_days_to_pass"] == 2.0 and sim["mean_months"] == 1.0
    assert sim["mean_paid"] == pytest.approx(500.0 / 6) and sim["mean_payouts"] == pytest.approx(1 / 6)
    # per attempt: 90% of $500 / 6 = $75, minus one month at $50, minus $100 activation x 1/6 = $8.33
    assert ts.net_per_attempt(sim, rule) == pytest.approx(75.0 - 50.0 - 100.0 / 6)
    assert ts.simulate(pnl, np.zeros(7), RULES)["pass_rate"] is None  # 7 days: too short for a 60-day attempt


def test_an_attempt_gives_up_after_the_horizon():
    rows = floors([100.0] * 10, rules=replace(RULES, combine_max_days=4))
    assert [r["outcome"] for r in rows] == ["running", "running", "running", "gave up"]
    c = ts.combine(np.full(10, 100.0), np.zeros(10), replace(RULES, combine_max_days=4))
    assert c["state"].tolist() == [3] * 7 + [0] * 3 and c["days"][:7].tolist() == [4] * 7


def test_unset_fees_and_cap_give_a_message_not_a_number():
    unset = ts.Rules(combine_max_days=2)
    sim = ts.simulate(np.array([1_500.0, 1_500.0, 200]), np.zeros(3), unset)
    assert sim["pass_rate"] == 0.5 and sim["mean_paid"] is None and sim["payout_note"] == ts.PAYOUT_CAP_MESSAGE
    assert ts.net_per_attempt(sim, unset) is None and ts.money_note(unset) == "Set the fee in config/topstep.toml"
    assert ts.money_note(replace(unset, combine_monthly=1.0, activation=1.0)) == ts.PAYOUT_CAP_MESSAGE
    assert ts.money_note(RULES) is None
    assert unset.missing_trading_fees() == ["commission_per_side_MES", "commission_per_side_MNQ"]


def test_the_daily_loss_limit_ends_a_day_at_the_limit():
    rule = replace(RULES, daily_loss_limit=500.0)
    rows = floors([300.0], [-600.0], rule)  # dipped $600, so the day ended at -$500
    assert rows[0]["balance"] == 49_500


def test_a_strategy_with_no_edge_rarely_passes():
    # The plan: with a $3,000 target and a $2,000 limit, luck alone passes about 40%
    # (2,000 / 5,000) before the trailing floor cuts that down.
    rng = np.random.default_rng(11)
    pnl = rng.normal(0, 250, 5000)
    rule = replace(RULES, consistency_kind="total_profit", best_day_share=1.0)
    sim = ts.simulate(pnl, np.minimum(pnl, 0), rule)
    assert 0.02 < sim["pass_rate"] < 0.4


# ------------------------------------------------------------------ the rules file


def test_the_rules_file_holds_the_plans_values_and_no_fees():
    rules = ts.load_rules(REPO_ROOT / "config" / "topstep.toml")
    assert (rules.account_size, rules.profit_target, rules.max_loss_limit) == (50_000, 3_000, 2_000)
    assert rules.daily_loss_limit is None and rules.max_micro_contracts == 50 and rules.flat_by == "15:10"
    assert (rules.consistency_kind, rules.best_day_share) == ("target", 0.5)
    assert (rules.payout_winning_days, rules.payout_min_day_profit, rules.payout_share_of_balance) == (5, 150, 0.5)
    assert rules.profit_split == 0.9 and rules.floor_to_start_after_payout is True
    assert rules.max_payout is None and rules.combine_monthly is None and rules.activation is None
    assert rules.commission_per_side == {"MES": None, "MNQ": None}
    assert rules.max_download_usd == 10.0 and rules.flat_minutes_before_close() == 0
    text = (REPO_ROOT / "config" / "topstep.toml").read_text()
    assert text.count("Check at help.topstep.com") >= 12


def test_a_bad_rules_file_is_refused_by_name(tmp_path: Path):
    good = (REPO_ROOT / "config" / "topstep.toml").read_text()
    p = tmp_path / "topstep.toml"
    p.write_text(good.replace("profit_target = 3000", "profit_target = \"lots\""))
    with pytest.raises(ts.RulesError, match="profit_target must be a number"):
        ts.load_rules(p)
    p.write_text(good.replace('kind = "target"', 'kind = "vibes"'))
    with pytest.raises(ts.RulesError, match="kind"):
        ts.load_rules(p)
    p.write_text(good.replace("# combine_monthly =", "combine_monthly = 49"))
    assert ts.load_rules(p).combine_monthly == 49.0
    with pytest.raises(ts.RulesError, match="missing"):
        ts.load_rules(tmp_path / "nope.toml")
    p.write_text(good.replace('flat_by = "15:10"', 'flat_by = "14:40"'))
    assert ts.load_rules(p).flat_minutes_before_close() == 20


def test_rules_travel_to_workers_and_have_a_fingerprint():
    again = ts.Rules.from_dict(RULES.to_dict())
    assert again == RULES and again.sim_key() == RULES.sim_key()
    assert replace(RULES, combine_monthly=99.0).sim_key() == RULES.sim_key()  # fees do not change the simulation
    assert replace(RULES, profit_target=6_000.0).sim_key() != RULES.sim_key()


# ------------------------------------------------------------------ a model through the rules


def test_size_limit_keeps_the_worst_stretch_under_half_the_loss_limit():
    assert ts.size_limit(-300.0, RULES) == (3, False)   # 3 x $300 = $900 <= $1,000; 4 x $300 > $1,000
    assert ts.size_limit(-1_200.0, RULES) == (1, True)  # even one contract uses more than half
    assert ts.size_limit(0.0, RULES) == (10, False)


def test_evaluate_reports_every_size_with_its_coin_flip_twin():
    data = wiggly(days=20)
    model = scripted(every_20_minutes)
    small = replace(RULES, twin_seeds=3, account_size=50_000, profit_target=300.0, max_loss_limit=400.0,
                    consistency_kind="total_profit", best_day_share=1.0, payout_min_day_profit=10.0, express_days=5,
                    combine_max_days=5)
    out = ts.evaluate(data, model, None, small, 5, None, max_size=2)
    assert [s["contracts"] for s in out["sizes"]] == [1, 2]
    assert out["numbers"]["days"] == 15 and out["sim_key"] == small.sim_key() and out["feed"] == "test"
    for s in out["sizes"]:
        assert set(s["sim"]) == set(s["twin"]) and s["sim"]["attempts"] == 11 and s["twin"]["pass_rate"] is not None
    assert out["double_slippage_pnl"] < out["numbers"]["net_pnl"]
    best = ts.best_size(out, small)
    assert best["contracts"] in (1, 2) and best["net"] == max(ts.net_per_attempt(s["sim"], small) for s in out["sizes"])
    assert ts.best_size(out, replace(small, activation=None)) is None
    with pytest.raises(ValueError, match="Set the fee"):
        ts.evaluate(data, model, None, ts.Rules(), 5, None, max_size=1)
