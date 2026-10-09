"""Recipes (fleet2/models/futures/recipe.py): models put together from building blocks.

The cut-off test runs on many random recipes and on every signal, filter and exit at
least once, so every block is proved never to use a later price, whoever picks the
blocks (a random mix or Claude Haiku)."""
from __future__ import annotations

import random
from datetime import date

import numpy as np
import pytest

from coordinator.futures_data import FakeFuturesSource
from fleet2.models.futures import module_for, recipe as R, tries_bucket
from fleet2.models.futures.base import check_module, draw_params, mutate_params, neighbours, params_with_defaults
from fleet2.sim import futures_backtest as fb
from fleet2.sim import futures_data as wfd


@pytest.fixture(scope="module")
def data() -> wfd.FuturesData:
    """About four months of synthetic minutes, with overnight gaps and a contract roll."""
    src = FakeFuturesSource(seed=5, first=date(2024, 1, 2), last=date(2024, 5, 10))
    return wfd.build("synthetic", {s: src.series(s) for s in ("MES", "MNQ")})


def every_block() -> list[dict]:
    """Each signal with each exit, every filter, every direction, entry style and side."""
    out = []
    filters = sorted(R.FILTERS)
    for i, signal in enumerate(sorted(R.SIGNALS)):
        for j, exit_name in enumerate(sorted(R.EXITS)):
            k = i * len(R.EXITS) + j
            out.append(R.validate({
                "signal": signal, "exit": exit_name, "filters": [filters[k % len(filters)]],
                "direction": R.DIRECTIONS[k % 2], "entries": R.ENTRIES[(k // 2) % 2], "side": R.SIDES[k % 3]}))
    rng = random.Random("recipes:cutoff")
    out += [R.random_recipe(rng) for _ in range(15)]
    return out


RECIPES = every_block()


# ------------------------------------------------------------------ checking and naming


def test_a_recipe_has_one_canonical_form_and_name():
    a = {"signal": "gap", "direction": "fade", "filters": ["vwap_side", "quiet_day"], "exit": "vwap_return",
         "entries": "first", "side": "both"}
    b = {**a, "filters": ["quiet_day", "vwap_side", "quiet_day"]}
    assert R.validate(b)["filters"] == ["quiet_day", "vwap_side"]
    assert R.name_of(a) == R.name_of(b) and R.name_of(a).startswith("recipe_") and len(R.name_of(a)) == 17
    assert R.name_of({**a, "direction": "follow"}) != R.name_of(a)


@pytest.mark.parametrize("bad, words", [
    ("not a dict", "must be an object"),
    ({"signal": "rsi"}, "signal must be one of"),
    ({"signal": "gap", "direction": "follow", "filters": ["moon_phase"], "exit": "hold", "entries": "first",
      "side": "both"}, "unknown filters: moon_phase"),
    ({"signal": "gap", "direction": "follow", "filters": ["quiet_day", "busy_day"], "exit": "hold",
      "entries": "first", "side": "both"}, "contradict"),
    ({"signal": "gap", "direction": "follow", "filters": ["quiet_day", "vwap_side", "after_gap"], "exit": "hold",
      "entries": "first", "side": "both"}, "at most 2 filters"),
    ({"signal": "gap", "direction": "follow", "filters": [], "exit": "hold", "entries": "first", "side": "both",
      "code": "import os"}, "unknown recipe fields: code"),
])
def test_anything_but_known_blocks_is_refused(bad, words):
    with pytest.raises(R.BadRecipe, match=words):
        R.validate(bad)


def test_random_recipes_are_valid_varied_and_repeatable():
    made = [R.random_recipe(random.Random(f"seed:{i}")) for i in range(200)]
    assert all(R.validate(r) == r for r in made)
    assert len({R.name_of(r) for r in made}) > 150
    assert {r["signal"] for r in made} == set(R.SIGNALS) and {r["exit"] for r in made} == set(R.EXITS)
    assert R.random_recipe(random.Random("seed:7")) == made[7]


def test_a_recipe_is_found_by_its_name_and_never_runs_another():
    r = RECIPES[0]
    name = R.name_of(r)
    module = module_for(name, {"recipe": r})
    assert module.__name__ == name and module is R.family(dict(r))  # built once, then reused
    with pytest.raises(KeyError, match="hold no recipe"):
        module_for(name, {})
    with pytest.raises(KeyError, match="not " + name):
        module_for(name, {"recipe": RECIPES[1]})
    assert module_for("gap_fade").NAME == "Gap fade"
    assert tries_bucket(name) == "recipe" and tries_bucket("gap_fade") == "gap_fade"


@pytest.mark.parametrize("r", RECIPES[:12], ids=lambda r: R.name_of(r))
def test_a_recipe_works_like_a_model_file(r):
    module = R.family(r)
    check_module(module)
    assert module.DESCRIPTION.startswith("A recipe model: ") and module.HOW_IT_WORKS.count(".") >= 3
    for key, spec in module.SEARCH_SPACE.items():  # the defaults are settings the search could draw
        value = module.DEFAULT_PARAMS[key]
        assert (value in spec[0]) if spec[-1] == "choice" else (spec[0] <= value <= spec[1]), key
    rng = random.Random("a recipe's settings")
    drawn = draw_params(module, rng)
    assert drawn["recipe"] == r  # the recipe travels with every set of settings
    assert mutate_params(module, drawn, rng)["recipe"] == r
    assert all(n["recipe"] == r for n in neighbours(module, drawn))


# ------------------------------------------------------------------ the cut-off test


def answers(data: wfd.FuturesData, module, params: dict) -> tuple[wfd.Bars, np.ndarray]:
    bars, wanted = fb.model_targets(data, module, params)
    return bars, wanted[params["symbol"]]


def test_every_block_is_covered():
    assert {r["signal"] for r in RECIPES} == set(R.SIGNALS)
    assert {r["exit"] for r in RECIPES} == set(R.EXITS)
    assert {f for r in RECIPES for f in r["filters"]} == set(R.FILTERS)
    assert {r["direction"] for r in RECIPES} == set(R.DIRECTIONS) and {r["side"] for r in RECIPES} == set(R.SIDES)


@pytest.mark.parametrize("r", RECIPES, ids=lambda r: R.name_of(r))
def test_recipe_answers_never_change_when_later_prices_are_cut_off(data, r):
    module = R.family(r)
    rng = np.random.default_rng(int(R.name_of(r)[-6:], 16))
    settings = [params_with_defaults(module, None)]
    draw = random.Random(f"cutoff:{R.name_of(r)}")
    settings += [draw_params(module, draw) for _ in range(2)]
    settings.append({**settings[0], "bar_minutes": 1})
    for params in settings:
        bars, full = answers(data, module, params)
        cuts = list(rng.integers(1, data.n_minutes, 4))
        cuts += [int(data.day_start[rng.integers(1, data.n_days)])]          # at a day's open
        cuts += [int(data.day_end[rng.integers(0, data.n_days)]) - 1]        # one minute before a close
        for t in cuts:
            cut_bars, cut = answers(data.until_minute(int(t)), module, params)
            closed = int(np.count_nonzero(cut_bars.complete))
            assert np.array_equal(cut[:closed], full[:closed]), (
                f"{module.NAME} {params}: an answer before minute {t} changed when later prices were cut off")


def test_most_recipes_trade(data):
    """Recipes that never want a position would prove nothing in the cut-off test."""
    costs = fb.FuturesCosts(1.0, {"MES": 0.5, "MNQ": 0.5})
    trading = sum(fb.run(data, R.family(r), None, costs, first_day=30).n_trades > 0 for r in RECIPES)
    assert trading >= 0.6 * len(RECIPES), f"only {trading} of {len(RECIPES)} recipes traded"


def test_a_fair_value_gap_retest_is_found_from_past_bars_only():
    """An upward gap forms at bar 3 (its low 104 is above bar 1's high 101); bar 5 dips
    back into the latest gap and closes inside it: that is the signal. A gap the price
    has already gone through no longer counts."""
    from types import SimpleNamespace

    n = 8
    bars = SimpleNamespace(n=n, first=np.r_[True, np.zeros(n - 1, bool)])
    s = SimpleNamespace(high=np.array([100, 101, 101.5, 106, 107, 106.5, 108, 109.]),
                        low=np.array([99, 100, 101.0, 104, 105, 103.0, 106, 107.]),
                        close=np.array([100, 101, 101.5, 105, 106, 104.5, 107, 108.]))
    up, down = R.fair_value_gaps(s, bars, np.zeros(n), 10)
    assert up.tolist() == [False] * 5 + [True, False, False] and not down.any()
    big = R.fair_value_gaps(s, bars, np.full(n, 10.0), 10)[0]  # gaps under 10 points are ignored
    assert not big.any()
    gone = SimpleNamespace(high=s.high, low=np.array([99, 100, 101.0, 104, 105, 100.0, 103, 107.]), close=s.close)
    assert not R.fair_value_gaps(gone, bars, np.zeros(n), 10)[0][6]  # bar 5 went through the gap: bar 6 is no retest


# ------------------------------------------------------------------ smart money, past-only


def toy(high, low, close, open_=None, first=None):
    from types import SimpleNamespace

    high, low, close = (np.array(x, dtype=float) for x in (high, low, close))
    n = len(high)
    bars = SimpleNamespace(n=n, first=np.r_[True, np.zeros(n - 1, bool)] if first is None else np.array(first))
    s = SimpleNamespace(high=high, low=low, close=close,
                        open=close.copy() if open_ is None else np.array(open_, dtype=float))
    return s, bars


def test_a_swing_only_counts_once_the_bars_after_it_have_closed():
    """Bar 2 is a swing high (above the 2 bars on each side); it is known at bar 4, not before."""
    s, bars = toy(high=[10, 11, 15, 12, 11, 10, 9], low=[9, 10, 13, 11, 10, 9, 8], close=[9.5, 10.5, 14, 11.5, 10.5, 9.5, 8.5])
    high_level, _, new_high, _ = R.confirmed_swings(s, bars, 2)
    assert new_high.tolist() == [False] * 4 + [True, False, False]
    assert np.isnan(high_level[:4]).all() and (high_level[4:] == 15).all()


def test_breaks_of_structure_and_a_change_of_character():
    """A swing low at bar 2 (known at bar 3) is broken by bar 4's close. A swing high at
    bar 3 (known at bar 4) is broken by bar 5's close: against the last break, a change
    of character. A swing high at bar 5 (known at bar 6) is broken by bar 8's close: the
    same way as the last break. Bar 9 closes above it again: no second break."""
    high = [12, 11, 10, 11, 9.5, 12, 11, 11.5, 13, 14]
    low = [11, 10, 9, 10, 8.0, 11, 10, 10.5, 12, 13]
    close = [11.5, 10.5, 9.5, 10.5, 8.5, 11.5, 10.5, 11, 12.8, 13.5]
    s, bars = toy(high, low, close)
    up, down, turned = R.structure_breaks(s, bars, 1)
    assert down.tolist() == [False] * 4 + [True] + [False] * 5
    assert up.tolist() == [False] * 5 + [True, False, False, True, False]
    assert turned.tolist() == [False] * 5 + [True] + [False] * 4


def test_an_order_block_retest_waits_for_the_break_and_the_pullback():
    """Bar 3 falls (close below open) before bar 5 breaks above the swing high of bar 2;
    bar 7 dips into bar 3's range (10..11.5) and closes inside: an upward retest."""
    high = [10, 11, 12, 11.5, 11, 13, 13.5, 12.5, 14]
    low = [9, 10, 11, 10.0, 10, 12, 12.5, 11.0, 13]
    close = [9.5, 10.5, 11.5, 10.2, 10.5, 12.8, 13, 11.8, 13.8]
    open_ = [9.2, 10.2, 11.2, 11.2, 10.3, 11.0, 12.8, 13.0, 12.0]
    s, bars = toy(high, low, close, open_)
    up, down = R.order_block_retests(s, bars, 1, 10)
    assert up.tolist() == [False] * 7 + [True, False] and not down.any()


def test_a_sweep_takes_yesterdays_low_and_closes_back_above_it():
    """Day 1's low is 99; on day 2, bar 4 trades to 98 and closes at 100: an upward sweep."""
    s, bars = toy(high=[101, 102, 101.5, 101, 100.5, 101], low=[99, 100, 100, 99.5, 98, 99.5],
                  close=[100, 101, 101, 100, 100, 100.5], first=[True, False, False, True, False, False])
    s.instrument = np.zeros(6, dtype=np.int64)
    up, down = R.liquidity_sweeps(s, bars)
    assert up.tolist() == [False] * 4 + [True, False] and not down.any()


@pytest.mark.parametrize("setup", ["order_block", "choch", "sweep"])
def test_every_smart_money_setup_trades_and_never_peeks(data, setup):
    from fleet2.models.futures import smart_money

    params = {**params_with_defaults(smart_money, None), "setup": setup}
    bars, full = answers(data, smart_money, params)
    assert np.count_nonzero(full) > 0
    rng = np.random.default_rng(len(setup))
    for t in [*rng.integers(1, data.n_minutes, 5), int(data.day_end[rng.integers(0, data.n_days)]) - 1]:
        cut_bars, cut = answers(data.until_minute(int(t)), smart_money, params)
        closed = int(np.count_nonzero(cut_bars.complete))
        assert np.array_equal(cut[:closed], full[:closed]), f"{setup}: an answer before minute {t} changed"
