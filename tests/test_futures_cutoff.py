"""The cut-off test every futures model must pass: for random cut-off points t, the
targets computed on prices that end at t are exactly the targets computed on all the
prices, for every decision bar that had closed by t. A model that used one later price
anywhere would fail it (the last test shows the check catches one).

Also the model files' plain-English text and search settings."""
from __future__ import annotations

import random
import re
from datetime import date

import numpy as np
import pytest

from coordinator.futures_data import FakeFuturesSource
from fleet2.models import get_module
from fleet2.models.futures import REGISTRY
from fleet2.models.futures.base import check_module, draw_params, params_with_defaults
from fleet2.sim import futures_backtest as fb
from fleet2.sim import futures_data as wfd

FILES = sorted(REGISTRY)


@pytest.fixture(scope="module")
def data() -> wfd.FuturesData:
    """About four months of synthetic MES and MNQ minutes, with overnight gaps and a
    contract roll at the quarter."""
    src = FakeFuturesSource(seed=5, first=date(2024, 1, 2), last=date(2024, 5, 10))
    return wfd.build("synthetic", {s: src.series(s) for s in ("MES", "MNQ")})


def settings(name: str, count: int = 6) -> list[dict]:
    module = REGISTRY[name]
    rng = random.Random(f"cutoff:{name}")
    out = [params_with_defaults(module, None)]
    out += [draw_params(module, rng) for _ in range(count)]
    for size in (1, 3, 5, 15):  # every bar size at least once
        out.append({**out[0], "bar_minutes": size})
    return out


def answers(data: wfd.FuturesData, name: str, params: dict) -> tuple[wfd.Bars, np.ndarray]:
    bars, wanted = fb.model_targets(data, REGISTRY[name], params)
    return bars, wanted[params["symbol"]]


@pytest.mark.parametrize("name", FILES)
def test_answers_never_change_when_later_prices_are_cut_off(data, name):
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    traded = 0
    for params in settings(name):
        bars, full = answers(data, name, params)
        traded += int(np.count_nonzero(full) > 0)
        cuts = list(rng.integers(1, data.n_minutes, 6))
        cuts += [int(data.day_start[rng.integers(1, data.n_days)]) for _ in range(2)]  # at a day's open
        cuts += [int(data.day_end[rng.integers(0, data.n_days)]) - 1]  # one minute before a close
        for t in cuts:
            cut_bars, cut = answers(data.until_minute(int(t)), name, params)
            closed = int(np.count_nonzero(cut_bars.complete))
            assert cut_bars.complete[:closed].all()
            assert np.array_equal(cut[:closed], full[:closed]), (
                f"{name} {params}: an answer before minute {t} changed when the prices after it were cut off")
    assert traded >= 3, f"{name} hardly ever wanted a position, so the test proved little"


def test_the_cutoff_check_catches_a_model_that_peeks(data):
    """A model that uses the whole day's range (only known at the close) fails."""
    from fleet2.models.futures import features as f

    def peeking(bars, params):
        s = bars.series("MES")
        starts = np.flatnonzero(bars.first)
        full_day = (np.maximum.reduceat(s.high, starts) - np.minimum.reduceat(s.low, starts))[f.day_number(bars)]
        return {"MES": np.where(full_day > np.median(full_day), 1.0, -1.0)}

    bars = wfd.resample(data, 5)
    full = peeking(bars, {})["MES"]
    t = int(data.day_start[30]) + 100
    cut_bars = wfd.resample(data.until_minute(t), 5)
    cut = peeking(cut_bars, {})["MES"]
    closed = int(np.count_nonzero(cut_bars.complete))
    assert not np.array_equal(cut[:closed], full[:closed])


@pytest.mark.parametrize("name", FILES)
def test_each_model_trades_in_a_backtest(data, name):
    costs = fb.FuturesCosts(1.0, {"MES": 0.5, "MNQ": 0.5})
    run = fb.run(data, REGISTRY[name], None, costs, first_day=30)
    assert run.n_trades > 0, f"{name} made no trade with its default settings"


# ------------------------------------------------------------------ the model files


def sentences(text: str) -> int:
    return len(re.findall(r"[.!?](\s|$)", text.strip()))


@pytest.mark.parametrize("name", FILES)
def test_model_files_follow_the_interface_and_the_house_style(name):
    module = REGISTRY[name]
    check_module(module)
    assert get_module(name) is module  # the shared lookup finds futures files too
    assert sentences(module.DESCRIPTION) == 1 and len(module.DESCRIPTION) < 140
    assert sentences(module.HOW_IT_WORKS) == 3
    space = module.SEARCH_SPACE
    assert space["bar_minutes"] == ((1, 3, 5, 15), "choice") and space["symbol"] == (("MES", "MNQ"), "choice")
    assert space["stop_ticks"][-1] == "int" and space["target_ticks"][-1] == "int"
    for key, spec in space.items():  # the defaults are settings the search could have drawn
        value = module.DEFAULT_PARAMS[key]
        assert (value in spec[0]) if spec[-1] == "choice" else (spec[0] <= value <= spec[1]), key
