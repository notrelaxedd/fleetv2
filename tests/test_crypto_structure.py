"""The crypto structure model: swing levels appear only once confirmed, and breaks of
structure up and down turn holding a coin on and off."""
from __future__ import annotations

import numpy as np

from fleet2.models import crypto_structure as cs
from fleet2.models.base import clean_targets, params_with_defaults
from fleet2.sim.marketdata import History, MarketData
from fleet2.universe import CRYPTO


def structure(c, min_break=0.0):
    """(trend, bar of last break) for one coin whose highs and lows are 0.5 either side of its closes."""
    trend, at = cs.structure(c + 0.5, c - 0.5, c, 3, 2, min_break)
    return int(trend), int(at)


def bars_from_closes(closes: list[float]) -> MarketData:
    """Every coin gets the same closes; each bar's high and low are 0.5 either side."""
    c = np.tile(np.asarray(closes, dtype=float), (len(CRYPTO), 1))
    times = np.arange(c.shape[1], dtype=np.int64) * 3600 + 1_600_000_000
    return MarketData("crypto", "1Hour", "test", CRYPTO, times, c, c + 0.5, c - 0.5, c, np.full(c.shape, 1e6))


PARAMS = {"swing_left": 3, "swing_right": 2, "min_break_atr": 0.0, "lookback": 30, "rebalance_every": 1, "max_coins": 8}


def test_swing_high_is_known_only_after_right_bars_closed():
    high = np.array([1, 2, 3, 4, 9, 5, 4, 3, 2, 1], dtype=float)  # peak at bar 4
    levels = cs.swing_levels(high, left=3, right=2, kind="high")
    assert np.isnan(levels[:6]).all()  # bars 5 and 6 must close before the peak counts
    assert (levels[6:] == 9).all()


def test_swing_low_mirrors_swing_high():
    low = np.array([9, 8, 7, 6, 1, 5, 6, 7, 8, 9], dtype=float)
    levels = cs.swing_levels(low, left=3, right=2, kind="low")
    assert np.isnan(levels[:6]).all() and (levels[6:] == 1).all()


def test_break_up_then_break_down():
    # a swing high at 12 (bar 4), a swing low at 8 (bar 8), then a close at 13 breaks up
    closes = [10, 10.5, 11, 11.5, 12, 11, 10, 9, 8, 9, 10, 11, 13]
    c = np.asarray(closes, dtype=float)
    assert structure(c) == (1, 12)
    c2 = np.append(c, [12, 10, 7])  # then a close at 7 breaks below the swing low at 7.5
    assert structure(c2) == (-1, 15)


def test_no_swings_means_no_trend():
    c = np.linspace(10, 20, 30)  # a straight line has no swing highs or lows
    assert structure(c) == (0, -1)


def test_min_break_size_filters_a_small_poke_above_the_swing_high():
    closes = [10, 10.5, 11, 11.5, 12, 11, 10, 9, 8, 9, 10, 11, 12.6]  # 0.1 above the 12.5 swing high
    c = np.asarray(closes, dtype=float)
    assert structure(c)[0] == 1
    assert structure(c, 0.5)[0] == 0  # needs half a bar's range (0.5)


def test_holds_after_break_up_and_goes_to_cash_after_break_down():
    up = [10] * 18 + [10, 10.5, 11, 11.5, 12, 11, 10, 9, 8, 9, 10, 11, 13]
    data = bars_from_closes(up)
    targets = clean_targets(cs.target_positions(History(data, data.n_bars), PARAMS), cs.SYMBOLS)
    assert targets == {s: 1 / 8 for s in CRYPTO}
    down = bars_from_closes(up + [12, 10, 7])
    assert cs.target_positions(History(down, down.n_bars), PARAMS) == {}


def test_at_most_max_coins_and_freshest_break_first():
    up = [10] * 18 + [10, 10.5, 11, 11.5, 12, 11, 10, 9, 8, 9, 10, 11, 13, 13.5]
    data = bars_from_closes(up)
    close = data.close.copy()
    close[2, -1] = 12.4  # coin 2 closes back under the 12.5 swing high on the last bar: its break is older
    data = MarketData(data.market, data.timeframe, data.feed, data.symbols, data.times, data.open,
                      np.maximum(data.high, close + 0.5), np.minimum(data.low, close - 0.5), close, data.volume)
    targets = cs.target_positions(History(data, data.n_bars), {**PARAMS, "max_coins": 3})
    assert len(targets) == 3 and CRYPTO[2] not in targets
    assert all(w == 1 / 3 for w in targets.values())


def test_skips_a_coin_with_too_many_missing_bars_and_fills_a_few():
    up = [10] * 18 + [10, 10.5, 11, 11.5, 12, 11, 10, 9, 8, 9, 10, 11, 13]
    data = bars_from_closes(up)
    arrays = {k: getattr(data, k).copy() for k in ("open", "high", "low", "close", "volume")}
    for arr in arrays.values():
        arr[0, :10] = np.nan  # first coin: a third of the window missing
        arr[1, 5] = np.nan  # second coin: one missing bar
    data = MarketData(data.market, data.timeframe, data.feed, data.symbols, data.times, **arrays)
    targets = cs.target_positions(History(data, data.n_bars), PARAMS)
    assert CRYPTO[0] not in targets and CRYPTO[1] in targets


def test_rows_are_coins_and_match_one_coin_at_a_time():
    rng = np.random.default_rng(4)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (5, 400)), axis=1))
    h, lo = c * (1 + rng.uniform(0, 0.01, c.shape)), c * (1 - rng.uniform(0, 0.01, c.shape))
    trend, at = cs.structure(h, lo, c, 4, 3, 0.2)
    for i in range(5):
        one = cs.structure(h[i], lo[i], c[i], 4, 3, 0.2)
        assert (int(trend[i]), int(at[i])) == (int(one[0]), int(one[1]))
        levels = cs.swing_levels(h[i], 4, 3, "high")
        np.testing.assert_array_equal(levels, cs.swing_levels(h, 4, 3, "high")[i])


def test_defaults_and_search_space_agree():
    params = params_with_defaults(cs, None)
    for name, (low, high, _) in cs.SEARCH_SPACE.items():
        assert low <= params[name] <= high
    assert cs.warmup(params) == params["lookback"] + 1
