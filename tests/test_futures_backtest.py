"""The futures backtester, against hand-worked examples: fills at the next 1-minute open
plus slippage and fees, whole contracts long and short, pessimistic stops and targets,
no entries after the cut-off, flat by the close, decision bars of several minutes, the
day records (P&L, worst dip, trades) and the coin-flip twin. Plus the lookahead proof:
a day's result never changes when later prices change."""
from __future__ import annotations

import types
from datetime import date

import numpy as np
import pytest

from fleet2.models.futures.base import BadTargets
from fleet2.sim import cme_session as cme
from fleet2.sim import futures_backtest as fb
from fleet2.sim import futures_data as wfd

MES_PV, TICK = 5.0, 0.25
COSTS = fb.FuturesCosts(slippage_ticks=1.0, fee_per_side={"MES": 0.5, "MNQ": 0.4})
DAY1, DAY2 = date(2024, 3, 11), date(2024, 3, 12)


def minutes(days: list[date], close: np.ndarray | None = None, spread: float = 0.5) -> dict[str, np.ndarray]:
    """Bars for every session minute of `days`: open = previous close, high/low `spread`
    points around the open/close (all on the tick grid)."""
    t = np.concatenate([np.arange(cme.session_minutes(d)) * 60 + cme.session(d)[0] for d in days]).astype(np.int64)
    if close is None:
        close = np.full(t.shape, 100.0)
    open_ = np.r_[close[0], close[:-1]]
    return {"t": t, "o": open_.astype(float), "h": np.maximum(open_, close) + spread,
            "l": np.minimum(open_, close) - spread, "c": close.astype(float), "v": np.ones(t.shape),
            "iid": np.ones(t.shape, dtype=np.int64)}


def data_of(series: dict[str, np.ndarray]) -> wfd.FuturesData:
    return wfd.build("test", {"MES": series})


def scripted(target: np.ndarray | None = None, **defaults) -> types.ModuleType:
    """A stand-in model that answers with a fixed array (per decision bar), or a function."""
    m = types.ModuleType("scripted")
    m.NAME, m.MARKET, m.SYMBOLS = "Scripted", "futures", ("MES",)
    m.DESCRIPTION = m.HOW_IT_WORKS = "test"
    m.DEFAULT_PARAMS = {"symbol": "MES", "bar_minutes": 1, "stop_ticks": 0, "target_ticks": 0, **defaults}
    m.SEARCH_SPACE = {}
    m.plan = target

    def targets(bars, params):
        if callable(m.plan):
            return {"MES": m.plan(bars)}
        out = np.zeros(bars.n)
        out[: min(bars.n, m.plan.shape[0])] = m.plan[: bars.n]
        return {"MES": out}

    m.targets = targets
    return m


def plan(n: int, spans: list[tuple[int, int, float]]) -> np.ndarray:
    out = np.zeros(n)
    for a, b, v in spans:
        out[a:b] = v
    return out


# ------------------------------------------------------------------ fills and costs


def test_a_long_trade_fills_at_the_next_minutes_open_plus_slippage_and_fees():
    close = 100.0 + np.arange(390) * 0.25  # rises one tick a minute
    data = data_of(minutes([DAY1], close))
    model = scripted(plan(390, [(0, 4, 1.0)]))  # long on bars 0..3, flat from bar 4
    run = fb.run(data, model, None, COSTS)
    buy = data.open[0, 1] + TICK          # decided on bar 0, filled at minute 1's open, one tick worse
    sell = data.open[0, 5] - TICK         # flat from bar 4: sold at minute 5's open
    assert buy == 100.25 and sell == 100.75
    expected = MES_PV * (sell - buy) - 2 * 0.5
    assert run.pnl.tolist() == [pytest.approx(expected)]
    assert run.trades.tolist() == [1] and run.fees.tolist() == [1.0] and run.slippage.tolist() == [2 * TICK * MES_PV]
    tl = run.trade_list
    assert tl["side"].tolist() == [1] and tl["reason"].tolist() == ["model"] and tl["pnl"][0] == pytest.approx(expected)
    assert tl["entry_t"][0] == data.times[1] and tl["exit_t"][0] == data.times[5]
    assert run.minutes.tolist() == [4]


def test_a_short_trade_and_whole_contracts():
    close = 100.0 - np.arange(390) * 0.25  # falls one tick a minute
    data = data_of(minutes([DAY1], close))
    run = fb.run(data, scripted(plan(390, [(0, 10, -1.0)])), None, COSTS, contracts=3)
    sell, buy = data.open[0, 1] - TICK, data.open[0, 11] + TICK
    assert run.pnl[0] == pytest.approx(3 * MES_PV * (sell - buy) - 6 * 0.5)
    assert run.trade_list["side"].tolist() == [-1] and run.trade_list["contracts"].tolist() == [3]
    half = fb.run(data, scripted(plan(390, [(0, 10, -0.34)])), None, COSTS, contracts=3)
    assert half.trade_list["contracts"].tolist() == [1]  # 0.34 x 3 = 1.02, rounded to 1 contract


def test_adding_and_taking_off_contracts():
    close = 100.0 + np.arange(390) * 0.25
    data = data_of(minutes([DAY1], close))
    run = fb.run(data, scripted(plan(390, [(0, 2, 0.5), (2, 4, 1.0), (4, 6, 0.5)])), None,
                 fb.FuturesCosts(0.0, {"MES": 0.0}), contracts=2)
    # 1 at open[1]=100.25, 1 more at open[3]=100.75 (average 100.5), 1 off at open[5]=101.25, last at open[7]=101.75
    assert run.pnl[0] == pytest.approx(MES_PV * ((101.25 - 100.5) + (101.75 - 100.5)))
    assert run.trades.tolist() == [1] and run.trade_list["contracts"].tolist() == [2]


def test_a_flip_closes_one_trade_and_opens_the_other():
    data = data_of(minutes([DAY1]))
    run = fb.run(data, scripted(plan(390, [(0, 5, 1.0), (5, 10, -1.0)])), None, COSTS)
    assert run.trades.tolist() == [2] and run.trade_list["side"].tolist() == [1, -1]
    assert run.fees.tolist() == [4 * 0.5]


# ------------------------------------------------------------------ stops and targets


def test_a_stop_closes_the_trade_at_the_stop_price_minus_slippage():
    close = np.full(390, 100.0)
    close[20:] = 98.0  # minute 20 opens at 100 and closes at 98: the low passes the stop on the way
    data = data_of(minutes([DAY1], close, spread=0.0))
    model = scripted(plan(390, [(0, 300, 1.0)]), stop_ticks=4)  # stop 4 ticks (1 point) below the entry
    run = fb.run(data, model, None, COSTS)
    entry = 100.0 + TICK
    stop = entry - 1.0
    assert run.trade_list["reason"].tolist() == ["stop"]
    assert run.pnl[0] == pytest.approx(MES_PV * ((stop - TICK) - entry) - 1.0)
    assert run.trades.tolist() == [1]  # the model still wants long, but it is not re-entered until it changes its mind


def test_a_stop_that_gaps_fills_at_the_open():
    close = np.full(390, 100.0)
    data = data_of(minutes([DAY1], close, spread=0.0))
    o = data.open.copy()
    lo = data.low.copy()
    o[0, 30] = lo[0, 30] = 97.0  # minute 30 opens far below the stop
    gapped = wfd.FuturesData(**{**data.__dict__, "open": o, "low": lo})
    run = fb.run(gapped, scripted(plan(390, [(0, 300, 1.0)]), stop_ticks=4), None, COSTS)
    assert run.trade_list["reason"].tolist() == ["stop"]
    assert run.pnl[0] == pytest.approx(MES_PV * ((97.0 - TICK) - (100.0 + TICK)) - 1.0)


def test_stop_and_target_in_the_same_minute_assume_the_stop():
    data = data_of(minutes([DAY1], spread=0.0))
    hi, lo = data.high.copy(), data.low.copy()
    hi[0, 10], lo[0, 10] = 105.0, 95.0  # one wild minute touches both
    wild = wfd.FuturesData(**{**data.__dict__, "high": hi, "low": lo})
    run = fb.run(wild, scripted(plan(390, [(0, 300, 1.0)]), stop_ticks=8, target_ticks=8), None, COSTS)
    assert run.trade_list["reason"].tolist() == ["stop"] and run.pnl[0] < 0


def test_a_target_needs_the_price_to_trade_through_it_and_pays_no_slippage():
    data = data_of(minutes([DAY1], spread=0.0))
    entry = 100.0 + TICK
    target = entry + 8 * TICK
    hi = data.high.copy()
    hi[0, 10] = target          # touches the target: not enough
    touch = wfd.FuturesData(**{**data.__dict__, "high": hi})
    model = scripted(np.ones(390), target_ticks=8)
    assert fb.run(touch, model, None, COSTS).trade_list["reason"].tolist() == ["close"]
    hi[0, 10] = target + TICK   # trades through
    through = wfd.FuturesData(**{**data.__dict__, "high": hi})
    run = fb.run(through, model, None, COSTS)
    assert run.trade_list["reason"].tolist() == ["target"]
    assert run.pnl[0] == pytest.approx(MES_PV * (target - entry) - 1.0)
    assert run.slippage[0] == pytest.approx(TICK * MES_PV)  # only the entry paid slippage


# ------------------------------------------------------------------ the end of the day


def test_no_new_trades_after_the_cutoff_and_flat_at_the_close():
    data = data_of(minutes([DAY1, DAY2]))
    late = scripted(plan(780, [(379, 389, 1.0)]))  # wants in at 14:49, fills at 14:50: too late
    assert fb.run(data, late, None, COSTS).trades.tolist() == [0, 0]
    early = scripted(plan(780, [(370, 389, 1.0)]))  # in at 14:41, still long at the close
    run = fb.run(data, early, None, COSTS)
    assert run.trades.tolist() == [1, 0] and run.trade_list["reason"].tolist() == ["close"]
    assert run.trade_list["exit_t"][0] == data.times[389]  # the last minute of the day: closed at 15:00
    flip = scripted(plan(780, [(370, 382, 1.0), (382, 389, -1.0)]))  # a flip after the cut-off only closes
    run = fb.run(data, flip, None, COSTS)
    assert run.trades.tolist() == [1, 0] and run.trade_list["exit_t"][0] == data.times[383]


def test_never_holds_overnight():
    data = data_of(minutes([DAY1, DAY2]))
    always = scripted(np.ones(780))
    run = fb.run(data, always, None, COSTS)
    assert run.trades.tolist() == [1, 1]  # closed at day 1's close, opened again on day 2
    assert [cme.trading_day(int(t)) for t in run.trade_list["exit_t"]] == [DAY1, DAY2]


def test_half_day_closes_at_noon():
    half = date(2024, 11, 29)
    data = data_of(minutes([half]))
    run = fb.run(data, scripted(np.ones(210)), None, COSTS)
    assert run.trade_list["exit_t"][0] == cme.session(half)[0] + 209 * 60


def test_decisions_on_five_minute_bars_fill_at_the_next_minute():
    close = 100.0 + np.arange(390) * 0.25
    data = data_of(minutes([DAY1], close))
    run = fb.run(data, scripted(plan(78, [(0, 1, 1.0)]), bar_minutes=5), None, COSTS)
    tl = run.trade_list
    assert tl["entry_t"][0] == data.times[5] and tl["exit_t"][0] == data.times[10]


def test_a_day_without_prices_is_not_traded():
    mes = minutes([DAY1, DAY2])
    mnq = minutes([DAY1])
    data = wfd.build("test", {"MES": mes, "MNQ": mnq})
    model = scripted(np.ones(780))
    model.SYMBOLS = ("MES", "MNQ")
    model.plan = lambda bars: np.ones(bars.n)
    model.targets = lambda bars, params: {"MNQ": np.ones(bars.n)}
    assert fb.run(data, model, None, COSTS).trades.tolist() == [1, 0]


def test_only_the_period_is_traded():
    data = data_of(minutes([DAY1, DAY2, date(2024, 3, 13)]))
    run = fb.run(data, scripted(np.ones(3 * 390)), None, COSTS, first_day=1, last_day=2)
    assert run.days.tolist() == [20240312] and run.trades.tolist() == [1]


# ------------------------------------------------------------------ day records


def test_the_worst_dip_counts_the_open_trade_even_when_the_day_ends_green():
    close = np.full(390, 100.0)
    close[50:60] = 95.0   # a five point drop while long
    close[60:] = 104.0    # then a strong close
    data = data_of(minutes([DAY1], close, spread=0.0))
    run = fb.run(data, scripted(np.ones(390)), None, COSTS)
    entry = 100.0 + TICK
    assert run.pnl[0] == pytest.approx(MES_PV * ((104.0 - TICK) - entry) - 1.0) and run.pnl[0] > 0
    assert run.dip[0] == pytest.approx(MES_PV * (95.0 - entry) - 0.5)  # the entry fee is already paid


def test_summary_numbers():
    run = fb.FuturesRun(days=np.arange(20240101, 20240106), pnl=np.array([100.0, -50.0, 200.0, -300.0, 50.0]),
                        dip=np.array([-20.0, -80.0, 0.0, -350.0, -10.0]), trades=np.array([1, 1, 2, 1, 0]),
                        minutes=np.array([10, 20, 30, 40, 0]), slippage=np.zeros(5), fees=np.zeros(5), contracts=1,
                        feed="test", trade_list=fb._trade_arrays([("MES", 0, 600, 1, 1, 100.0, "model"),
                                                                  ("MES", 0, 1200, -1, 1, -50.0, "stop")]))
    s = fb.summarize(run)
    assert s["net_pnl"] == 0.0 and s["days_traded"] == 4 and s["trades"] == 5
    assert s["best_day"] == 200.0 and s["worst_day"] == -300.0 and s["worst_dip"] == -350.0
    # running total 100, 50, 250, -50: from the 250 high the next day dipped to 250 - 350 = -100
    assert s["worst_stretch"] == pytest.approx(-350.0)
    assert s["win_rate"] == 0.5 and s["profit_factor"] == 2.0 and s["avg_hold_minutes"] == 15.0
    assert s["daily_sharpe"] == pytest.approx(0.0) and s["best_day_share"] is None
    assert fb.daily_sharpe(np.array([1.0, 1.0, 1.0])) is None


def test_targets_are_checked():
    data = data_of(minutes([DAY1]))
    with pytest.raises(BadTargets, match="between -1 and \\+1"):
        fb.run(data, scripted(np.full(390, 1.5)), None, COSTS)
    with pytest.raises(BadTargets, match="finite"):
        fb.run(data, scripted(np.full(390, np.nan)), None, COSTS)
    bad = scripted(np.zeros(390))
    bad.targets = lambda bars, params: {"ES": np.zeros(bars.n)}
    with pytest.raises(BadTargets, match="not one of this model's symbols"):
        fb.run(data, bad, None, COSTS)


# ------------------------------------------------------------------ the coin-flip twin


def wiggly(days: int = 20, seed: int = 4) -> wfd.FuturesData:
    ds = cme.trading_days(date(2024, 1, 2), date(2024, 3, 1))[:days]
    rng = np.random.default_rng(seed)
    n = sum(cme.session_minutes(d) for d in ds)
    close = 5000 + np.round(np.cumsum(rng.normal(0, 1.0, n)) / TICK) * TICK
    return data_of(minutes(ds, close))


def every_20_minutes(bars):
    """Long for 10 bars, flat for 10, short for 10, flat for 10..."""
    phase = (bars.minute // 10) % 4
    return np.select([phase == 0, phase == 2], [1.0, -1.0], 0.0)


def test_the_twin_trades_at_the_same_times_in_random_directions():
    data = wiggly()
    model = scripted(every_20_minutes)
    real = fb.run(data, model, None, COSTS)
    twin = fb.run(data, model, None, COSTS, flip_seed=1)
    assert twin.trade_list["entry_t"].tolist() == real.trade_list["entry_t"].tolist()
    assert twin.trade_list["exit_t"].tolist() == real.trade_list["exit_t"].tolist()
    flipped = twin.trade_list["side"] != real.trade_list["side"]
    assert 0.3 < flipped.mean() < 0.7
    again = fb.run(data, model, None, COSTS, flip_seed=1)
    assert again.pnl.tolist() == twin.pnl.tolist()  # the same seed is the same twin
    assert fb.run(data, model, None, COSTS, flip_seed=2).pnl.tolist() != twin.pnl.tolist()


# ------------------------------------------------------------------ no lookahead


def momentum(bars):
    """A causal toy: long when the last close is above the close 10 bars ago."""
    c = bars.series("MES").close
    past = np.r_[np.full(10, np.nan), c[:-10]]
    return np.where(np.isnan(past), 0.0, np.sign(c - past))


def scramble_after(data: wfd.FuturesData, minute: int, seed: int = 9) -> wfd.FuturesData:
    rng = np.random.default_rng(seed)
    out = {}
    for name in ("open", "high", "low", "close"):
        arr = getattr(data, name).copy()
        arr[:, minute:] = np.round(rng.uniform(4000, 6000, arr[:, minute:].shape) / TICK) * TICK
        out[name] = arr
    out["high"] = np.maximum.reduce([out["high"], out["open"], out["close"]])
    out["low"] = np.minimum.reduce([out["low"], out["open"], out["close"]])
    return wfd.FuturesData(**{**data.__dict__, **out})


def closed_before(run: fb.FuturesRun, t: int) -> list[tuple]:
    tl = run.trade_list
    return [(int(a), int(b), int(s), round(float(p), 6)) for a, b, s, p in
            zip(tl["entry_t"], tl["exit_t"], tl["side"], tl["pnl"]) if b < t]


def test_results_never_depend_on_later_prices():
    data = wiggly()
    model = scripted(momentum, stop_ticks=12, target_ticks=20)
    cut_day = 12
    cut = int(data.day_start[cut_day]) + 150  # part way through day 12
    a = fb.run(data, model, None, COSTS)
    b = fb.run(scramble_after(data, cut), model, None, COSTS)
    assert a.pnl[:cut_day].tolist() == b.pnl[:cut_day].tolist()
    assert a.dip[:cut_day].tolist() == b.dip[:cut_day].tolist()
    before = closed_before(a, int(data.times[cut]))
    assert before == closed_before(b, int(data.times[cut])) and len(before) > 50
    assert a.pnl[cut_day:].tolist() != b.pnl[cut_day:].tolist()


def test_the_check_catches_a_model_that_peeks():
    def peeking(bars):
        c = bars.series("MES").close
        return np.sign(np.r_[c[5:], np.repeat(c[-1], 5)] - c)  # the close five bars ahead: the future

    data = wiggly()
    model = scripted(peeking)
    cut = int(data.day_start[12]) + 150
    a = fb.run(data, model, None, COSTS)
    b = fb.run(scramble_after(data, cut), model, None, COSTS)
    assert closed_before(a, int(data.times[cut])) != closed_before(b, int(data.times[cut]))
