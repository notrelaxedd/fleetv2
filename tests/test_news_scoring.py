"""News prediction scoring: entry strictly after the decision, real costs, the coin-flip
twin on the very same bars, and the gate that keeps a news strategy off paper money
until its predictions beat that twin."""
from __future__ import annotations

import itertools
import math
import random
from datetime import datetime, timedelta, timezone

import pytest
from psycopg.types.json import Jsonb

from coordinator import news_scoring as ns
from coordinator.limits import Limits

UTC = timezone.utc
LIMITS = Limits()  # short_borrow_fee_pct 3.0, min_confidence 0.6, min_scored_predictions 100
FEE = LIMITS.short_borrow_fee_pct
STOCK_RT = 2 * 5 / 100       # 0.10 %
CRYPTO_RT = 2 * (10 + 25) / 100  # 0.70 %
LATER = datetime(2026, 12, 1, tzinfo=UTC)  # long after every bar in these tests


def day(d: int) -> datetime:
    """Alpaca's daily bar ts for 2026-10-d: midnight New York (EDT) = 04:00 UTC."""
    return datetime(2026, 10, d, 4, tzinfo=UTC)


def bar(conn, symbol, ts, open_, close, tf="1Day") -> None:
    conn.execute(
        "INSERT INTO bars VALUES (%s, %s, %s, %s, %s, %s, %s, 1, 'test')"
        " ON CONFLICT (symbol, timeframe, ts) DO UPDATE SET open = EXCLUDED.open, close = EXCLUDED.close",
        (symbol, tf, ts, open_, max(open_, close), min(open_, close), close),
    )


def days(conn, symbol, rows) -> None:
    """rows: {day_of_october: (open, close)}"""
    for d, (o, c) in rows.items():
        bar(conn, symbol, day(d), o, c)


def hours(conn, symbol, start, n, price=100.0, overrides=None) -> None:
    overrides = overrides or {}
    for i in range(n):
        o, c = overrides.get(i, (price, price))
        bar(conn, symbol, start + timedelta(hours=i), o, c, tf="1Hour")


@pytest.fixture
def analysis(conn) -> dict:
    conn.execute(
        "INSERT INTO news (id, headline, symbols, ours, created_at) VALUES (1, 'h', %s, %s, %s)",
        (["AAPL"], ["AAPL"], day(5)),
    )
    row = conn.execute(
        "INSERT INTO news_analyses (news_id, source, requested_model, served_model, prompt_version, context)"
        " VALUES (1, 'live', 'claude-opus-5-5', 'claude-opus-5-5', 'v1', %s) RETURNING id",
        (Jsonb({}),),
    ).fetchone()
    return {"news_id": 1, "analysis_id": row["id"]}


@pytest.fixture
def predict(conn, analysis):
    def make(symbol="AAPL", direction="up", decided_at=None, horizon=1, confidence=0.7, source="live",
             market=None, model="claude-opus-5-5", **scores) -> int:
        market = market or ("crypto" if "/" in symbol else "stocks")
        cols = {
            "analysis_id": analysis["analysis_id"], "news_id": analysis["news_id"], "source": source,
            "model": model, "symbol": symbol, "market": market, "direction": direction,
            "expected_move_pct": 1.0, "horizon_days": horizon, "confidence": confidence,
            "already_priced_in": False, "decided_at": decided_at or datetime(2026, 10, 6, 15, tzinfo=UTC),
            **scores,
        }
        names = ", ".join(cols)
        marks = ", ".join(["%s"] * len(cols))
        return conn.execute(f"INSERT INTO news_predictions ({names}) VALUES ({marks}) RETURNING id",
                            list(cols.values())).fetchone()["id"]
    return make


def get(conn, pid) -> dict:
    return conn.execute("SELECT * FROM news_predictions WHERE id = %s", (pid,)).fetchone()


def twin_sign(pid) -> float:
    return 1.0 if ns.twin_direction(pid) == "up" else -1.0


def week(conn, symbol="AAPL", spy=True) -> None:
    """Tue 6 .. Fri 9, Mon 12 Oct: each bar opens at 100 + day and closes 1 higher."""
    rows = {d: (100.0 + d, 101.0 + d) for d in (6, 7, 8, 9, 12)}
    days(conn, symbol, rows)
    if spy:
        days(conn, "SPY", {d: (500.0, 500.0) for d in rows})


# ------------------------------------------------------------------ entry and exit

@pytest.mark.parametrize("decided_at", [
    datetime(2026, 10, 6, 15, tzinfo=UTC),  # 11:00 Tuesday New York, during the session
    datetime(2026, 10, 7, 3, tzinfo=UTC),   # 23:00 Tuesday New York, Wednesday in UTC
])
def test_daily_entry_is_the_next_bar_to_start_after_the_decision(conn, predict, decided_at):
    week(conn)
    pid = predict(decided_at=decided_at)
    assert ns.score_pending(conn, LIMITS, LATER) == 1
    p = get(conn, pid)
    assert p["entry_ts"] == day(7) and p["entry_price"] == 107.0   # Wednesday's open
    assert p["exit_ts"] == day(7) and p["exit_price"] == 108.0     # horizon 1: same bar's close


def test_a_decision_after_midnight_new_york_waits_for_the_next_session(conn, predict):
    week(conn)
    pid = predict(decided_at=datetime(2026, 10, 7, 5, tzinfo=UTC))  # 01:00 Wednesday NY: Wednesday's bar has started
    ns.score_pending(conn, LIMITS, LATER)
    assert get(conn, pid)["entry_ts"] == day(8)


def test_exit_counts_bars_not_calendar_days(conn, predict):
    week(conn)
    pid = predict(decided_at=datetime(2026, 10, 8, 15, tzinfo=UTC), horizon=2)  # Thursday
    ns.score_pending(conn, LIMITS, LATER)
    p = get(conn, pid)
    assert (p["entry_ts"], p["exit_ts"]) == (day(9), day(12))  # Friday open to Monday close
    assert p["move_pct"] == pytest.approx((113.0 / 109.0 - 1) * 100)


def test_hourly_crypto_entry_is_strictly_after_the_decision(conn, predict):
    t0 = datetime(2026, 10, 6, 0, tzinfo=UTC)
    for sym in ("ETH/USD", "BTC/USD"):
        hours(conn, sym, t0, 80, overrides={i: (100.0 + i, 100.5 + i) for i in range(80)})
    a = predict("ETH/USD", decided_at=t0 + timedelta(hours=10, minutes=30))
    b = predict("ETH/USD", decided_at=t0 + timedelta(hours=11))  # exactly at a bar start: that bar is too late
    c = predict("ETH/USD", decided_at=t0 + timedelta(hours=10, minutes=30), horizon=2)
    assert ns.score_pending(conn, LIMITS, LATER) == 3
    pa, pb, pc = get(conn, a), get(conn, b), get(conn, c)
    assert pa["entry_ts"] == t0 + timedelta(hours=11) and pa["entry_price"] == 111.0
    assert pa["exit_ts"] == t0 + timedelta(hours=34) and pa["exit_price"] == 134.5  # 24 bars, entry is bar 1
    assert pb["entry_ts"] == t0 + timedelta(hours=12)
    assert pc["exit_ts"] == t0 + timedelta(hours=11 + 47)


def test_exit_not_available_yet_stays_unscored(conn, predict):
    days(conn, "AAPL", {7: (100.0, 101.0)})
    days(conn, "SPY", {7: (500.0, 500.0)})
    pid = predict(horizon=2)  # needs Wednesday and Thursday
    assert ns.score_pending(conn, LIMITS, LATER) == 0
    p = get(conn, pid)
    assert p["scored_at"] is None and p["entry_ts"] is None
    days(conn, "AAPL", {8: (102.0, 103.0)})
    days(conn, "SPY", {8: (500.0, 500.0)})
    # Thursday's bar is stored but not finished at 15:00 Thursday: still nothing.
    assert ns.score_pending(conn, LIMITS, datetime(2026, 10, 8, 15, tzinfo=UTC)) == 0
    assert ns.score_pending(conn, LIMITS, datetime(2026, 10, 9, 4, tzinfo=UTC)) == 1
    assert get(conn, pid)["exit_price"] == 103.0


def test_missing_bars_for_long_get_a_note_but_stay_unscored(conn, predict):
    days(conn, "AAPL", {7: (100.0, 101.0)})
    pid = predict(horizon=3)
    ns.score_pending(conn, LIMITS, datetime(2026, 10, 12, tzinfo=UTC))  # within 2 x 3 + 4 days
    assert get(conn, pid)["score_note"] is None
    ns.score_pending(conn, LIMITS, LATER)
    p = get(conn, pid)
    assert p["score_note"] == ns.OVERDUE_NOTE and p["scored_at"] is None
    days(conn, "AAPL", {8: (101.0, 102.0), 9: (102.0, 104.0)})
    days(conn, "SPY", {d: (500.0, 500.0) for d in (7, 8, 9)})
    assert ns.score_pending(conn, LIMITS, LATER) == 1
    p = get(conn, pid)
    assert p["scored_at"] == LATER and p["score_note"] is None


def test_none_direction_is_never_scored(conn, predict):
    week(conn)
    pid = predict(direction="none")
    assert ns.score_pending(conn, LIMITS, LATER) == 0
    assert get(conn, pid)["scored_at"] is None and get(conn, pid)["score_note"] is None


# ------------------------------------------------------------------ costs

def test_stock_costs_up_and_down_with_borrow_fee(conn, predict):
    days(conn, "AAPL", {7: (100.0, 102.0), 8: (99.0, 98.0)})
    days(conn, "SPY", {7: (500.0, 500.0), 8: (500.0, 500.0)})
    up = predict(direction="up")                    # Wed 100 -> Wed 102: +2 %
    down = predict(direction="down")                # same bars, a short: +2 % is a loss
    down2 = predict(direction="down", horizon=2)    # Wed 100 -> Thu 98: -2 %, two days of borrow
    ns.score_pending(conn, LIMITS, LATER)
    pu, pd, pd2 = get(conn, up), get(conn, down), get(conn, down2)
    assert pu["move_pct"] == pytest.approx(2.0)
    assert pu["signed_pct"] == pytest.approx(2.0 - 0.10)
    assert pd["signed_pct"] == pytest.approx(-2.0 - 0.10 - 3.0 * 1 / 365)
    assert pd2["move_pct"] == pytest.approx(-2.0)
    assert pd2["signed_pct"] == pytest.approx(2.0 - 0.10 - 3.0 * 2 / 365)  # 1.8835616...
    assert pd2["signed_pct"] == pytest.approx(1.883561643835616)


def test_borrow_fee_counts_calendar_days_over_a_weekend(conn, predict):
    week(conn)
    pid = predict(direction="down", decided_at=datetime(2026, 10, 8, 15, tzinfo=UTC), horizon=2)
    ns.score_pending(conn, LIMITS, LATER)
    p = get(conn, pid)  # Friday open 109 to Monday close 113: four calendar days
    assert p["signed_pct"] == pytest.approx(-(113 / 109 - 1) * 100 - STOCK_RT - FEE * 4 / 365)
    assert ns.days_held(day(9), day(12)) == 4
    # A daylight-saving change (bar ts 04:00 vs 05:00 UTC) does not change the count.
    assert ns.days_held(datetime(2026, 10, 30, 4, tzinfo=UTC), datetime(2026, 11, 2, 5, tzinfo=UTC)) == 4


def test_borrow_fee_follows_the_limit(conn, predict):
    days(conn, "AAPL", {7: (100.0, 98.0)})
    days(conn, "SPY", {7: (500.0, 500.0)})
    pid = predict(direction="down")
    ns.score_pending(conn, Limits(short_borrow_fee_pct=36.5), LATER)
    assert get(conn, pid)["signed_pct"] == pytest.approx(2.0 - 0.10 - 0.10)  # 36.5 % a year = 0.1 % a day


def test_crypto_costs_and_down_has_no_borrow_fee(conn, predict):
    t0 = datetime(2026, 10, 6, 0, tzinfo=UTC)
    hours(conn, "BTC/USD", t0, 30)
    hours(conn, "ETH/USD", t0, 30, overrides={1: (100.0, 100.0), 24: (103.0, 103.0)})
    up = predict("ETH/USD", "up", decided_at=t0 + timedelta(minutes=5))
    down = predict("ETH/USD", "down", decided_at=t0 + timedelta(minutes=5))
    ns.score_pending(conn, LIMITS, LATER)
    pu, pd = get(conn, up), get(conn, down)
    assert pu["move_pct"] == pytest.approx(3.0) and pu["bench_pct"] == pytest.approx(0.0)
    assert pu["signed_pct"] == pytest.approx(3.0 - 0.70)
    assert pd["signed_pct"] == pytest.approx(-3.0 - 0.70)
    assert ns.cost_pct("crypto", "down", 10, LIMITS) == pytest.approx(CRYPTO_RT)
    assert ns.cost_pct("stocks", "up", 10, LIMITS) == pytest.approx(STOCK_RT)


# ------------------------------------------------------------------ the coin-flip twin

def test_twin_is_a_seeded_coin_flip():
    expect = ["up" if random.Random(f"twin:{i}").random() < 0.5 else "down" for i in range(1, 2001)]
    assert [ns.twin_direction(i) for i in range(1, 2001)] == expect
    assert [ns.twin_direction(i) for i in range(1, 2001)] == expect  # same every time
    ups = expect.count("up")
    assert 900 < ups < 1100


def test_twin_uses_the_same_bars_and_the_same_cost_rules(conn, predict):
    week(conn)
    ids = [predict(direction=d, decided_at=datetime(2026, 10, 6, 15, tzinfo=UTC), horizon=2)
           for d in ("up", "down") * 6]
    ns.score_pending(conn, LIMITS, LATER)
    seen = set()
    for pid in ids:
        p = get(conn, pid)
        seen.add(p["twin_direction"])
        assert p["twin_direction"] == ns.twin_direction(pid)
        move = p["move_pct"]
        assert move == pytest.approx((109 / 107 - 1) * 100)  # Wed open to Thu close
        expect = twin_sign(pid) * move - ns.cost_pct("stocks", p["twin_direction"], 2, LIMITS)
        assert p["twin_signed_pct"] == pytest.approx(expect)
        if p["twin_direction"] == p["direction"]:
            assert p["twin_signed_pct"] == pytest.approx(p["signed_pct"])
    assert seen == {"up", "down"}


def test_rescoring_gives_the_same_twin_and_score(conn, predict):
    week(conn)
    pid = predict(horizon=2)
    ns.score_pending(conn, LIMITS, LATER)
    first = get(conn, pid)
    conn.execute("UPDATE news_predictions SET scored_at = NULL, twin_direction = NULL, twin_signed_pct = NULL")
    ns.score_pending(conn, LIMITS, LATER)
    again = get(conn, pid)
    for col in ("entry_ts", "exit_ts", "move_pct", "signed_pct", "twin_direction", "twin_signed_pct", "hit"):
        assert again[col] == first[col]


def test_bars_outside_the_holding_window_do_not_change_the_score(conn, predict):
    """The lookahead proof: scramble the bar running at decision time and every bar after
    the exit; the score must not move."""
    week(conn)
    pid = predict(decided_at=datetime(2026, 10, 6, 15, tzinfo=UTC), horizon=2)  # Wed..Thu
    ns.score_pending(conn, LIMITS, LATER)
    first = get(conn, pid)
    conn.execute("UPDATE news_predictions SET scored_at = NULL")
    for sym in ("AAPL", "SPY"):
        for d in (5, 6, 9, 12):
            bar(conn, sym, day(d), 1.0 + d * 37, 999.0 - d)
    ns.score_pending(conn, LIMITS, LATER)
    again = get(conn, pid)
    for col in ("entry_price", "exit_price", "move_pct", "bench_pct", "signed_pct", "twin_signed_pct", "hit"):
        assert again[col] == first[col]


# ------------------------------------------------------------------ hit and benchmark

@pytest.mark.parametrize("direction,stock,spy,hit", [
    ("up", (100, 102), (500, 515), False),    # up 2 %, market up 3 %: made money, still a miss
    ("up", (100, 102), (500, 505), True),     # up 2 %, market up 1 %
    ("down", (100, 99), (500, 485), False),   # down 1 %, market down 3 %: fell less than the market
    ("down", (100, 99), (500, 505), True),
])
def test_hit_is_against_the_benchmark(conn, predict, direction, stock, spy, hit):
    days(conn, "AAPL", {7: stock})
    days(conn, "SPY", {7: spy})
    pid = predict(direction=direction)
    ns.score_pending(conn, LIMITS, LATER)
    p = get(conn, pid)
    assert p["bench_pct"] == pytest.approx((spy[1] / spy[0] - 1) * 100)
    assert p["hit"] is hit


def test_the_benchmark_itself_is_judged_on_its_own_move(conn, predict):
    days(conn, "SPY", {7: (500.0, 505.0)})
    pid = predict("SPY", "up")
    ns.score_pending(conn, LIMITS, LATER)
    p = get(conn, pid)
    assert p["bench_pct"] == pytest.approx(p["move_pct"]) and p["hit"] is True


def test_missing_benchmark_bar_uses_the_nearest_earlier_one(conn, predict):
    days(conn, "AAPL", {7: (100.0, 101.0), 8: (101.0, 102.0)})
    days(conn, "SPY", {6: (400.0, 410.0), 8: (500.0, 520.0)})  # no SPY bar on Wednesday
    pid = predict(horizon=2)
    ns.score_pending(conn, LIMITS, LATER)
    assert get(conn, pid)["bench_pct"] == pytest.approx((520 / 400 - 1) * 100)  # Tuesday open, Thursday close


def test_benchmark_not_refreshed_up_to_the_exit_waits(conn, predict):
    days(conn, "AAPL", {7: (100.0, 101.0), 8: (101.0, 102.0)})
    days(conn, "SPY", {7: (500.0, 505.0)})  # SPY stops before the exit bar
    pid = predict(horizon=2)
    assert ns.score_pending(conn, LIMITS, LATER) == 0
    days(conn, "SPY", {8: (505.0, 510.0)})
    assert ns.score_pending(conn, LIMITS, LATER) == 1
    assert get(conn, pid)["bench_pct"] == pytest.approx(2.0)


def test_benchmark_far_too_old_is_not_used(conn, predict):
    days(conn, "AAPL", {7: (100.0, 101.0)})
    bar(conn, "SPY", day(7) - timedelta(days=30), 400.0, 400.0)
    bar(conn, "SPY", day(12), 500.0, 500.0)
    predict()
    assert ns.score_pending(conn, LIMITS, LATER) == 0


# ------------------------------------------------------------------ idempotence and batching

def test_scoring_is_idempotent_and_frozen(conn, predict):
    week(conn)
    pid = predict()
    assert ns.score_pending(conn, LIMITS, LATER) == 1
    first = get(conn, pid)
    assert ns.score_pending(conn, LIMITS, LATER + timedelta(days=1)) == 0
    days(conn, "AAPL", {7: (1.0, 2.0)})  # a later re-download changes the bar
    ns.score_pending(conn, LIMITS, LATER + timedelta(days=2))
    assert get(conn, pid) == first


def test_score_pending_is_bounded(conn, predict):
    week(conn)
    for _ in range(7):
        predict()
    assert ns.score_pending(conn, LIMITS, LATER, batch=3) == 3
    assert ns.score_pending(conn, LIMITS, LATER, batch=3) == 3
    assert ns.score_pending(conn, LIMITS, LATER, batch=3) == 1
    assert ns.score_pending(conn, LIMITS, LATER, batch=3) == 0


def test_unscorable_rows_do_not_block_the_batch(conn, predict):
    week(conn)
    for _ in range(5):
        predict("MSFT", decided_at=datetime(2026, 10, 1, tzinfo=UTC))  # no MSFT bars at all
    pid = predict()
    assert ns.score_pending(conn, LIMITS, LATER, batch=2) == 1
    assert get(conn, pid)["scored_at"] is not None


# ------------------------------------------------------------------ stats

KEYS = {"n", "predictions", "pending", "none_count", "fallback_excluded", "hit_rate", "mean_signed_pct",
        "mean_twin_pct", "mean_relative_pct", "mean_twin_relative_pct", "edge_pct", "edge_se_pct", "beats_twin",
        "calibration", "by_direction"}
_ENTRY = itertools.count()


def scored(predict, move, twin="down", direction="up", bench=0.0, cost=STOCK_RT, hit=None, confidence=0.7,
           entry=None, **kw) -> int:
    """A scored prediction whose numbers agree with each other: the call and its twin pay
    `cost`, hit is the move against the benchmark. Each gets its own entry day unless
    `entry` is given."""
    sign, tsign = (1.0 if direction == "up" else -1.0), (1.0 if twin == "up" else -1.0)
    if hit is None:
        hit = sign * (move - bench) > 0
    if entry is None:
        entry = datetime(2026, 1, 1, 5, tzinfo=UTC) + timedelta(days=next(_ENTRY))
    return predict(direction=direction, confidence=confidence, signed_pct=sign * move - cost,
                   twin_signed_pct=tsign * move - cost, hit=hit, twin_direction=twin, move_pct=move,
                   bench_pct=bench, entry_ts=entry, scored_at=LATER, **kw)


def test_stats_with_nothing_scored(conn, predict):
    predict()                     # pending
    predict(direction="down")     # pending
    predict(direction="none")
    s = ns.stats(conn)
    assert set(s) == KEYS
    assert (s["n"], s["predictions"], s["pending"], s["none_count"], s["fallback_excluded"]) == (0, 0, 2, 1, 0)
    for key in ("hit_rate", "mean_signed_pct", "mean_twin_pct", "mean_relative_pct", "mean_twin_relative_pct",
                "edge_pct", "edge_se_pct"):
        assert s[key] is None
    assert s["beats_twin"] is False
    assert [b["label"] for b in s["calibration"]] == ["0–0.5", "0.5–0.6", "0.6–0.7", "0.7–0.8", "0.8–1.0"]
    assert all(b == {"label": b["label"], "n": 0, "hit_rate": None} for b in s["calibration"])
    assert s["by_direction"] == {"up": {"n": 0, "hit_rate": None, "mean_signed_pct": None},
                                 "down": {"n": 0, "hit_rate": None, "mean_signed_pct": None}}
    assert set(ns.stats(conn, trading=LIMITS)) == KEYS


def test_one_scored_prediction_has_no_standard_error(conn, predict):
    scored(predict, 2.0)  # up 2 %, twin down: +1.9 against -2.1
    s = ns.stats(conn)
    assert s["n"] == 1 and s["edge_pct"] == pytest.approx(4.0)
    assert s["edge_se_pct"] is None and s["beats_twin"] is False


def test_beats_twin_math(conn, predict):
    # Paired differences (direction - twin) x (move - bench) = 1, 2, 3, 4: mean 2.5,
    # sd sqrt(5/3) = 1.2910, se = 0.6455.
    scored(predict, 0.5, hit=True)
    scored(predict, -1.0, twin="up", direction="down", hit=True)
    scored(predict, 1.5, hit=False)
    scored(predict, 2.0, hit=True)
    s = ns.stats(conn)
    assert s["n"] == s["predictions"] == 4 and s["pending"] == 0
    assert s["hit_rate"] == pytest.approx(0.75)
    assert s["mean_signed_pct"] == pytest.approx((0.4 + 0.9 + 1.4 + 1.9) / 4)
    assert s["mean_twin_pct"] == pytest.approx((-0.6 - 1.1 - 1.6 - 2.1) / 4)
    assert s["mean_relative_pct"] == pytest.approx(s["mean_signed_pct"])  # the benchmark did not move
    assert s["edge_pct"] == pytest.approx(2.5)
    assert s["edge_pct"] == pytest.approx(s["mean_relative_pct"] - s["mean_twin_relative_pct"])
    assert s["edge_se_pct"] == pytest.approx(math.sqrt(5 / 3) / 2)
    assert s["beats_twin"] is True
    assert s["by_direction"]["up"] == {"n": 3, "hit_rate": pytest.approx(2 / 3),
                                       "mean_signed_pct": pytest.approx(3.7 / 3)}
    assert s["by_direction"]["down"] == {"n": 1, "hit_rate": 1.0, "mean_signed_pct": pytest.approx(0.9)}


def test_noisy_edge_does_not_beat_twin(conn, predict):
    # Differences 3, -1, 2, -2: mean 0.5, sd sqrt(17/3) = 2.3805, se 1.1902 -> 0.5 < 2.38.
    for move in (1.5, -0.5, 1.0, -1.0):
        scored(predict, move)
    s = ns.stats(conn)
    assert s["edge_pct"] == pytest.approx(0.5)
    assert s["edge_se_pct"] == pytest.approx(math.sqrt(17 / 3) / 2)
    assert s["beats_twin"] is False


def test_the_edge_is_measured_against_the_market(conn, predict):
    """Up 3 % while the market is up 3 %: money made, but no skill shown."""
    scored(predict, 3.0, bench=3.0)
    scored(predict, 2.0, bench=2.0, twin="up")
    s = ns.stats(conn)
    assert s["mean_signed_pct"] == pytest.approx(2.5 - STOCK_RT)
    assert s["mean_relative_pct"] == pytest.approx(-STOCK_RT)
    assert s["edge_pct"] == pytest.approx(0.0) and s["beats_twin"] is False
    assert s["hit_rate"] == 0.0


def test_calibration_buckets(conn, predict):
    for conf, hit in ((0.49, True), (0.5, True), (0.59, False), (0.6, True), (0.7, False), (0.79, False),
                      (0.8, True), (1.0, False)):
        scored(predict, 1.0, hit=hit, confidence=conf)
    cal = ns.stats(conn)["calibration"]
    assert [(b["n"], b["hit_rate"]) for b in cal] == [(1, 1.0), (2, 0.5), (1, 1.0), (2, 0.0), (2, 0.5)]


def test_stats_filters(conn, predict):
    scored(predict, 1.0, cost=0.0, confidence=0.9)
    scored(predict, 2.0, cost=0.0, confidence=0.5, source="replay")
    scored(predict, 3.0, cost=0.0, symbol="BTC/USD", model="other")
    predict(confidence=0.4)  # pending, low confidence
    assert ns.stats(conn)["n"] == 3
    assert ns.stats(conn, market="stocks")["n"] == 2
    assert ns.stats(conn, market="crypto")["mean_signed_pct"] == 3.0
    assert ns.stats(conn, source="replay")["mean_signed_pct"] == 2.0
    assert ns.stats(conn, model="other")["n"] == 1
    s = ns.stats(conn, min_confidence=0.6)
    assert (s["n"], s["pending"]) == (2, 0)
    assert ns.stats(conn)["pending"] == 1


def test_unscored_rows_never_count_in_stats(conn, predict):
    scored(predict, 1.0, cost=0.0)
    predict(signed_pct=50.0, twin_signed_pct=0.0, hit=True)  # half-written row, scored_at NULL
    s = ns.stats(conn)
    assert s["n"] == 1 and s["mean_signed_pct"] == 1.0 and s["pending"] == 1


def test_stats_after_real_scoring(conn, predict):
    week(conn)
    for d in ("up", "down", "none"):
        predict(direction=d)
    ns.score_pending(conn, LIMITS, LATER)
    s = ns.stats(conn, market="stocks")
    assert (s["n"], s["pending"], s["none_count"]) == (2, 0, 1)
    assert s["mean_signed_pct"] == pytest.approx((-2 * STOCK_RT - FEE / 365) / 2)  # up and down cancel


# ------------------------------------------------------------------ gate

GATE = Limits(min_scored_predictions=4, min_confidence=0.6)


def test_gate_needs_enough_predictions(conn, predict):
    for _ in range(3):
        scored(predict, 1.0)
    scored(predict, 1.0, confidence=0.55)   # below min_confidence: would not trade
    scored(predict, 1.0, symbol="BTC/USD")  # other market
    predict()                               # unscored
    assert ns.gate(conn, "stocks", GATE) == (False, "Needs 4 independent scored calls it would trade; has 3.")
    assert ns.gate(conn, "stocks", Limits()) == (False, "Needs 100 independent scored calls it would trade; has 3.")


def test_gate_does_not_beat_the_twin(conn, predict):
    for move in (1.5, -0.5, 1.0, -1.0):
        scored(predict, move)
    ok, reason = ns.gate(conn, "stocks", GATE)
    assert not ok
    assert reason == "Does not beat the coin-flip twin against the market yet: edge +0.50% ± 1.19% over 4 independent calls."


def test_gate_beats_twin_but_loses_money(conn, predict):
    """A falling market: every call beat the market (skill), but the book still lost money."""
    for move, bench in ((-2.0, -3.0), (-2.2, -3.1), (-1.8, -3.0), (-2.0, -3.2)):
        scored(predict, move, bench=bench)
    ok, reason = ns.gate(conn, "stocks", GATE)
    assert not ok
    assert reason == ("Beats the coin-flip twin but loses money after costs: "
                      "-2.10% a call on average over 4 independent calls.")


def test_gate_passes_counting_both_sources(conn, predict):
    for move, source in ((0.5, "live"), (1.0, "replay"), (1.5, "live"), (2.0, "replay")):
        scored(predict, move, source=source)
    assert ns.gate(conn, "stocks", GATE) == (
        True, "Ready: 4 independent calls, edge +2.50% ± 0.65% over the coin flip, against the market.")
    assert ns.gate(conn, "crypto", GATE) == (False, "Needs 4 independent scored calls it would trade; has 0.")


def test_gate_unknown_market(conn):
    with pytest.raises(ValueError):
        ns.gate(conn, "futures", GATE)


# ------------------------------------------------------------------ (a) only what would be traded

WINNERS = 6  # strong, steady winners: the gate would open on them if they counted


@pytest.mark.parametrize("market,kind,kw", [
    ("stocks", "already priced in", {"already_priced_in": True}),
    ("stocks", "too small a move to beat the costs", {"expected_move_pct": 0.15}),  # needs 2 x 0.10
    ("crypto", "too small a move to beat the costs", {"symbol": "ETH/USD", "expected_move_pct": 1.3}),  # 2 x 0.70
    ("stocks", "not confident enough", {"confidence": 0.59}),
    ("stocks", "SPY and QQQ are left to the futures paper books", {"symbol": "SPY"}),
    ("stocks", "SPY and QQQ are left to the futures paper books", {"symbol": "QQQ"}),
    ("crypto", "Alpaca cannot short coins", {"symbol": "ETH/USD", "direction": "down", "twin": "up"}),
])
def test_gate_counts_only_the_calls_it_would_trade(conn, predict, market, kind, kw):
    kw = {"direction": "up", "expected_move_pct": 2.0, **kw}
    sign = 1.0 if kw["direction"] == "up" else -1.0
    for i in range(WINNERS):
        scored(predict, sign * (1.0 + 0.1 * i), cost=0.0, **kw)
    p = get(conn, conn.execute("SELECT max(id) AS id FROM news_predictions").fetchone()["id"])
    assert ns.skip_reason(p, market, GATE) == kind
    assert ns.stats(conn, market=market)["beats_twin"] is True   # every call: looks great
    assert ns.gate(conn, market, GATE) == (False, "Needs 4 independent scored calls it would trade; has 0.")
    tradeable = {**kw, "symbol": "ETH/USD" if market == "crypto" else "AAPL", "direction": "up", "twin": "down",
                 "confidence": 0.7, "already_priced_in": False, "expected_move_pct": 2.0}
    for i in range(WINNERS):
        scored(predict, 1.0 + 0.1 * i, cost=0.0, **tradeable)
    assert ns.gate(conn, market, GATE)[0] is True
    assert ns.stats(conn, market=market, trading=GATE)["n"] == WINNERS


def test_the_trader_and_the_gate_share_one_predicate():
    """news_trading's entry filter is news_scoring.skip_reason itself: same answer on every case."""
    from coordinator import news_trading
    for market, symbols in (("stocks", ("AAPL", "SPY", "QQQ")), ("crypto", ("ETH/USD", "BTC/USD"))):
        for symbol in symbols:
            for direction in ("up", "down", "none"):
                for confidence in (0.5, 0.6, 0.9):
                    for priced_in in (False, True):
                        for move in (0.05, 0.15, 0.25, 1.3, 1.5, 5.0):
                            p = {"symbol": symbol, "direction": direction, "confidence": confidence,
                                 "already_priced_in": priced_in, "expected_move_pct": move}
                            ours = ns.skip_reason(p, market, LIMITS)
                            assert news_trading._skip_reason(p, market, LIMITS) == ours
                            assert ns.would_trade(p, market, LIMITS) is (ours is None)
    assert ns.round_trip_cost_pct("stocks") == pytest.approx(news_trading.round_trip_cost_pct("stocks"))
    assert ns.round_trip_cost_pct("crypto") == pytest.approx(news_trading.round_trip_cost_pct("crypto"))


# ------------------------------------------------------------------ (b) skill, not market drift

def rising_market(conn, start, n_days, symbols=("AAPL", "SPY")) -> None:
    """A market that rises every day; the stocks move exactly with it (no stock-specific move)."""
    rng = random.Random(1)
    for i in range(n_days):
        r = 1.0 + rng.uniform(0.0, 0.03)
        for sym in symbols:
            bar(conn, sym, start + timedelta(days=i), 100.0, 100.0 * r)


def test_always_up_in_a_rising_market_has_no_edge(conn, predict):
    """Every call is 'up' and tradeable, the market rises every day and the stock only moves
    with it. The calls make money and beat the coin flip on absolute returns (the twin's
    'down' half loses the drift and pays the borrow fee), but there is no skill: the
    gate stays shut."""
    start = datetime(2026, 8, 3, 4, tzinfo=UTC)
    rising_market(conn, start, 40)
    for i in range(30):
        predict(decided_at=start + timedelta(days=i, hours=-6), expected_move_pct=2.0)
    assert ns.score_pending(conn, LIMITS, LATER) == 30
    s = ns.stats(conn, market="stocks", trading=GATE)
    assert s["n"] == 30 and s["hit_rate"] == 0.0
    assert s["mean_signed_pct"] > 1.0                              # the book would have made money...
    assert s["mean_signed_pct"] - s["mean_twin_pct"] > 0.5         # ...and beaten the twin on raw returns
    assert s["edge_pct"] == pytest.approx(0.0, abs=1e-9)           # but not against the market
    ok, reason = ns.gate(conn, "stocks", GATE)
    assert not ok and reason.startswith("Does not beat the coin-flip twin against the market yet: edge +0.00%")


def test_never_shorting_does_not_earn_an_edge_from_the_borrow_fee(conn, predict):
    """With no move at all, a 'down' twin pays a borrow fee the 'up' call does not; that
    difference is not skill, so the twin is charged the call's own costs."""
    for _ in range(6):
        pid = predict(direction="up", signed_pct=-STOCK_RT, twin_direction="down",
                      twin_signed_pct=-STOCK_RT - FEE / 365, move_pct=0.0, bench_pct=0.0, hit=False,
                      scored_at=LATER, entry_ts=datetime(2026, 1, 1, 5, tzinfo=UTC) + timedelta(days=next(_ENTRY)))
    s = ns.stats(conn, trading=GATE)
    assert s["edge_pct"] == pytest.approx(0.0) and s["beats_twin"] is False
    assert get(conn, pid)["signed_pct"] > get(conn, pid)["twin_signed_pct"]


# ------------------------------------------------------------------ (c) duplicate calls

def test_the_same_call_on_the_same_bar_counts_once(conn, predict):
    entry = day(7)
    for move in (1.0, 1.0, 1.0, 3.0):  # four articles, same AAPL up call; one held two days
        scored(predict, move, entry=entry)
    scored(predict, -1.0, direction="down", twin="up", entry=entry)  # the opposite call is another one
    scored(predict, 1.0, symbol="MSFT", entry=entry)                 # so is another symbol
    s = ns.stats(conn, market="stocks", trading=GATE)
    assert (s["n"], s["predictions"]) == (3, 6)
    assert s["by_direction"]["up"]["n"] == 2 and s["by_direction"]["down"]["n"] == 1
    assert s["edge_pct"] == pytest.approx((2 * 1.5 + 2.0 + 2.0) / 3)  # the four AAPL calls average to one
    assert ns.stats(conn, market="stocks")["n"] == 6                  # the cards still count every call


def test_coins_count_once_per_entry_day(conn, predict):
    t0 = datetime(2026, 10, 6, 0, tzinfo=UTC)
    for hour in (1, 5, 23):  # three calls entering at different hours of one UTC day
        scored(predict, 2.0, symbol="ETH/USD", expected_move_pct=2.0, cost=CRYPTO_RT, entry=t0 + timedelta(hours=hour))
    scored(predict, 2.0, symbol="ETH/USD", expected_move_pct=2.0, cost=CRYPTO_RT, entry=t0 + timedelta(hours=25))
    assert ns.stats(conn, market="crypto", trading=GATE)["n"] == 2


def test_duplicates_do_not_shrink_the_standard_error(conn, predict):
    """Four real calls (differences 1, 2, 3, 4) each copied by nine articles: the edge and
    its standard error are those of four calls, not 36."""
    for move in (0.5, 1.0, 1.5, 2.0):
        entry = datetime(2026, 1, 1, 5, tzinfo=UTC) + timedelta(days=next(_ENTRY))
        for _ in range(9):
            scored(predict, move, entry=entry)
    s = ns.stats(conn, trading=GATE)
    assert (s["n"], s["predictions"]) == (4, 36)
    assert s["edge_se_pct"] == pytest.approx(math.sqrt(5 / 3) / 2)
    assert ns.gate(conn, "stocks", Limits(min_scored_predictions=5))[1] == (
        "Needs 5 independent scored calls it would trade; has 4.")


# ------------------------------------------------------------------ (d) refusal fallbacks

@pytest.fixture
def fallback(conn, analysis) -> int:
    """A replay analysis that a refusal fallback answered (another model than the one asked)."""
    return conn.execute(
        "INSERT INTO news_analyses (news_id, source, requested_model, served_model, prompt_version, context)"
        " VALUES (1, 'replay', 'claude-opus-5-5', 'claude-later-model', 'v1', %s) RETURNING id",
        (Jsonb({}),),
    ).fetchone()["id"]


def test_replays_answered_by_a_fallback_model_never_count(conn, predict, fallback):
    for move in (0.5, 1.0, 1.5, 2.0):  # would pass the gate on their own
        scored(predict, move, source="replay", analysis_id=fallback, model="claude-later-model")
    predict(source="replay", analysis_id=fallback)  # pending: not counted as waiting either
    s = ns.stats(conn, market="stocks")
    assert (s["n"], s["pending"], s["fallback_excluded"]) == (0, 0, 4)
    assert ns.gate(conn, "stocks", GATE) == (False, "Needs 4 independent scored calls it would trade; has 0.")
    for move in (0.5, 1.0, 1.5, 2.0):  # honest replays (asked and answered by the same model)
        scored(predict, move, source="replay")
    assert ns.gate(conn, "stocks", GATE)[0] is True
    assert ns.stats(conn, market="stocks", trading=GATE)["fallback_excluded"] == 4


def test_a_live_prediction_answered_by_a_fallback_still_counts(conn, predict, fallback):
    """A live prediction is made before the outcome exists, whichever model answered it."""
    conn.execute("UPDATE news_analyses SET source = 'live' WHERE id = %s", (fallback,))
    scored(predict, 1.0, analysis_id=fallback, model="claude-later-model")
    s = ns.stats(conn, market="stocks", trading=GATE)
    assert (s["n"], s["fallback_excluded"]) == (1, 0)


def test_a_dated_snapshot_name_of_the_requested_model_still_counts(conn, predict, fallback):
    """Anthropic may report the model it ran by its dated name; that is the model asked."""
    conn.execute("UPDATE news_analyses SET served_model = 'claude-opus-5-5-20260601' WHERE id = %s", (fallback,))
    scored(predict, 1.0, source="replay", analysis_id=fallback, model="claude-opus-5-5-20260601")
    s = ns.stats(conn, market="stocks", trading=GATE)
    assert (s["n"], s["fallback_excluded"]) == (1, 0)


# ------------------------------------------------------------------ the reviewer's scenarios

def _article_prediction(conn, i, symbol, direction, decided, priced_in=False, expected=0.05):
    news_id = 10_000 + i
    conn.execute("INSERT INTO news (id, headline, symbols, ours, created_at) VALUES (%s, 'h', %s, %s, %s)",
                 (news_id, [symbol], [symbol], decided))
    a = conn.execute("INSERT INTO news_analyses (news_id, source, requested_model, served_model, prompt_version, "
                     "context) VALUES (%s, 'replay', 'm', 'm', 'v', %s) RETURNING id", (news_id, Jsonb({}))).fetchone()
    conn.execute(
        "INSERT INTO news_predictions (analysis_id, news_id, source, model, symbol, market, direction, "
        "expected_move_pct, horizon_days, confidence, already_priced_in, decided_at) VALUES "
        "(%s, %s, 'replay', 'm', %s, 'stocks', %s, %s, 1, 0.7, %s, %s)",
        (a["id"], news_id, symbol, direction, expected, priced_in, decided))


def test_reviewer_untradeable_pure_beta_calls_keep_the_gate_shut(conn):
    """300 'up' calls, all priced in and too small to trade, on a stock that only moves with
    a rising market: this used to open the gate ("edge +1.47% ± 0.11%")."""
    start = datetime(2026, 8, 3, 4, tzinfo=UTC)
    rising_market(conn, start, 400)
    for i in range(300):
        _article_prediction(conn, i, "AAPL", "up", start + timedelta(days=i, hours=-6), priced_in=True)
    assert ns.score_pending(conn, LIMITS, datetime(2027, 12, 1, tzinfo=UTC)) == 300
    assert ns.stats(conn, market="stocks")["hit_rate"] == 0.0
    assert ns.gate(conn, "stocks", LIMITS) == (False, "Needs 100 independent scored calls it would trade; has 0.")


def test_reviewer_nine_articles_a_day_are_twelve_calls_not_108(conn):
    """12 market days, each with nine articles making the same AAPL call (7 right, 5 wrong):
    this used to open the gate on '108 predictions'."""
    start = datetime(2026, 8, 3, 4, tzinfo=UTC)
    outcomes = [+3, +3, -2, +3, +3, -2, +3, -2, +3, -2, +3, -2]
    i = 0
    for d, r in enumerate(outcomes):
        ts = start + timedelta(days=2 * d)
        bar(conn, "AAPL", ts, 100.0, 100.0 + r)
        bar(conn, "SPY", ts, 100.0, 100.0)
        for k in range(9):
            _article_prediction(conn, i, "AAPL", "up", ts - timedelta(hours=6, minutes=k), expected=1.0)
            i += 1
    for sym in ("AAPL", "SPY"):
        bar(conn, sym, start + timedelta(days=40), 100.0, 100.0)
    assert ns.score_pending(conn, LIMITS, datetime(2027, 1, 1, tzinfo=UTC)) == 108
    s = ns.stats(conn, market="stocks", trading=LIMITS)
    assert (s["n"], s["predictions"]) == (12, 108)
    assert ns.gate(conn, "stocks", LIMITS) == (False, "Needs 100 independent scored calls it would trade; has 12.")
    ok, reason = ns.gate(conn, "stocks", Limits(min_scored_predictions=12))
    assert not ok and reason.startswith("Does not beat the coin-flip twin against the market yet")
