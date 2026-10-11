"""News prediction scoring: entry strictly after the decision, real costs, the coin-flip
twin on the very same bars, and the gate that keeps a news strategy off paper money
until its predictions beat that twin."""
from __future__ import annotations

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

KEYS = {"n", "pending", "none_count", "hit_rate", "mean_signed_pct", "mean_twin_pct", "edge_pct",
        "edge_se_pct", "beats_twin", "calibration", "by_direction"}


def scored(predict, signed, twin, hit=True, direction="up", confidence=0.7, **kw) -> int:
    return predict(direction=direction, confidence=confidence, signed_pct=signed, twin_signed_pct=twin, hit=hit,
                   twin_direction="up", move_pct=signed, bench_pct=0.0, scored_at=LATER, **kw)


def test_stats_with_nothing_scored(conn, predict):
    predict()                     # pending
    predict(direction="down")     # pending
    predict(direction="none")
    s = ns.stats(conn)
    assert set(s) == KEYS
    assert (s["n"], s["pending"], s["none_count"]) == (0, 2, 1)
    for key in ("hit_rate", "mean_signed_pct", "mean_twin_pct", "edge_pct", "edge_se_pct"):
        assert s[key] is None
    assert s["beats_twin"] is False
    assert [b["label"] for b in s["calibration"]] == ["0–0.5", "0.5–0.6", "0.6–0.7", "0.7–0.8", "0.8–1.0"]
    assert all(b == {"label": b["label"], "n": 0, "hit_rate": None} for b in s["calibration"])
    assert s["by_direction"] == {"up": {"n": 0, "hit_rate": None, "mean_signed_pct": None},
                                 "down": {"n": 0, "hit_rate": None, "mean_signed_pct": None}}


def test_one_scored_prediction_has_no_standard_error(conn, predict):
    scored(predict, 2.0, -1.0)
    s = ns.stats(conn)
    assert s["n"] == 1 and s["edge_pct"] == pytest.approx(3.0)
    assert s["edge_se_pct"] is None and s["beats_twin"] is False


def test_beats_twin_math(conn, predict):
    # Paired differences 1, 2, 3, 4: mean 2.5, sd sqrt(5/3) = 1.2910, se = 0.6455.
    for signed, twin, hit, d in ((1.0, 0.0, True, "up"), (1.5, -0.5, True, "down"),
                                 (2.0, -1.0, False, "up"), (3.0, -1.0, True, "up")):
        scored(predict, signed, twin, hit, d)
    s = ns.stats(conn)
    assert s["n"] == 4 and s["pending"] == 0
    assert s["hit_rate"] == pytest.approx(0.75)
    assert s["mean_signed_pct"] == pytest.approx(7.5 / 4)
    assert s["mean_twin_pct"] == pytest.approx(-2.5 / 4)
    assert s["edge_pct"] == pytest.approx(2.5)
    assert s["edge_se_pct"] == pytest.approx(math.sqrt(5 / 3) / 2)
    assert s["beats_twin"] is True
    assert s["by_direction"]["up"] == {"n": 3, "hit_rate": pytest.approx(2 / 3), "mean_signed_pct": pytest.approx(2.0)}
    assert s["by_direction"]["down"] == {"n": 1, "hit_rate": 1.0, "mean_signed_pct": 1.5}


def test_noisy_edge_does_not_beat_twin(conn, predict):
    # Differences 3, -1, 2, -2: mean 0.5, sd sqrt(17/3) = 2.3805, se 1.1902 -> 0.5 < 2.38.
    for signed, twin in ((3.0, 0.0), (0.0, 1.0), (2.0, 0.0), (-1.0, 1.0)):
        scored(predict, signed, twin)
    s = ns.stats(conn)
    assert s["edge_pct"] == pytest.approx(0.5)
    assert s["edge_se_pct"] == pytest.approx(math.sqrt(17 / 3) / 2)
    assert s["beats_twin"] is False


def test_calibration_buckets(conn, predict):
    for conf, hit in ((0.49, True), (0.5, True), (0.59, False), (0.6, True), (0.7, False), (0.79, False),
                      (0.8, True), (1.0, False)):
        scored(predict, 1.0, 0.0, hit=hit, confidence=conf)
    cal = ns.stats(conn)["calibration"]
    assert [(b["n"], b["hit_rate"]) for b in cal] == [(1, 1.0), (2, 0.5), (1, 1.0), (2, 0.0), (2, 0.5)]


def test_stats_filters(conn, predict):
    scored(predict, 1.0, 0.0, confidence=0.9)
    scored(predict, 2.0, 0.0, confidence=0.5, source="replay")
    scored(predict, 3.0, 0.0, symbol="BTC/USD", model="other")
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
    scored(predict, 1.0, 0.0)
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
        scored(predict, 1.0, 0.0)
    scored(predict, 1.0, 0.0, confidence=0.55)   # below min_confidence: would not trade
    scored(predict, 1.0, 0.0, symbol="BTC/USD")  # other market
    predict()                                    # unscored
    assert ns.gate(conn, "stocks", GATE) == (False, "Needs 4 scored predictions; has 3.")
    assert ns.gate(conn, "stocks", Limits()) == (False, "Needs 100 scored predictions; has 3.")


def test_gate_does_not_beat_the_twin(conn, predict):
    for signed, twin in ((3.0, 0.0), (0.0, 1.0), (2.0, 0.0), (-1.0, 1.0)):
        scored(predict, signed, twin)
    ok, reason = ns.gate(conn, "stocks", GATE)
    assert not ok
    assert reason == "Does not beat the coin-flip twin yet: edge +0.50% ± 1.19%."


def test_gate_beats_twin_but_loses_money(conn, predict):
    for signed, twin in ((-0.1, -1.1), (-0.2, -1.3), (-0.1, -1.2), (-0.2, -1.2)):
        scored(predict, signed, twin)
    ok, reason = ns.gate(conn, "stocks", GATE)
    assert not ok
    assert reason == "Beats the coin-flip twin but loses money after costs: -0.15% a prediction on average."


def test_gate_passes_counting_both_sources(conn, predict):
    for signed, twin, source in ((1.0, 0.0, "live"), (2.0, 0.0, "replay"), (3.0, 0.0, "live"),
                                 (4.0, 0.0, "replay")):
        scored(predict, signed, twin, source=source)
    assert ns.gate(conn, "stocks", GATE) == (
        True, "Ready: 4 scored predictions, edge +2.50% ± 0.65% over the coin flip.")
    assert ns.gate(conn, "crypto", GATE) == (False, "Needs 4 scored predictions; has 0.")


def test_gate_unknown_market(conn):
    with pytest.raises(ValueError):
        ns.gate(conn, "futures", GATE)
