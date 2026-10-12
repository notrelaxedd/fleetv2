"""Crypto structure: hold coins after a break of structure up, sit in cash after one down."""
from __future__ import annotations

from typing import Any

import numpy as np

from fleet2.universe import CRYPTO

NAME = "Crypto structure"
MARKET = "crypto"
DESCRIPTION = "Holds a coin after its price closes above its last swing high and sells when it closes below its last swing low."
HOW_IT_WORKS = (
    "About once a day it marks each coin's swing highs and lows on hourly prices, the peaks and dips that stand "
    "out from the bars around them, counting one only once a few later bars have closed and proved it. "
    "When the price has closed above the last swing high the trend has turned up, so it holds that coin. "
    "When it has closed below the last swing low the trend has turned down, so it sells and waits in cash."
)
SYMBOLS = CRYPTO
# Decides about daily on swings two days wide: every crypto fill costs about 35 bp, and on
# zero-edge random-walk prices half-day swings paid ~$2,900 a year in fees on $10,000,
# against ~$1,000 (as much as Crypto trend) with these defaults.
DEFAULT_PARAMS: dict[str, Any] = {"swing_left": 48, "swing_right": 24, "min_break_atr": 0.5, "lookback": 720,
                                  "rebalance_every": 24, "max_coins": 8}
SEARCH_SPACE = {
    "swing_left": (8, 72, "int"),
    "swing_right": (4, 36, "int"),
    "min_break_atr": (0.0, 1.0, "float"),
    "lookback": (240, 1440, "int"),
    "rebalance_every": (12, 72, "int"),
    "max_coins": (3, 8, "int"),
}
# The typical bar size a break is measured against: the average high-low range of the last day.
ATR_BARS = 24


def rebalance_every(params: dict[str, Any]) -> int:
    return max(1, int(params["rebalance_every"]))


def warmup(params: dict[str, Any]) -> int:
    return int(params["lookback"]) + 1


def _carry_forward(arr: np.ndarray) -> np.ndarray:
    """Each NaN replaced by the last value before it along the last axis (left NaN at the start)."""
    idx = np.where(np.isnan(arr), -1, np.arange(arr.shape[-1]))
    np.maximum.accumulate(idx, axis=-1, out=idx)
    return np.where(idx >= 0, np.take_along_axis(arr, np.maximum(idx, 0), axis=-1), np.nan)


def swing_levels(values: np.ndarray, left: int, right: int, kind: str) -> np.ndarray:
    """The last confirmed swing high (kind "high") or low ("low") as known at the close of
    each bar, along the last axis (one row per coin), NaN before the first one.

    Bar j is a swing high when its high is above the `left` bars before it and not below
    the `right` bars after it. That is only known once bar j + right has closed, so the
    level appears at index j + right, never at j (charts that draw it at j repaint)."""
    n = values.shape[-1]
    out = np.full(values.shape, np.nan)
    span = left + right + 1
    if n < span:
        return out
    v = values if kind == "high" else -values
    centre = v[..., left:n - right]
    before = np.maximum.reduce([v[..., left - d:n - right - d] for d in range(1, left + 1)])
    after = np.maximum.reduce([v[..., left + d:n - right + d] for d in range(1, right + 1)])
    is_swing = (centre > before) & (centre >= after)
    out[..., left + right:] = np.where(is_swing, values[..., left:n - right], np.nan)
    return _carry_forward(out)


def structure(high: np.ndarray, low: np.ndarray, close: np.ndarray, left: int, right: int,
              min_break_atr: float) -> tuple[np.ndarray, np.ndarray]:
    """(trend, bar of the last break) for each row: trend is 1 after a break up, -1 after a
    break down, 0 when neither has happened in these bars (bar is -1 then).

    A break up is a close above the swing high known at the end of the previous bar, by
    at least min_break_atr times the average range of the last ATR_BARS bars; a break
    down is the same below the swing low."""
    n = close.shape[-1]
    csum = np.concatenate((np.zeros(close.shape[:-1] + (1,)), np.cumsum(high - low, axis=-1)), axis=-1)
    k = np.arange(1, n + 1)
    atr = (csum[..., k] - csum[..., np.maximum(k - ATR_BARS, 0)]) / np.minimum(k, ATR_BARS)
    margin = min_break_atr * atr
    # the levels a close at bar k is compared with were known at the close of bar k - 1
    pad = np.full(close.shape[:-1] + (1,), np.nan)
    sh = np.concatenate((pad, swing_levels(high, left, right, "high")[..., :-1]), axis=-1)
    sl = np.concatenate((pad, swing_levels(low, left, right, "low")[..., :-1]), axis=-1)
    with np.errstate(invalid="ignore"):
        last_up = _last_true(close > sh + margin)
        last_down = _last_true(close < sl - margin)
    trend = np.sign(last_up - last_down)
    return trend, np.where(trend == 0, -1, np.maximum(last_up, last_down))


def _last_true(mask: np.ndarray) -> np.ndarray:
    """Index of the last True along the last axis, -1 where there is none."""
    n = mask.shape[-1]
    return np.where(mask.any(axis=-1), n - 1 - np.argmax(mask[..., ::-1], axis=-1), -1)


def target_positions(history: Any, params: dict[str, Any]) -> dict[str, float]:
    left, right = int(params["swing_left"]), int(params["swing_right"])
    lookback, slots = int(params["lookback"]), int(params["max_coins"])
    if len(history) < lookback:
        return {}
    close = np.stack([history.close(s, bars=lookback) for s in SYMBOLS])
    usable = np.isnan(close).sum(axis=1) <= lookback // 10
    # a missing bar is a bar with no trades: price unchanged since the last close
    close = _carry_forward(close)
    first = close[np.arange(len(SYMBOLS)), np.argmax(~np.isnan(close), axis=1)]
    close = np.where(np.isnan(close), first[:, None], close)
    high = np.stack([history.high(s, bars=lookback) for s in SYMBOLS])
    low = np.stack([history.low(s, bars=lookback) for s in SYMBOLS])
    high, low = np.where(np.isnan(high), close, high), np.where(np.isnan(low), close, low)
    trend, at = structure(high, low, close, left, right, float(params["min_break_atr"]))
    rising = [(int(at[i]), s) for i, s in enumerate(SYMBOLS) if usable[i] and trend[i] == 1]
    chosen = [s for _, s in sorted(rising, reverse=True)[:slots]]
    return {s: 1.0 / slots for s in chosen}
