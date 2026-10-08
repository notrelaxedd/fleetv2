"""Dip buying: buy a strong stock after a sharp short drop, sell when it bounces back."""
from __future__ import annotations

from typing import Any

import numpy as np

from fleet2.universe import STOCKS

NAME = "Dip buyer"
MARKET = "stocks"
DESCRIPTION = "Buys strong stocks after a sudden short drop and sells them once the price bounces back."
HOW_IT_WORKS = (
    "It only looks at stocks that trade above their long-run average price, so the bigger trend is up. "
    "When one of them falls well below its average of the last couple of weeks, it buys, expecting a bounce. "
    "It sells when the price climbs back to that average, or after a set number of days if it never does."
)
SYMBOLS = tuple(s for s in STOCKS if s not in ("SPY", "QQQ"))
DEFAULT_PARAMS: dict[str, Any] = {"short_avg": 10, "dip_pct": 4.0, "trend_avg": 150, "max_hold": 10, "max_positions": 10}
SEARCH_SPACE = {
    "short_avg": (5, 20, "int"),
    "dip_pct": (2.0, 10.0, "float"),
    "trend_avg": (50, 200, "int"),
    "max_hold": (3, 20, "int"),
    "max_positions": (5, 15, "int"),
}


def rebalance_every(params: dict[str, Any]) -> int:
    return 1


def warmup(params: dict[str, Any]) -> int:
    return max(int(params["trend_avg"]), int(params["short_avg"]) + int(params["max_hold"])) + 1


def _rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    """mean of x[j-n+1 .. j] for every j >= n-1 (NaN before)."""
    out = np.full(x.shape, np.nan)
    if x.shape[0] >= n:
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def holding_since(close: np.ndarray, short: int, dip: float, max_hold: int) -> int | None:
    """Bars since the open dip trade began, or None. A trade begins on a close more than
    `dip` below its short average and ends on a close back at or above that average, or
    once it has lasted max_hold bars. Rebuilt from the history alone, so the model needs
    no memory between decisions."""
    avg = _rolling_mean(close, short)
    tail = min(max_hold, close.shape[0] - short)
    entry = None
    for j in range(close.shape[0] - tail, close.shape[0]):
        if np.isnan(close[j]) or np.isnan(avg[j]):
            continue
        if entry is None:
            if close[j] < avg[j] * (1.0 - dip):
                entry = j
        elif close[j] >= avg[j] or j - entry >= max_hold:
            entry = None
    return None if entry is None else close.shape[0] - 1 - entry


def target_positions(history: Any, params: dict[str, Any]) -> dict[str, float]:
    short, trend, max_hold = int(params["short_avg"]), int(params["trend_avg"]), int(params["max_hold"])
    dip, slots = float(params["dip_pct"]) / 100.0, int(params["max_positions"])
    need = warmup(params)
    if len(history) < need:
        return {}
    picks = []
    for symbol in SYMBOLS:
        c = history.close(symbol, bars=need)
        if np.isnan(c[-trend:]).any():
            continue
        if c[-1] <= np.mean(c[-trend:]):
            continue
        since = holding_since(c, short, dip, max_hold)
        if since is not None:
            picks.append((since, symbol))
    chosen = [s for _, s in sorted(picks)[:slots]]
    return {s: 1.0 / slots for s in chosen}
