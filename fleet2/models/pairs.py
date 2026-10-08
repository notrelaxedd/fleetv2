"""Pairs: buy whichever of two similar companies has fallen unusually far behind the other."""
from __future__ import annotations

from typing import Any

import numpy as np

NAME = "Pairs"
MARKET = "stocks"
DESCRIPTION = "Watches pairs of similar companies and buys whichever one has fallen unusually far behind its partner."
HOW_IT_WORKS = (
    "Companies like Coca-Cola and Pepsi usually move together. "
    "When one falls much further than usual compared with its partner, the model buys the one that fell behind. "
    "It sells once the gap between the two is back to normal."
)
PAIRS = (("KO", "PEP"), ("XOM", "CVX"), ("V", "MA"), ("HD", "LOW"), ("WMT", "COST"))
SYMBOLS = tuple(s for pair in PAIRS for s in pair)
DEFAULT_PARAMS: dict[str, Any] = {"lookback": 60, "entry_z": 2.0, "exit_z": 0.5}
SEARCH_SPACE = {
    "lookback": (20, 120, "int"),
    "entry_z": (1.0, 3.0, "float"),
    "exit_z": (0.0, 1.0, "float"),
}


def rebalance_every(params: dict[str, Any]) -> int:
    return 1


def warmup(params: dict[str, Any]) -> int:
    return 2 * int(params["lookback"]) + 1


def gap_scores(a: np.ndarray, b: np.ndarray, lookback: int) -> np.ndarray:
    """How unusual the price gap is on each bar: today's log ratio of a to b, minus its
    average over the previous `lookback` bars, in units of their spread (a z-score)."""
    ratio = np.log(a) - np.log(b)
    out = np.full(ratio.shape, np.nan)
    for j in range(lookback, ratio.shape[0]):
        window = ratio[j - lookback:j]
        sd = np.std(window)
        if sd > 0 and not np.isnan(sd) and not np.isnan(ratio[j]):
            out[j] = (ratio[j] - np.mean(window)) / sd
    return out


def open_leg(z: np.ndarray, entry: float, exit_level: float) -> int:
    """Replays the gap scores in order: -1 holding the first stock (it fell behind),
    +1 holding the second, 0 none. Rebuilt from history, so no memory is needed."""
    state = 0
    for value in z:
        if np.isnan(value):
            continue
        if state == 0:
            state = -1 if value < -entry else 1 if value > entry else 0
        elif abs(value) < exit_level:
            state = 0
    return state


def target_positions(history: Any, params: dict[str, Any]) -> dict[str, float]:
    lookback = int(params["lookback"])
    entry, exit_level = float(params["entry_z"]), min(float(params["exit_z"]), float(params["entry_z"]))
    need = warmup(params)
    if len(history) < need:
        return {}
    out: dict[str, float] = {}
    for first, second in PAIRS:
        a, b = history.close(first, bars=need), history.close(second, bars=need)
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        leg = open_leg(gap_scores(a, b, lookback)[lookback:], entry, exit_level)
        if leg:
            out[first if leg < 0 else second] = 1.0 / len(PAIRS)
    return out
