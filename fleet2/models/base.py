"""The model interface, and checks every model's answer must pass.

A model is one Python file under fleet2/models with:

    NAME          "Momentum"
    MARKET        "stocks" or "crypto"
    DESCRIPTION   one plain-English sentence
    HOW_IT_WORKS  three sentences, no jargon
    SYMBOLS       the symbols it may hold (all from fleet2.universe)
    DEFAULT_PARAMS {name: value}
    SEARCH_SPACE  {name: (low, high, "int" | "float")}: what model search may try
    def rebalance_every(params) -> int    how many bars between decisions
    def warmup(params) -> int             how many past bars a decision needs
    def target_positions(history, params) -> {symbol: weight}

target_positions receives a fleet2.sim.marketdata.History, which shows only bars that
closed before the decision, and returns the share of the model's money to hold in each
symbol: weights from 0 to 1 that add up to at most 1 (the rest is cash). Symbols left
out are not held. Models only buy and sell (no short selling, no borrowing). A model
keeps no memory between decisions: everything it knows comes from the history, so a
backtest and paper trading make exactly the same decision on the same data.
"""
from __future__ import annotations

import math
import random
from types import ModuleType
from typing import Any

REQUIRED = ("NAME", "MARKET", "DESCRIPTION", "HOW_IT_WORKS", "SYMBOLS", "DEFAULT_PARAMS", "SEARCH_SPACE",
            "rebalance_every", "warmup", "target_positions")


class BadTargets(ValueError):
    """A model returned weights the backtester and the coordinator refuse."""


def check_module(module: ModuleType) -> None:
    """Raise when a model file lacks part of the interface."""
    missing = [name for name in REQUIRED if not hasattr(module, name)]
    if missing:
        raise TypeError(f"model {module.__name__} is missing {', '.join(missing)}")


def clean_targets(targets: Any, symbols: tuple[str, ...] | list[str]) -> dict[str, float]:
    """The model's weights, checked: known symbols, finite, 0..1 each, at most 1 in total."""
    if not isinstance(targets, dict):
        raise BadTargets(f"target_positions must return a dict, got {type(targets).__name__}")
    out: dict[str, float] = {}
    allowed = set(symbols)
    for symbol, weight in targets.items():
        if symbol not in allowed:
            raise BadTargets(f"{symbol!r} is not one of this model's symbols")
        w = float(weight)
        if not math.isfinite(w) or w < 0 or w > 1:
            raise BadTargets(f"weight {weight!r} for {symbol} must be between 0 and 1")
        if w > 0:
            out[symbol] = w
    if sum(out.values()) > 1 + 1e-9:
        raise BadTargets(f"weights add up to {sum(out.values()):.3f}, more than 1")
    return out


def params_with_defaults(module: ModuleType, params: dict[str, Any] | None) -> dict[str, Any]:
    """The model's default parameters overlaid with the given ones (unknown names dropped)."""
    merged = dict(module.DEFAULT_PARAMS)
    for key, value in (params or {}).items():
        if key in merged:
            merged[key] = value
    return merged


def draw_params(module: ModuleType, rng: random.Random) -> dict[str, Any]:
    """One random parameter set from the model's SEARCH_SPACE (used by model search)."""
    out: dict[str, Any] = {}
    for name in sorted(module.SEARCH_SPACE):
        low, high, kind = module.SEARCH_SPACE[name]
        out[name] = rng.randint(int(low), int(high)) if kind == "int" else round(rng.uniform(float(low), float(high)), 3)
    return out
