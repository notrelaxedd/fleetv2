"""config/limits.toml: the money limits and the hot-worker threshold, read once at start.

Missing keys fall back to the safe defaults below; a value of the wrong type or out of
range stops the coordinator with a message naming the key, rather than trading on a
typo.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Limits:
    """Dollar amounts as floats; percentages as plain numbers (2.0 means 2%)."""

    starting_balance_per_model: float = 10_000.0
    max_per_position: float = 1_000.0
    max_per_model: float = 10_000.0
    daily_loss_limit_pct: float = 2.0
    hot_temp_c: float = 80.0
    # Futures models paper trading on Alpaca (SPY/QQQ shares standing in for contracts).
    futures_paper_max_contracts: float = 2.0
    futures_paper_max_dollars: float = 150_000.0


RANGES: dict[str, tuple[str, float, float]] = {
    "starting_balance_per_model": ("money", 1.0, 10_000_000.0),
    "max_per_position": ("money", 1.0, 10_000_000.0),
    "max_per_model": ("money", 1.0, 10_000_000.0),
    "daily_loss_limit_pct": ("safety", 0.1, 50.0),
    "hot_temp_c": ("fleet", 30.0, 120.0),
    "futures_paper_max_contracts": ("futures", 1.0, 50.0),
    "futures_paper_max_dollars": ("futures", 1_000.0, 10_000_000.0),
}


class LimitsError(ValueError):
    """config/limits.toml has a bad value."""


def load_limits(path: Path) -> Limits:
    """Read the file (defaults when it does not exist) and check every value."""
    data: dict[str, Any] = {}
    if path.is_file():
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    values: dict[str, float] = {}
    for key, (section, low, high) in RANGES.items():
        raw = (data.get(section) or {}).get(key)
        if raw is None:
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise LimitsError(f"{path}: [{section}] {key} must be a number, got {raw!r}")
        if not low <= float(raw) <= high:
            raise LimitsError(f"{path}: [{section}] {key} must be between {low:g} and {high:g}, got {raw!r}")
        values[key] = float(raw)
    limits = Limits(**values)
    if limits.max_per_position > limits.max_per_model:
        raise LimitsError(f"{path}: max_per_position cannot be larger than max_per_model")
    return limits
