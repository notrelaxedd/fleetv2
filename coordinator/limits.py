"""config/limits.toml: the money limits, the news strategies' trading limits and the
hot-worker threshold, read once at start.

Missing keys fall back to the safe defaults below; a value of the wrong type or out of
range stops the coordinator with a message naming the key, rather than trading on a
typo.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
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
    # Shorts by the news strategies: dollars per short position and in total per model.
    max_per_short_position: float = 500.0
    max_short_per_model: float = 5_000.0
    # When a news strategy trades a prediction (docs/NEWS_PLAN.md).
    max_trades_per_day: float = 10.0
    min_confidence: float = 0.6
    cost_multiple: float = 2.0
    min_scored_predictions: float = 100.0
    stop_loss_pct: float = 5.0
    short_borrow_fee_pct: float = 3.0
    # Replays only use news from this day on: after the model's training cutoff, so
    # Claude cannot already know how the news turned out.
    no_replay_before: date = date(2026, 8, 1)


RANGES: dict[str, tuple[str, float, float]] = {
    "starting_balance_per_model": ("money", 1.0, 10_000_000.0),
    "max_per_position": ("money", 1.0, 10_000_000.0),
    "max_per_model": ("money", 1.0, 10_000_000.0),
    "daily_loss_limit_pct": ("safety", 0.1, 50.0),
    "hot_temp_c": ("fleet", 30.0, 120.0),
    "futures_paper_max_contracts": ("futures", 1.0, 50.0),
    "futures_paper_max_dollars": ("futures", 1_000.0, 10_000_000.0),
    "max_per_short_position": ("money", 1.0, 10_000_000.0),
    "max_short_per_model": ("money", 1.0, 10_000_000.0),
    "max_trades_per_day": ("news", 0.0, 1_000.0),
    "min_confidence": ("news", 0.0, 1.0),
    "cost_multiple": ("news", 0.0, 100.0),
    "min_scored_predictions": ("news", 0.0, 100_000.0),
    "stop_loss_pct": ("news", 0.1, 50.0),
    "short_borrow_fee_pct": ("news", 0.0, 100.0),
}


class LimitsError(ValueError):
    """config/limits.toml has a bad value."""


def load_limits(path: Path) -> Limits:
    """Read the file (defaults when it does not exist) and check every value."""
    data: dict[str, Any] = {}
    if path.is_file():
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    values: dict[str, Any] = {}
    for key, (section, low, high) in RANGES.items():
        raw = (data.get(section) or {}).get(key)
        if raw is None:
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise LimitsError(f"{path}: [{section}] {key} must be a number, got {raw!r}")
        if not low <= float(raw) <= high:
            raise LimitsError(f"{path}: [{section}] {key} must be between {low:g} and {high:g}, got {raw!r}")
        values[key] = float(raw)
    raw_day = (data.get("news") or {}).get("no_replay_before")
    if raw_day is not None:
        if isinstance(raw_day, str):
            try:
                raw_day = date.fromisoformat(raw_day)
            except ValueError:
                raise LimitsError(f"{path}: [news] no_replay_before must be a date like 2026-08-01, got {raw_day!r}") from None
        if not isinstance(raw_day, date) or hasattr(raw_day, "hour"):
            raise LimitsError(f"{path}: [news] no_replay_before must be a date like 2026-08-01, got {raw_day!r}")
        values["no_replay_before"] = raw_day
    limits = Limits(**values)
    if limits.max_per_position > limits.max_per_model:
        raise LimitsError(f"{path}: max_per_position cannot be larger than max_per_model")
    if limits.max_per_short_position > limits.max_short_per_model:
        raise LimitsError(f"{path}: max_per_short_position cannot be larger than max_short_per_model")
    if limits.max_short_per_model > limits.max_per_model:
        raise LimitsError(f"{path}: max_short_per_model cannot be larger than max_per_model")
    return limits
