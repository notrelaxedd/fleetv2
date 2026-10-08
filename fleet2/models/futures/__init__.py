"""Futures models: day-trading MES and MNQ for Topstep research (see base.py).

They live apart from the stock and crypto models (fleet2.models.REGISTRY), with their
own interface: they answer for every bar at once with numpy, and may go short.
"""
from __future__ import annotations

from types import ModuleType

from fleet2.models.futures import gap_fade, opening_range, pullback, trend_day, vwap_revert

REGISTRY: dict[str, ModuleType] = {
    "opening_range": opening_range,
    "vwap_revert": vwap_revert,
    "trend_day": trend_day,
    "gap_fade": gap_fade,
    "pullback": pullback,
}


def get_module(name: str) -> ModuleType:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown futures model file {name!r} (known: {', '.join(sorted(REGISTRY))})") from None
