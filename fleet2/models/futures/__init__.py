"""Futures models: day-trading MES and MNQ for Topstep research (see base.py).

They live apart from the stock and crypto models (fleet2.models.REGISTRY), with their
own interface: they answer for every bar at once with numpy, and may go short.
"""
from __future__ import annotations

from types import ModuleType

REGISTRY: dict[str, ModuleType] = {}


def get_module(name: str) -> ModuleType:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown futures model file {name!r} (known: {', '.join(sorted(REGISTRY))})") from None
