"""Futures models: day-trading MES and MNQ for Topstep research (see base.py).

They live apart from the stock and crypto models (fleet2.models.REGISTRY), with their
own interface: they answer for every bar at once with numpy, and may go short.

Besides the five files there are recipes (recipe.py): models put together from building
blocks, named "recipe_..." and rebuilt from the recipe kept in their settings. Look a
model up with module_for(name, settings), which handles both.
"""
from __future__ import annotations

from types import ModuleType
from typing import Any

from fleet2.models.futures import fair_value_gap, gap_fade, opening_range, pullback, recipe, trend_day, vwap_revert

REGISTRY: dict[str, ModuleType] = {
    "opening_range": opening_range,
    "vwap_revert": vwap_revert,
    "trend_day": trend_day,
    "gap_fade": gap_fade,
    "pullback": pullback,
    "fair_value_gap": fair_value_gap,
}


def get_module(name: str) -> ModuleType:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown futures model file {name!r} (known: {', '.join(sorted(REGISTRY))})") from None


def module_for(name: str, params: dict[str, Any] | None = None) -> ModuleType:
    """A model file by name, or the recipe in `params` when the name is a recipe's (the
    name must match the recipe, so a model can never run a different recipe)."""
    if not recipe.is_recipe(name):
        return get_module(name)
    raw = (params or {}).get("recipe")
    if raw is None:
        raise KeyError(f"{name} is a recipe, but its settings hold no recipe")
    module = recipe.family(raw)
    if module.__name__ != name:
        raise KeyError(f"the recipe in the settings is {module.__name__}, not {name}")
    return module


def tries_bucket(name: str) -> str:
    """What the "chance this is luck" figure counts tries under: the model file, or
    "recipe" for every recipe together (choosing among recipes is part of the search)."""
    return "recipe" if recipe.is_recipe(name) else name
