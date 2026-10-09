"""The models: one Python file each, all with the same small interface (see base.py).

REGISTRY maps a model file's name to its module. A model found by model search is one
of these files with its own parameters; it never needs new code.
"""
from __future__ import annotations

from types import ModuleType

from fleet2.models import crypto_trend, dip_buy, momentum, pairs

REGISTRY: dict[str, ModuleType] = {
    "momentum": momentum,
    "dip_buy": dip_buy,
    "crypto_trend": crypto_trend,
    "pairs": pairs,
}


def get_module(name: str) -> ModuleType:
    """The model file by name (stock and crypto files, then futures files); KeyError
    names the known ones."""
    from fleet2.models.futures import REGISTRY as FUTURES
    from fleet2.models.futures import get_module as futures_module

    try:
        return REGISTRY[name] if name in REGISTRY else futures_module(name)
    except KeyError:
        known = ", ".join(sorted(REGISTRY) + sorted(FUTURES))
        raise KeyError(f"unknown model file {name!r} (known: {known})") from None
