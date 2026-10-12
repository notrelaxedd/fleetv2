"""The futures model interface (day trading MES and MNQ), and checks every answer must pass.

A futures model is one Python file under fleet2/models/futures with:

    NAME          "Opening range"
    MARKET        "futures"
    DESCRIPTION   one plain-English sentence
    HOW_IT_WORKS  three sentences, no jargon
    SYMBOLS       the symbols it may trade ("MES", "MNQ")
    DEFAULT_PARAMS {name: value}, always including
                    symbol        "MES" or "MNQ"
                    bar_minutes   1, 3, 5 or 15: the size of the bars it decides on
                    stop_ticks    0 for none, else the loss in ticks at which a trade is closed
                    target_ticks  0 for none, else the gain in ticks at which a trade is closed
    SEARCH_SPACE  {name: (low, high, "int" | "float")} or {name: (choices, "choice")}
    def targets(bars, params) -> {symbol: array}

targets() receives every decision bar at once (fleet2.sim.futures_data.Bars, built from
1-minute bars) and returns, per symbol, one number per bar from -1 (fully short) to +1
(fully long); 0 is flat. The number for bar k is the position wanted once bar k has
closed; it is filled at the open of the next 1-minute bar. It must be computed with
numpy from bars 0..k only: tests/test_futures_cutoff.py cuts the prices off at random
bars and checks that every earlier answer stays exactly the same.

The backtester turns the answer into whole contracts (the size being tested), handles
the stop and target, blocks new trades near the close and closes everything by the end
of the session, so a model never has to.
"""
from __future__ import annotations

import random
from types import ModuleType
from typing import Any

import numpy as np

from fleet2.sim.futures_data import BAR_SIZES

REQUIRED = ("NAME", "MARKET", "DESCRIPTION", "HOW_IT_WORKS", "SYMBOLS", "DEFAULT_PARAMS", "SEARCH_SPACE", "targets")
ALWAYS = ("symbol", "bar_minutes", "stop_ticks", "target_ticks")


class BadTargets(ValueError):
    """A futures model returned targets the backtester refuses."""


def check_module(module: ModuleType) -> None:
    missing = [name for name in REQUIRED if not hasattr(module, name)]
    missing += [f"DEFAULT_PARAMS[{p!r}]" for p in ALWAYS if p not in getattr(module, "DEFAULT_PARAMS", {})]
    if missing:
        raise TypeError(f"futures model {module.__name__} is missing {', '.join(missing)}")
    if module.MARKET != "futures":
        raise TypeError(f"{module.__name__} is not a futures model")


def clean_targets(raw: Any, symbols: tuple[str, ...], n: int) -> dict[str, np.ndarray]:
    """The model's targets, checked: known symbols, one finite number per bar, -1..+1."""
    if not isinstance(raw, dict):
        raise BadTargets(f"targets must return a dict of arrays, got {type(raw).__name__}")
    out: dict[str, np.ndarray] = {}
    for symbol, values in raw.items():
        if symbol not in symbols:
            raise BadTargets(f"{symbol!r} is not one of this model's symbols")
        arr = np.asarray(values, dtype=float)
        if arr.shape != (n,):
            raise BadTargets(f"{symbol}: {arr.shape} targets for {n} bars")
        if not np.isfinite(arr).all():
            raise BadTargets(f"{symbol}: targets must be finite numbers")
        if np.abs(arr).max(initial=0.0) > 1.0 + 1e-9:
            raise BadTargets(f"{symbol}: targets must be between -1 and +1")
        out[symbol] = np.clip(arr, -1.0, 1.0)
    return out


def params_with_defaults(module: ModuleType, params: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(module.DEFAULT_PARAMS)
    for key, value in (params or {}).items():
        if key in merged:
            merged[key] = value
    if int(merged["bar_minutes"]) not in BAR_SIZES:
        raise ValueError(f"bar_minutes must be one of {BAR_SIZES}")
    return merged


# ------------------------------------------------------------------ drawing settings (model search)


def _clip(module: ModuleType, name: str, value: Any) -> Any:
    spec = module.SEARCH_SPACE[name]
    if spec[-1] == "choice":
        return value if value in spec[0] else spec[0][0]
    low, high, kind = spec
    value = min(max(value, low), high)
    return int(round(value)) if kind == "int" else round(float(value), 3)


def draw_params(module: ModuleType, rng: random.Random) -> dict[str, Any]:
    """One random set of settings from the model's SEARCH_SPACE."""
    out: dict[str, Any] = {}
    for name in sorted(module.SEARCH_SPACE):
        spec = module.SEARCH_SPACE[name]
        if spec[-1] == "choice":
            out[name] = rng.choice(list(spec[0]))
        else:
            low, high, kind = spec
            out[name] = rng.randint(int(low), int(high)) if kind == "int" else round(rng.uniform(float(low), float(high)), 3)
    return params_with_defaults(module, out)


def mutate_params(module: ModuleType, parent: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """A small change to a kept model's settings: every number moves by up to 15% (at
    least one step for whole numbers), and each choice changes one time in five."""
    out = dict(params_with_defaults(module, parent))
    for name in sorted(module.SEARCH_SPACE):
        spec = module.SEARCH_SPACE[name]
        if spec[-1] == "choice":
            if rng.random() < 0.2:
                out[name] = rng.choice(list(spec[0]))
            continue
        low, high, kind = spec
        value = float(out[name])
        step = max(abs(value) * 0.15, (high - low) * 0.02)
        moved = value + rng.uniform(-step, step)
        if kind == "int" and int(round(moved)) == int(value):
            moved = value + rng.choice((-1, 1))
        out[name] = _clip(module, name, moved)
    return out


def neighbours(module: ModuleType, params: dict[str, Any], frac: float = 0.10) -> list[dict[str, Any]]:
    """Two neighbours per numeric setting, that setting moved by -10% and +10% (whole
    numbers by at least one step), everything else unchanged. Choices are left alone."""
    out = []
    base = params_with_defaults(module, params)
    for name in sorted(module.SEARCH_SPACE):
        spec = module.SEARCH_SPACE[name]
        if spec[-1] == "choice":
            continue
        low, high, kind = spec
        value = float(base[name])
        for sign in (-1, 1):
            moved = value * (1 + sign * frac)
            if kind == "int" and int(round(moved)) == int(value):
                moved = value + sign
            moved = _clip(module, name, moved)
            if moved != base[name]:
                out.append({**base, name: moved})
    return out
