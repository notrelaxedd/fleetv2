"""Recipes: futures models put together from building blocks (docs/AI_PLAN.md, stage A).

A recipe is a choice of blocks, never code:

    {"signal": "range_break", "direction": "follow", "filters": ["quiet_day"],
     "exit": "vwap_cross", "entries": "first", "side": "both"}

- signal     what starts a trade (SIGNALS): an "up" and a "down" event per bar
- direction  "follow" buys on up and sells short on down; "fade" does the opposite
- filters    up to two conditions a trade must also meet (FILTERS)
- exit       when a trade ends, besides its stop, its target and the close (EXITS)
- entries    "first": only the day's first signal; "every": any signal
- side       "both", "long" (buys only) or "short" (sells short only)

family(recipe) turns a recipe into an object that works like a model file (NAME,
DESCRIPTION, HOW_IT_WORKS, DEFAULT_PARAMS, SEARCH_SPACE, targets...), so model search,
the backtester and live trading use it exactly as they use the five files. Its name,
"recipe_" plus a hash of the recipe, is the same wherever the recipe is rebuilt, and a
model stores its recipe in its settings ("recipe"), so every job can rebuild it.

Every block is computed with numpy from the bar itself and earlier bars only (the
helpers in features.py); tests/test_futures_recipes.py runs the cut-off test on many
random recipes. Recipes may come from a random mix (random_recipe) or from Claude
Haiku (stage B); either way they are checked by validate() and run by this file only.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from types import ModuleType
from typing import Any

import numpy as np

from fleet2.models.futures import features as f
from fleet2.sim.futures_data import Bars

PREFIX = "recipe_"
SYMBOLS = ("MES", "MNQ")
MAX_FILTERS = 2

# Every recipe has these settings. "normal" below means a normal day's high-to-low range:
# the average of the last lookback_days days' ranges.
COMMON_SPACE: dict[str, tuple] = {
    "symbol": (SYMBOLS, "choice"),
    "bar_minutes": ((1, 3, 5, 15), "choice"),
    "start_minute": (0, 240, "int"),         # no new trades before this many minutes after the open
    "last_entry_minute": (60, 380, "int"),   # nor after this many
    "stop_ticks": (0, 80, "int"),
    "target_ticks": (0, 160, "int"),
}
COMMON_DEFAULTS: dict[str, Any] = {"symbol": "MES", "bar_minutes": 5, "start_minute": 30, "last_entry_minute": 240,
                                   "stop_ticks": 32, "target_ticks": 0}
LOOKBACK_SPACE = {"lookback_days": (5, 30, "int")}

# name: (plain words, settings {name: search range}, uses a normal day's range)
SIGNALS: dict[str, dict[str, Any]] = {
    "range_break": {
        "plural": "opening-range breaks",
        "title": "opening-range break",
        "up": "the price closes above the high of the first minutes of the day",
        "down": "the price closes below the low of the first minutes of the day",
        "space": {"range_minutes": (15, 90, "int"), "range_buffer": (0.0, 0.3, "float")},
        "normal": True,
    },
    "vwap_stretch": {
        "plural": "stretches from VWAP",
        "title": "stretch from the day's average price",
        "up": "the price is well above the day's average price (VWAP)",
        "down": "the price is well below the day's average price (VWAP)",
        "space": {"stretch": (0.1, 0.8, "float")},
        "normal": True,
    },
    "average_cross": {
        "plural": "average crosses",
        "title": "average cross",
        "up": "the short average of today's prices crosses above the longer one",
        "down": "the short average of today's prices crosses below the longer one",
        "space": {"fast_bars": (2, 12, "int"), "slow_bars": (8, 40, "int")},
        "normal": False,
    },
    "gap": {
        "plural": "gaps",
        "title": "gap from yesterday",
        "up": "the day opened well above yesterday's close",
        "down": "the day opened well below yesterday's close",
        "space": {"gap_size": (0.05, 0.6, "float")},
        "normal": True,
    },
    "day_move": {
        "plural": "moves from the open",
        "title": "move from the open",
        "up": "the price is well above today's open",
        "down": "the price is well below today's open",
        "space": {"move_size": (0.1, 1.0, "float")},
        "normal": True,
    },
    "new_extreme": {
        "plural": "new highs and lows",
        "title": "new high or low of the day",
        "up": "the price closes above the day's high so far",
        "down": "the price closes below the day's low so far",
        "space": {},
        "normal": False,
    },
    "fvg": {
        "plural": "fair value gap retests",
        "title": "fair value gap retest",
        "up": "the price dips back into a recent upward fair value gap and holds above its bottom",
        "down": "the price rises back into a recent downward fair value gap and holds below its top",
        "space": {"fvg_size": (0.02, 0.3, "float"), "fvg_bars": (3, 60, "int")},
        "normal": True,
    },
    "momentum": {
        "plural": "momentum",
        "title": "short-term momentum",
        "up": "the price has risen clearly over the last few bars",
        "down": "the price has fallen clearly over the last few bars",
        "space": {"momentum_bars": (2, 24, "int"), "momentum_size": (0.05, 0.6, "float")},
        "normal": True,
    },
}

FILTERS: dict[str, dict[str, Any]] = {
    "quiet_day": {"words": "the day has been calm so far (a small range)", "space": {"quiet": (0.3, 1.2, "float")},
                  "normal": True},
    "busy_day": {"words": "the day has been busy so far (a wide range)", "space": {"busy": (0.4, 1.6, "float")},
                 "normal": True},
    "vwap_side": {"words": "the price is on the trade's side of the day's average price", "space": {},
                  "normal": False},
    "day_side": {"words": "the price is on the trade's side of today's open", "space": {}, "normal": False},
    "after_gap": {"words": "the day opened with a clear gap from yesterday's close",
                  "space": {"gap_min": (0.05, 0.6, "float")}, "normal": True},
    "no_gap": {"words": "the day opened close to yesterday's close", "space": {"gap_max": (0.05, 0.6, "float")},
               "normal": True},
}
CLASHES = ({"quiet_day", "busy_day"}, {"after_gap", "no_gap"})

EXITS: dict[str, dict[str, Any]] = {
    "hold": {"words": "It holds a trade until its stop, its target or the close.", "space": {}},
    "vwap_cross": {"words": "It gets out when the price crosses back through the day's average price.", "space": {}},
    "vwap_return": {"words": "It takes its profit when the price gets back to the day's average price.", "space": {}},
    "opposite": {"words": "It gets out when the opposite signal appears.", "space": {}},
    "bars": {"words": "It gets out after a set number of bars.", "space": {"hold_bars": (2, 48, "int")}},
}
DIRECTIONS = ("follow", "fade")
ENTRIES = ("first", "every")
SIDES = ("both", "long", "short")


class BadRecipe(ValueError):
    """A recipe that is not made of known blocks (the message says what is wrong)."""


# ------------------------------------------------------------------ checking and naming


def validate(raw: Any) -> dict[str, Any]:
    """The recipe in its one canonical form, or BadRecipe saying what is wrong."""
    if not isinstance(raw, dict):
        raise BadRecipe("a recipe must be an object")
    known = {"signal", "direction", "filters", "exit", "entries", "side"}
    extra = sorted(set(raw) - known)
    if extra:
        raise BadRecipe(f"unknown recipe fields: {', '.join(extra)}")

    def pick(field: str, allowed: Any) -> str:
        value = raw.get(field)
        if value not in allowed:
            raise BadRecipe(f"{field} must be one of {', '.join(sorted(allowed))}, not {value!r}")
        return str(value)

    filters = raw.get("filters") or []
    if not isinstance(filters, list) or not all(isinstance(x, str) for x in filters):
        raise BadRecipe("filters must be a list of filter names")
    unknown = [x for x in filters if x not in FILTERS]
    if unknown:
        raise BadRecipe(f"unknown filters: {', '.join(unknown)} (known: {', '.join(sorted(FILTERS))})")
    filters = sorted(set(filters))
    if len(filters) > MAX_FILTERS:
        raise BadRecipe(f"at most {MAX_FILTERS} filters")
    for pair in CLASHES:
        if pair <= set(filters):
            raise BadRecipe(f"filters {' and '.join(sorted(pair))} contradict each other")
    return {"signal": pick("signal", SIGNALS), "direction": pick("direction", DIRECTIONS), "filters": filters,
            "exit": pick("exit", EXITS), "entries": pick("entries", ENTRIES), "side": pick("side", SIDES)}


def name_of(recipe: dict[str, Any]) -> str:
    """"recipe_" and the first 10 hex digits of the canonical recipe's hash."""
    text = json.dumps(validate(recipe), sort_keys=True, separators=(",", ":"))
    return PREFIX + hashlib.sha256(text.encode()).hexdigest()[:10]


def is_recipe(name: str) -> bool:
    return str(name).startswith(PREFIX)


def random_recipe(rng: random.Random) -> dict[str, Any]:
    """A random mix of blocks (each draw depends only on rng, so a seed re-creates it)."""
    filters: list[str] = []
    count = rng.choices((0, 1, 2), weights=(3, 5, 2))[0]
    while len(filters) < count:
        pick = rng.choice(sorted(FILTERS))
        if pick in filters or any(pair <= set(filters) | {pick} for pair in CLASHES):
            continue
        filters.append(pick)
    return validate({
        "signal": rng.choice(sorted(SIGNALS)), "direction": rng.choice(DIRECTIONS), "filters": filters,
        "exit": rng.choice(sorted(EXITS)), "entries": rng.choice(ENTRIES),
        "side": rng.choices(SIDES, weights=(6, 2, 2))[0],
    })


# ------------------------------------------------------------------ plain words


def _title(r: dict[str, Any]) -> str:
    verb = "Follow" if r["direction"] == "follow" else "Fade"
    plural = SIGNALS[r["signal"]]["plural"]
    when = {"quiet_day": "on quiet days", "busy_day": "on busy days", "vwap_side": "with VWAP",
            "day_side": "with the day", "after_gap": "after a gap", "no_gap": "without a gap"}
    tail = " ".join(when[x] for x in r["filters"])
    side = {"both": "", "long": " (buys only)", "short": " (shorts only)"}[r["side"]]
    out = {"hold": "", "vwap_cross": ", out at VWAP", "vwap_return": ", profit at VWAP",
           "opposite": ", out on the opposite signal", "bars": ", timed exit"}[r["exit"]]
    return f"{verb} {plural}{(' ' + tail) if tail else ''}{side}{out}"


def _description(r: dict[str, Any]) -> str:
    sig = SIGNALS[r["signal"]]
    buy, sell = (sig["up"], sig["down"]) if r["direction"] == "follow" else (sig["down"], sig["up"])
    if r["side"] == "long":
        return f"A recipe model: buys when {buy}."
    if r["side"] == "short":
        return f"A recipe model: sells short when {sell}."
    return f"A recipe model: buys when {buy}, and sells short when {sell}."


def _how_it_works(r: dict[str, Any]) -> str:
    sig = SIGNALS[r["signal"]]
    style = ("It trades in the direction of the move." if r["direction"] == "follow"
             else "It bets against the move, expecting it to turn back.")
    first = (f"It watches for the {sig['title']}: {sig['up']}, or {sig['down']}. {style}")
    if r["filters"]:
        second = "It only trades when " + " and ".join(FILTERS[x]["words"] for x in r["filters"]) + "."
    else:
        second = "It trades on any day."
    third = EXITS[r["exit"]]["words"]
    third += " It takes only the day's first signal." if r["entries"] == "first" else " It can trade several times a day."
    return f"{first} {second} {third}"


# ------------------------------------------------------------------ settings


def space_of(r: dict[str, Any]) -> dict[str, tuple]:
    space = dict(COMMON_SPACE)
    space.update(SIGNALS[r["signal"]]["space"])
    for x in r["filters"]:
        space.update(FILTERS[x]["space"])
    space.update(EXITS[r["exit"]]["space"])
    if uses_normal(r):
        space.update(LOOKBACK_SPACE)
    return space


def uses_normal(r: dict[str, Any]) -> bool:
    return bool(SIGNALS[r["signal"]]["normal"] or any(FILTERS[x]["normal"] for x in r["filters"]))


def _middle(spec: tuple) -> Any:
    if spec[-1] == "choice":
        return spec[0][0]
    low, high, kind = spec
    mid = (low + high) / 2
    return int(round(mid)) if kind == "int" else round(float(mid), 3)


def defaults_of(r: dict[str, Any]) -> dict[str, Any]:
    out = {name: _middle(spec) for name, spec in space_of(r).items()}
    out.update(COMMON_DEFAULTS)
    if uses_normal(r):
        out["lookback_days"] = 10
    out["recipe"] = dict(r)
    return out


# ------------------------------------------------------------------ the blocks


def _today_mean(x: np.ndarray, n: int, bars: Bars) -> np.ndarray:
    """Mean of x over the last n bars of the same day, this bar included (NaN until the
    day has n bars)."""
    c = np.r_[0.0, np.cumsum(x)]
    k = np.arange(bars.n)
    j = k + 1 - n
    ok = j >= f.first_index(bars)
    return np.where(ok, (c[k + 1] - c[np.maximum(j, 0)]) / max(n, 1), np.nan)


def _bars_ago(x: np.ndarray, n: int, bars: Bars) -> np.ndarray:
    """x of the bar n bars back on the same day (NaN when that is before the day's start)."""
    k = np.arange(bars.n)
    j = k - n
    ok = j >= f.first_index(bars)
    return np.where(ok, x[np.maximum(j, 0)], np.nan)


def _previous(x: np.ndarray, bars: Bars) -> np.ndarray:
    """x of the bar before, NaN on a day's first bar."""
    return np.where(bars.first, np.nan, np.r_[np.nan, x[:-1]])


def fair_value_gaps(s: Any, bars: Bars, min_size: np.ndarray, max_bars: int) -> tuple[np.ndarray, np.ndarray]:
    """Retests of the day's latest fair value gap, as (up, down) events per bar.

    A fair value gap is a three-bar pattern within one day: an upward gap forms at bar k
    when its low is above the high of bar k-2 (the zone between them is the gap); a
    downward one when its high is below the low of bar k-2. Gaps smaller than
    `min_size` are ignored. "up" at a later bar j: within `max_bars` bars of the latest
    upward gap of the day, the price dips into that gap (low at or below its top) and
    closes at or above its bottom, the gap not having been filled before (no low below
    its bottom since it formed). "down" is the mirror image. Everything uses bars up to
    j only (k <= j - 1, and the bars between k and j)."""
    k = np.arange(bars.n, dtype=float)
    high2, low2 = _bars_ago(s.high, 2, bars), _bars_ago(s.low, 2, bars)
    with np.errstate(invalid="ignore"):
        bull = (s.low - high2) > min_size
        bear = (low2 - s.high) > min_size

    def retest(formed: np.ndarray, top_at: np.ndarray, bottom_at: np.ndarray, rising: bool) -> np.ndarray:
        top = f.latest_today(top_at, formed, bars)
        bottom = f.latest_today(bottom_at, formed, bars)
        since = k - f.latest_today(k, formed, bars)
        # The extreme of the bars after the gap formed, up to the bar before this one.
        extreme = np.full(bars.n, np.inf if rising else -np.inf)
        for lag in range(1, max_bars + 1):
            v = _bars_ago(s.low if rising else s.high, lag, bars)
            use = (lag <= since - 1) & ~np.isnan(v)
            extreme = np.where(use, (np.minimum if rising else np.maximum)(extreme, v), extreme)
        with np.errstate(invalid="ignore"):
            fresh = (since >= 1) & (since <= max_bars)
            if rising:
                return fresh & (extreme >= bottom) & (s.low <= top) & (s.close >= bottom)
            return fresh & (extreme <= top) & (s.high >= bottom) & (s.close <= top)

    up = retest(bull, s.low, high2, True)      # an upward gap: from the high of k-2 up to the low of k
    down = retest(bear, low2, s.high, False)   # a downward gap: from the high of k up to the low of k-2
    return up, down


def _signal(r: dict[str, Any], s: Any, bars: Bars, p: dict[str, Any], normal: np.ndarray,
            ctx: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(up, down, allowed): the signal's events, and the bars on which it may start a trade."""
    name = r["signal"]
    allowed = np.ones(bars.n, dtype=bool)
    if name == "range_break":
        span = int(p["range_minutes"])
        in_range = bars.minute + bars.size <= span
        top = f.high_today_where(s.high, in_range, bars)
        bottom = f.low_today_where(s.low, in_range, bars)
        buf = float(p["range_buffer"]) * normal
        up, down = s.close > top + buf, s.close < bottom - buf
        allowed = bars.minute >= span
    elif name == "vwap_stretch":
        gap = s.close - ctx["vwap"]
        up, down = gap > float(p["stretch"]) * normal, gap < -float(p["stretch"]) * normal
    elif name == "average_cross":
        fast_n, slow_n = int(p["fast_bars"]), int(p["slow_bars"])
        fast, slow = _today_mean(s.close, fast_n, bars), _today_mean(s.close, slow_n, bars)
        pf, ps = _previous(fast, bars), _previous(slow, bars)
        up = (fast > slow) & (pf <= ps)
        down = (fast < slow) & (pf >= ps)
        if fast_n >= slow_n:  # no "fast" average to speak of
            up = down = np.zeros(bars.n, dtype=bool)
    elif name == "gap":
        g = ctx["gap"]
        up, down = g > float(p["gap_size"]) * normal, g < -float(p["gap_size"]) * normal
    elif name == "day_move":
        m = s.close - ctx["open"]
        up, down = m > float(p["move_size"]) * normal, m < -float(p["move_size"]) * normal
    elif name == "new_extreme":
        up = s.close > _previous(f.high_today(s.high, bars), bars)
        down = s.close < _previous(f.low_today(s.low, bars), bars)
    elif name == "momentum":
        d = s.close - _bars_ago(s.close, int(p["momentum_bars"]), bars)
        size = float(p["momentum_size"]) * normal
        up, down = d > size, d < -size
    elif name == "fvg":
        up, down = fair_value_gaps(s, bars, float(p["fvg_size"]) * normal, int(p["fvg_bars"]))
    else:  # validate() makes this unreachable
        raise BadRecipe(f"unknown signal {name!r}")
    return up, down, allowed


def _filter(name: str, s: Any, bars: Bars, p: dict[str, Any], normal: np.ndarray,
            ctx: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(ok for a long trade, ok for a short trade)."""
    if name in ("quiet_day", "busy_day"):
        today = f.high_today(s.high, bars) - f.low_today(s.low, bars)
        ok = today < float(p["quiet"]) * normal if name == "quiet_day" else today > float(p["busy"]) * normal
        return ok, ok
    if name == "vwap_side":
        return s.close > ctx["vwap"], s.close < ctx["vwap"]
    if name == "day_side":
        return s.close > ctx["open"], s.close < ctx["open"]
    if name == "after_gap":
        ok = np.abs(ctx["gap"]) > float(p["gap_min"]) * normal
        return ok, ok
    if name == "no_gap":
        ok = np.abs(ctx["gap"]) < float(p["gap_max"]) * normal
        return ok, ok
    raise BadRecipe(f"unknown filter {name!r}")


def _bars_since(enter: np.ndarray, bars: Bars) -> np.ndarray:
    """Bars since the latest bar of the same day where enter is true (NaN before one)."""
    k = np.arange(bars.n, dtype=float)
    last = f.latest_today(k, enter, bars)
    return k - last


def recipe_targets(r: dict[str, Any], bars: Bars, p: dict[str, Any]) -> dict[str, np.ndarray]:
    symbol = str(p["symbol"])
    s = bars.series(symbol)
    normal = (f.average_range(s, bars, int(p["lookback_days"])) if uses_normal(r)
              else np.full(bars.n, np.nan))
    ctx = {"vwap": f.vwap(s, bars), "open": f.day_open(s, bars)}
    ctx["gap"] = ctx["open"] - f.previous_close(s, bars)
    with np.errstate(invalid="ignore"):
        up, down, allowed = _signal(r, s, bars, p, normal, ctx)
        buy, sell = (up, down) if r["direction"] == "follow" else (down, up)
        window = allowed & (bars.minute >= int(p["start_minute"])) & (bars.minute < int(p["last_entry_minute"]))
        long_ok, short_ok = window.copy(), window.copy()
        for name in r["filters"]:
            ok_long, ok_short = _filter(name, s, bars, p, normal, ctx)
            long_ok &= ok_long
            short_ok &= ok_short
        enter_long = buy & long_ok & (r["side"] != "short")
        enter_short = sell & short_ok & (r["side"] != "long")
        if r["entries"] == "first":
            first = f.first_today(enter_long | enter_short, bars)
            enter_long, enter_short = enter_long & first, enter_short & first
        exit_name = r["exit"]
        if exit_name == "hold":
            leave_long = leave_short = np.zeros(bars.n, dtype=bool)
        elif exit_name == "vwap_cross":
            leave_long, leave_short = s.close < ctx["vwap"], s.close > ctx["vwap"]
        elif exit_name == "vwap_return":
            leave_long, leave_short = s.close >= ctx["vwap"], s.close <= ctx["vwap"]
        elif exit_name == "opposite":
            leave_long, leave_short = sell, buy
        elif exit_name == "bars":
            n = int(p["hold_bars"])
            leave_long = _bars_since(enter_long, bars) >= n
            leave_short = _bars_since(enter_short, bars) >= n
        else:
            raise BadRecipe(f"unknown exit {exit_name!r}")
        long = f.hold(enter_long, leave_long, 1.0, bars)
        short = f.hold(enter_short, leave_short, -1.0, bars)
    return {symbol: np.clip(long + short, -1.0, 1.0)}


# ------------------------------------------------------------------ a recipe as a model file

_FAMILIES: dict[str, ModuleType] = {}


def family(raw: Any) -> ModuleType:
    """The model-file object of a recipe (built once per process, then reused)."""
    r = validate(raw)
    name = name_of(r)
    if name in _FAMILIES:
        return _FAMILIES[name]
    m = ModuleType(name, _description(r))
    m.NAME = _title(r)
    m.MARKET = "futures"
    m.DESCRIPTION = _description(r)
    m.HOW_IT_WORKS = _how_it_works(r)
    m.SYMBOLS = SYMBOLS
    m.RECIPE = r
    m.DEFAULT_PARAMS = defaults_of(r)
    m.SEARCH_SPACE = space_of(r)
    m.targets = lambda bars, params, _r=r: recipe_targets(_r, bars, params)
    m.__file__ = sys.modules[__name__].__file__
    _FAMILIES[name] = m
    return m
