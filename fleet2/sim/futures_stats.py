"""The "chance this is luck" figure: the deflated Sharpe ratio of Bailey and López de
Prado (2014), which asks how likely a Sharpe ratio this good would be from settings
with no real edge, given how many settings model search tried to find it.

Try enough random settings and one of them will look good by luck alone. The more
settings were tried, and the more their results varied, the higher the best one is
expected to be by chance, and the more a found model has to beat that bar.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

EULER = 0.5772156649015329
_N = NormalDist()


def shape(pnl: np.ndarray) -> dict[str, float]:
    """Per-day Sharpe ratio (mean over spread, not scaled to a year), skew and kurtosis
    (3 for a bell curve) of daily P&L."""
    x = np.asarray(pnl, dtype=float)
    sd = float(np.std(x))
    if x.size < 3 or sd == 0.0:
        return {"sr_day": 0.0, "skew": 0.0, "kurt": 3.0, "n_days": int(x.size)}
    z = (x - x.mean()) / sd
    return {"sr_day": float(x.mean() / np.std(x, ddof=1)), "skew": float(np.mean(z ** 3)),
            "kurt": float(np.mean(z ** 4)), "n_days": int(x.size)}


def expected_best(trials: int, sr_variance: float) -> float:
    """The Sharpe ratio (per day) the best of `trials` no-edge settings reaches by luck."""
    if trials <= 1 or sr_variance <= 0:
        return 0.0
    a = _N.inv_cdf(1.0 - 1.0 / trials)
    b = _N.inv_cdf(1.0 - 1.0 / (trials * math.e))
    return math.sqrt(sr_variance) * ((1.0 - EULER) * a + EULER * b)


def chance_of_luck(sr_day: float, n_days: int, skew: float, kurt: float, trials: int, sr_variance: float) -> float | None:
    """1 - deflated Sharpe ratio: the probability that the model's true Sharpe ratio is
    no better than the best of `trials` settings with no edge. None without enough days."""
    if n_days < 3:
        return None
    bar = expected_best(trials, sr_variance)
    spread = 1.0 - skew * sr_day + (kurt - 1.0) / 4.0 * sr_day ** 2
    if spread <= 0:
        return None
    z = (sr_day - bar) * math.sqrt(n_days - 1) / math.sqrt(spread)
    return float(1.0 - _N.cdf(z))


def variance_from_sums(n: int, total: float, squares: float) -> float:
    """Variance of the tried settings' Sharpe ratios from their count, sum and sum of squares."""
    if n < 2:
        return 0.0
    mean = total / n
    return max(0.0, (squares - n * mean * mean) / (n - 1))
