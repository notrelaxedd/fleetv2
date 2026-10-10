"""The numbers on the dashboard, from a backtest Run (or a paper-trading record).

Every value is a plain float (or None when it cannot be computed honestly, for example
a Sharpe ratio of a model that never traded) so the coordinator can store it as JSON.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from fleet2.sim.backtest import Run

MIN_TRADES = 100  # below this the numbers could be luck: "not enough trades", never ranked first
# A t-statistic below this cannot be told apart from luck (the usual 95% threshold).
LUCK_T = 1.96
CURVE_POINTS = 240


def roi(equity: np.ndarray, money: float) -> float:
    return float(equity[-1] / money - 1.0) if money else 0.0


def max_drawdown(equity: np.ndarray) -> float:
    """The worst fall from a high point to a later low, as a fraction of the high.
    Pass the curve with the starting money in front, so a loss on the first bar counts."""
    peaks = np.maximum.accumulate(equity)
    falls = np.where(peaks > 0, 1.0 - equity / peaks, 0.0)
    return float(falls.max()) if falls.size else 0.0


def sharpe(equity: np.ndarray, bars_per_year: int) -> float | None:
    """Mean over spread of the per-bar returns, scaled to a year (no risk-free rate)."""
    if equity.shape[0] < 3:
        return None
    returns = np.diff(equity) / equity[:-1]
    sd = float(np.std(returns, ddof=1))
    if sd == 0.0 or not np.isfinite(sd):
        return None
    return float(np.mean(returns) / sd * np.sqrt(bars_per_year))


def t_stat(sharpe_ratio: float | None, bars: int, bars_per_year: int) -> float | None:
    """How many standard errors the mean per-bar return is above zero: the yearly Sharpe
    ratio times the square root of the years tested (mean / spread * sqrt(bars))."""
    if sharpe_ratio is None or bars <= 0 or bars_per_year <= 0:
        return None
    return float(sharpe_ratio * np.sqrt(bars / bars_per_year))


def beta_alpha(equity: np.ndarray, benchmark: np.ndarray, bars_per_year: int) -> tuple[float | None, float | None]:
    """How much the model moves with the benchmark (beta), and the return per year it made
    on top of that exposure (alpha, simple yearly sum of the per-bar excess). Both curves
    start from the starting money. Bars where either return is unknown are left out."""
    if equity.shape != benchmark.shape or equity.shape[0] < 4:
        return None, None
    with np.errstate(divide="ignore", invalid="ignore"):
        rm = np.diff(equity) / equity[:-1]
        rb = np.diff(benchmark) / benchmark[:-1]
    ok = np.isfinite(rm) & np.isfinite(rb)
    rm, rb = rm[ok], rb[ok]
    if rm.shape[0] < 3:
        return None, None
    var = float(np.var(rb, ddof=1))
    if var == 0.0 or not np.isfinite(var):
        return None, None
    beta = float(np.cov(rm, rb, ddof=1)[0, 1] / var)
    alpha = float((np.mean(rm) - beta * np.mean(rb)) * bars_per_year)
    return beta, alpha


def curve(run: Run) -> dict[str, list[Any]]:
    """Growth of $100 for the model and for buy and hold, thinned to CURVE_POINTS."""
    n = run.equity.shape[0]
    idx = np.unique(np.linspace(0, n - 1, min(n, CURVE_POINTS)).round().astype(int))
    scale = 100.0 / run.money
    return {
        "t": [int(run.times[i]) for i in idx],
        "model": [round(float(run.equity[i] * scale), 3) for i in idx],
        "benchmark": [None if np.isnan(run.benchmark[i]) else round(float(run.benchmark[i] * scale), 3) for i in idx],
    }


def summarize(run: Run) -> dict[str, Any]:
    """ROI, vs. buy and hold, max drawdown, Sharpe, win rate, profit factor, trades,
    average hold (seconds), years tested, t-statistic, beta and alpha against the
    benchmark, plus the curve and the period it covers."""
    model_roi = roi(run.equity, run.money)
    # The curve starts from the money the model was given, so the first bar's move counts.
    full = np.concatenate(([run.money], run.equity))
    bench_ok = not np.isnan(run.benchmark).all()
    bench_roi = roi(run.benchmark, run.money) if bench_ok else None
    pnls = [t.pnl for t in run.trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    holds = [t.exit_t - t.entry_t for t in run.trades]
    model_sharpe = sharpe(full, run.bars_per_year)
    bars = int(run.equity.shape[0])
    beta, alpha = (beta_alpha(full, np.concatenate(([run.money], run.benchmark)), run.bars_per_year)
                   if bench_ok else (None, None))
    return {
        "roi": model_roi,
        "benchmark_roi": bench_roi,
        "vs_buy_and_hold": None if bench_roi is None else model_roi - bench_roi,
        "max_drawdown": max_drawdown(full),
        "sharpe": model_sharpe,
        "win_rate": len(wins) / len(pnls) if pnls else None,
        "profit_factor": (sum(wins) / -sum(losses)) if losses else None,
        "trades": len(pnls),
        "avg_hold_s": float(np.mean(holds)) if holds else None,
        "enough_trades": len(pnls) >= MIN_TRADES,
        "years": bars / run.bars_per_year if run.bars_per_year else None,
        "t_stat": t_stat(model_sharpe, bars, run.bars_per_year),
        "beta": beta,
        "alpha": alpha,
        "costs_paid": round(run.costs_paid, 2),
        "start": int(run.times[0]),
        "end": int(run.times[-1]),
        "curve": curve(run),
    }
