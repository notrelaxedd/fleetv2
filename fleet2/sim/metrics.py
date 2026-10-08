"""The eight numbers on the dashboard, from a backtest Run (or a paper-trading record).

Every value is a plain float (or None when it cannot be computed honestly, for example
a Sharpe ratio of a model that never traded) so the coordinator can store it as JSON.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from fleet2.sim.backtest import Run

MIN_TRADES = 100  # below this the numbers could be luck: "not enough trades", never ranked first
CURVE_POINTS = 240


def roi(equity: np.ndarray, money: float) -> float:
    return float(equity[-1] / money - 1.0) if money else 0.0


def max_drawdown(equity: np.ndarray) -> float:
    """The worst fall from a high point to a later low, as a fraction of the high."""
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
    average hold (seconds), plus the curve and the period it covers."""
    model_roi = roi(run.equity, run.money)
    bench_ok = not np.isnan(run.benchmark).all()
    bench_roi = roi(run.benchmark, run.money) if bench_ok else None
    pnls = [t.pnl for t in run.trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    holds = [t.exit_t - t.entry_t for t in run.trades]
    return {
        "roi": model_roi,
        "benchmark_roi": bench_roi,
        "vs_buy_and_hold": None if bench_roi is None else model_roi - bench_roi,
        "max_drawdown": max_drawdown(run.equity),
        "sharpe": sharpe(run.equity, run.bars_per_year),
        "win_rate": len(wins) / len(pnls) if pnls else None,
        "profit_factor": (sum(wins) / -sum(losses)) if losses else None,
        "trades": len(pnls),
        "avg_hold_s": float(np.mean(holds)) if holds else None,
        "enough_trades": len(pnls) >= MIN_TRADES,
        "costs_paid": round(run.costs_paid, 2),
        "start": int(run.times[0]),
        "end": int(run.times[-1]),
        "curve": curve(run),
    }
