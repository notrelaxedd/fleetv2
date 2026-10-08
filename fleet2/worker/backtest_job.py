"""The backtest job: test one model on past prices, on the training period and on the
held-out period, and report the eight metrics and the Growth-of-$100 curve of each.

params (filled in by the coordinator when the job is created):
  {"model_id", "module", "params", "market", "limits": {"money", "max_per_position",
   "max_per_model"}, "held_out_fraction"}
The price bars come from the coordinator into memory (fleet2.sim.marketdata.load).
"""
from __future__ import annotations

from typing import Any, Callable

from fleet2.models import get_module
from fleet2.models.base import params_with_defaults
from fleet2.sim.backtest import Limits, run_backtest, split_at
from fleet2.sim.control import JobStopped
from fleet2.sim.marketdata import MarketData, load
from fleet2.sim.metrics import summarize
from fleet2.universe import HELD_OUT_FRACTION, MARKETS


def backtest_periods(data: MarketData, module: Any, params: dict[str, Any], limits: Limits,
                     held_out_fraction: float, emit: Callable[[float, str], None] | None = None,
                     should_stop: Callable[[], bool] | None = None, held_out_start_t: int | None = None) -> dict[str, Any]:
    """Training period [warm-up, split) and held-out period [split, end), both summarised.
    The held-out run may look back into training bars (that is the past, not the future)."""
    spec = MARKETS[data.market]
    split = split_at(data, held_out_start_t, held_out_fraction)
    if not 1 < split < data.n_bars:
        raise ValueError("the held-out start date is outside the price history; run a Data refresh")
    start = min(module.warmup(params), split - 1)
    periods = (("train", start, split), ("held_out", split, data.n_bars))
    total = sum(stop - begin for _, begin, stop in periods)
    out: dict[str, Any] = {}
    done = 0
    for name, begin, stop in periods:
        label = "training period" if name == "train" else "held-out period"

        def progress(frac: float, begin: int = begin, stop: int = stop, done: int = done, label: str = label) -> None:
            if emit is not None:
                emit((done + frac * (stop - begin)) / total, f"Backtesting the {label}")

        run = run_backtest(data, module, params, begin, stop, limits, spec["benchmark"], spec["bars_per_year"],
                           progress, should_stop)
        out[name] = summarize(run)
        done += stop - begin
    out["split_t"] = int(data.times[split])
    return out


def summary_line(result: dict[str, Any]) -> str:
    """"Held-out ROI +4.2% vs SPY +3.1% · 124 trades" for the Previous jobs table."""
    h = result["held_out"]
    bench = MARKETS[result["market"]]["benchmark"].split("/")[0]
    text = f"Held-out ROI {signed_pct(h['roi'])}"
    if h.get("benchmark_roi") is not None:
        text += f" vs {bench} {signed_pct(h['benchmark_roi'])}"
    text += f" · {h['trades']} trades"
    if not h["enough_trades"]:
        text += " (not enough trades)"
    return text


def signed_pct(fraction: float) -> str:
    """0.042 -> "+4.2%", -0.031 -> "−3.1%" (a real minus sign, never colour alone)."""
    text = f"{abs(fraction) * 100:.1f}%"
    return ("−" if fraction < 0 and text != "0.0%" else "+") + text


def run_backtest_job(params: dict[str, Any], checkpoint: dict[str, Any] | None, emit: Any, should_stop: Any) -> dict[str, Any]:
    module = get_module(str(params["module"]))
    model_params = params_with_defaults(module, params.get("params"))
    market = str(params.get("market") or module.MARKET)
    raw = params.get("limits") or {}
    limits = Limits(float(raw.get("money", 10_000)), float(raw.get("max_per_position", 1_000)),
                    float(raw.get("max_per_model", 10_000)))
    emit({}, 0.0, f"Loading {market} prices")
    if should_stop():
        raise JobStopped()
    data = load(params["_context"], market)
    emit({}, 0.02, f"Loaded {data.n_bars:,} bars of {len(data.symbols)} symbols")
    result = backtest_periods(
        data, module, model_params, limits, float(params.get("held_out_fraction", HELD_OUT_FRACTION)),
        lambda frac, detail: emit({}, 0.02 + 0.97 * frac, detail), should_stop, params.get("held_out_start_t"),
    )
    result.update({"model_id": params.get("model_id"), "market": market, "params": model_params,
                   "feed": data.feed, "limits": raw})
    result["summary"] = summary_line(result)
    return result
