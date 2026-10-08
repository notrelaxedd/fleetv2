"""Everything the Models screen shows, as plain dicts (the templates only format them).

Ranking: models are ranked by ROI on the held-out period, the part of history that
model search never sees. A model with under 100 held-out trades is labelled "not
enough trades" and cannot be ranked first: models with enough trades come first, by
ROI, then the rest, by ROI, then models not backtested yet.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import psycopg

from coordinator.fleet_view import TZ
from coordinator.models import STATUS_TEXT, list_models
from coordinator.settings import get_setting
from fleet2.sim.metrics import MIN_TRADES
from fleet2.universe import MARKETS

# The descriptions under each metric, word for word from the owner's spec.
METRICS = (
    ("roi", "ROI", "Total return on the money the model was given. +12% means $1,000 became $1,120."),
    ("vs_buy_and_hold", "vs. buy and hold",
     "How far ahead or behind the model is compared with simply buying SPY (or BTC) and waiting. "
     "Below zero means the model isn't earning its keep."),
    ("max_drawdown", "Max drawdown",
     "The worst fall from a high point to the next low. Smaller is safer. It's the pain you'd have had to sit through."),
    ("sharpe", "Sharpe ratio",
     "Return compared with how bumpy the ride was. Above 1 is good, above 2 is excellent, below 0 is losing money."),
    ("win_rate", "Win rate",
     "Share of trades that made money. A model can win under half its trades and still profit if its wins are "
     "bigger than its losses."),
    ("profit_factor", "Profit factor", "Dollars won for every dollar lost. Above 1.0 is profitable. 1.5 or more is strong."),
    ("trades", "Trades", "How many trades these results are based on. Under about 100, the numbers could just be luck."),
    ("avg_hold_s", "Average hold", "How long the model usually keeps a position before selling."),
)
MARKET_TEXT = {"stocks": "Stocks", "crypto": "Crypto"}
SPARK_POINTS = 40


def signed_pct(fraction: float | None, digits: int = 1) -> str:
    if fraction is None:
        return "-"
    return ("+" if fraction >= 0 else "−") + f"{abs(fraction) * 100:.{digits}f}%"


def hold_text(seconds: float | None) -> str:
    """Average hold in plain words: "45 minutes", "14 hours", "3.2 days", "6 weeks"."""
    if seconds is None:
        return "-"
    hours = seconds / 3600.0
    if hours < 1:
        return f"{max(1, round(seconds / 60))} minutes"
    if hours < 48:
        return f"{hours:.0f} hours" if hours >= 10 else f"{hours:.1f} hours"
    days = hours / 24.0
    return f"{days:.1f} days" if days < 21 else f"{days / 7:.0f} weeks"


def _tone(value: float | None, good_above: float = 0.0) -> str:
    if value is None:
        return "plain"
    return "gain" if value > good_above else "loss" if value < good_above else "plain"


def metric_cards(held: dict[str, Any], market: str) -> list[dict[str, Any]]:
    """The eight cards, each with its value, tone, word-for-word description and a note."""
    bench = MARKETS[market]["benchmark"].split("/")[0]
    cards = []
    for key, label, description in METRICS:
        value = held.get(key)
        card = {"key": key, "label": label, "description": description, "note": None, "tone": "plain"}
        if key == "roi":
            card.update(value=signed_pct(value), tone=_tone(value))
        elif key == "vs_buy_and_hold":
            card.update(value=signed_pct(value), tone=_tone(value),
                        note=f"Buying {bench} and waiting returned {signed_pct(held.get('benchmark_roi'))}")
        elif key == "max_drawdown":
            card.update(value=signed_pct(-value) if value else "0.0%", tone="loss" if value else "plain")
        elif key == "sharpe":
            card.update(value="-" if value is None else f"{value:.2f}".replace("-", "−"), tone=_tone(value),
                        note=None if value is not None else "Not enough ups and downs to measure")
        elif key == "win_rate":
            card.update(value="-" if value is None else f"{value * 100:.0f}%")
        elif key == "profit_factor":
            if value is None:
                none = "No losing trades" if held.get("trades") else "No trades"
                card.update(value=none)
            else:
                card.update(value=f"{value:.2f}", tone=_tone(value, 1.0))
        elif key == "trades":
            enough = (value or 0) >= MIN_TRADES
            card.update(value=f"{value or 0:,}", tone="plain" if enough else "warn",
                        note=None if enough else "Not enough trades: these results could be luck")
        elif key == "avg_hold_s":
            card.update(value=hold_text(value))
        cards.append(card)
    return cards


def _spark(curve: dict[str, Any] | None) -> list[float]:
    """The held-out Growth-of-$100 line thinned to SPARK_POINTS, for the list's trend line."""
    points = [p for p in ((curve or {}).get("model") or []) if p is not None]
    if len(points) <= SPARK_POINTS:
        return points
    step = (len(points) - 1) / (SPARK_POINTS - 1)
    return [points[round(i * step)] for i in range(SPARK_POINTS)]


def _date(epoch: int | None) -> str:
    if epoch is None:
        return "-"
    d = datetime.fromtimestamp(int(epoch), timezone.utc).astimezone(TZ)
    return d.strftime("%b ") + str(d.day) + d.strftime(", %Y")


def list_row(m: dict[str, Any], held_out_starts: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = m.get("metrics") or {}
    held = metrics.get("held_out") or None
    start = (held_out_starts or {}).get(m["market"])
    # A result on another held-out period (before the date was fixed) is not comparable.
    stale = bool(held and start is not None and metrics.get("split_t") is not None
                 and abs(int(metrics["split_t"]) - int(start)) > 3 * 86400)
    enough = bool(held and int(held.get("trades") or 0) >= MIN_TRADES)
    return {
        "id": m["id"],
        "name": m["name"],
        "description": m["description"],
        "market": MARKET_TEXT[m["market"]],
        "status": STATUS_TEXT[m["status"]],
        "status_key": m["status"] or "new",
        "origin": m["origin"],
        "roi": signed_pct(held.get("roi")) if held else "-",
        "roi_value": held.get("roi") if held else None,
        "tone": _tone(held.get("roi")) if held else "plain",
        "enough_trades": enough and not stale,
        "not_enough": bool(held) and not enough,
        "stale": stale,
        "spark": _spark(held.get("curve") if held else None),
    }


def ranked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ranked models (100+ held-out trades, current held-out period) first by ROI; then
    the rest, unranked: not enough trades (by ROI), an old held-out period, untested.
    Only ranked models get a rank number, so a model with under 100 trades is never
    ranked first, even when no model has enough trades yet."""
    def key(r: dict[str, Any]) -> tuple[int, float, str]:
        group = 0 if r["enough_trades"] else 3 if r["roi_value"] is None else 2 if r["stale"] else 1
        return (group, -(r["roi_value"] or 0.0), r["name"])
    out = sorted(rows, key=key)
    rank = 0
    for r in out:
        if r["enough_trades"]:
            rank += 1
            r["rank"] = rank
        else:
            r["rank"] = None
    return out


def detail(m: dict[str, Any], paper: dict[str, Any] | None = None) -> dict[str, Any]:
    """The right-hand side for one model."""
    metrics = m.get("metrics") or {}
    held = metrics.get("held_out")
    market = m["market"]
    bench = MARKETS[market]["benchmark"].split("/")[0]
    tags = [MARKET_TEXT[market], STATUS_TEXT[m["status"]]]
    if held and int(held.get("trades") or 0) < MIN_TRADES:
        tags.append("Not enough trades")
    out: dict[str, Any] = {
        "id": m["id"],
        "name": m["name"],
        "tags": tags,
        "description": m["description"],
        "how_it_works": m["how_it_works"],
        "origin": "Found by model search" if m["origin"] == "search" else "Starter model",
        "params": m["params"],
        "paper_trading": m["status"] == "paper_trading",
        "retired": m["status"] == "retired",
        "paper_button": "Stop paper trading" if m["status"] == "paper_trading" else "Start paper trading",
        "paper": paper,
        "tested": held is not None,
    }
    if held:
        curve = held.get("curve") or {}
        out["chart"] = {
            "title": "Growth of $100",
            "t": curve.get("t") or [],
            "model": curve.get("model") or [],
            "benchmark": curve.get("benchmark") or [],
            "model_label": m["name"],
            "benchmark_label": f"Buy and hold {bench}",
        }
        out["period"] = (f"Held-out period {_date(held.get('start'))} to {_date(held.get('end'))}: "
                         "prices model search never saw")
        out["metrics"] = metric_cards(held, market)
        train = metrics.get("train") or {}
        out["training_line"] = (f"Training period ROI {signed_pct(train.get('roi'))} over {train.get('trades', 0):,} trades"
                                if train else None)
        out["tested_at"] = m.get("backtested_at")
    else:
        out["untested"] = "Not tested yet. Press Run backtest to see how it would have done."
    return out


def models_page(conn: psycopg.Connection, selected_id: str | None = None,
                paper: dict[str, dict[str, Any]] | None = None, search: dict[str, Any] | None = None) -> dict[str, Any]:
    """The list (ranked) and the selected model (the first in the list by default)."""
    all_models = list_models(conn, include_retired=True)
    starts = get_setting(conn, "held_out_start", None) or {}
    rows = ranked([list_row(m, starts) for m in all_models])
    by_id = {m["id"]: m for m in all_models}
    chosen = selected_id if selected_id in by_id else (rows[0]["id"] if rows else None)
    for r in rows:
        r["selected"] = r["id"] == chosen
    notice = None
    if rows and not any(r["enough_trades"] for r in rows):
        notice = "No model has 100 trades on the held-out period yet, so none is ranked."
    return {
        "notice": notice,
        "models": rows,
        "selected": detail(by_id[chosen], (paper or {}).get(chosen)) if chosen else None,
        "search": search or {"running": False, "button": "Start model search", "detail": None},
    }
