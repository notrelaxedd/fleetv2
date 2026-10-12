"""The News screen as plain dicts (the templates only format them).

Claude reads market news and predicts how each stock or coin it names will move; every
prediction is saved and later scored against the real price and a coin-flip twin. This
screen shows the state of the machinery, how the scores stand, the two paper-trading
books and the latest articles with Claude's calls. See docs/NEWS_PLAN.md.

Article text and Claude's rationale are untrusted: nothing here marks any of it safe.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg

from coordinator import haiku, news_scoring
from coordinator.fleet_view import TZ, ago, header, money, pct

MARKETS = ("stocks", "crypto")
MARKET_NAME = {"stocks": "Stocks", "crypto": "Crypto"}
MARKET_BENCH = {"stocks": "the stock market (SPY)", "crypto": "Bitcoin"}
MARKET_NOTE = {
    "stocks": "Shorts allowed (whole shares, easy-to-borrow only); never SPY or QQQ",
    "crypto": "Long only: Alpaca cannot short crypto",
}
FEED_LIMIT = 50
CAP_WARN = 0.8  # amber from this share of a spending cap

ANALYSIS_TEXT = {
    "pending": "Waiting for Claude",
    "backfill": "Saved for replay only",
    "analysing": "Claude is reading it",
    "analysed": "Read by Claude",
    "skipped": "Skipped",
    "failed": "Claude could not read it",
    "over_budget": "Not read: over the spending cap",
}
BOOK_TEXT = {
    "not_started": "Not started",
    "active": "Paper trading",
    "closing": "Stopping: selling its positions",
    "closed": "Stopped",
}
DIRECTION = {"up": ("▲", "Up"), "down": ("▼", "Down"), "none": ("–", "No call")}


def share(value: float | None) -> str:
    """0.543 -> "54%"."""
    return "-" if value is None else f"{value * 100:.0f}%"


def when(ts: datetime | None) -> str:
    if ts is None:
        return "-"
    local = ts.astimezone(TZ)
    return f"{local:%b} {local.day}, {local.hour % 12 or 12}:{local:%M} {'am' if local.hour < 12 else 'pm'}"


def safe_url(url: str | None) -> str | None:
    """Only web links become links (a stored `javascript:` address never does)."""
    text = (url or "").strip()
    return text if text.lower().startswith(("http://", "https://")) else None


def _day_start(now: datetime) -> datetime:
    return now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def _tone(value: float | None) -> str:
    return "plain" if value is None or value == 0 else "gain" if value > 0 else "loss"


# ------------------------------------------------------------------ status line


def _service(obj: Any, ready: str, down: str) -> dict[str, Any]:
    if obj is None:
        return {"value": "Not set up", "tone": "muted", "note": "Waiting for the coordinator to start it."}
    if getattr(obj, "connected", False):
        return {"value": ready, "tone": "gain", "note": ""}
    return {"value": down, "tone": "warn", "note": str(getattr(obj, "problem", None) or "No reason was given.")}


def _cap_card(key: str, label: str, spent: float, cap: float, what: str) -> dict[str, Any]:
    over = cap > 0 and spent >= cap * CAP_WARN
    return {"key": key, "label": label, "value": f"{money(spent)} of {money(cap)}", "tone": "warn" if over else "plain",
            "description": what,
            "note": ("Over the cap: no more articles are read until it resets." if cap > 0 and spent >= cap
                     else "Close to the cap." if over else "")}


def status_cards(conn: psycopg.Connection, feed: Any, analyst: Any, ai: Any, now: datetime) -> list[dict[str, Any]]:
    start = _day_start(now)
    row = conn.execute(
        "SELECT (SELECT coalesce(sum(cost_usd), 0) FROM ai_calls WHERE feature = 'news' AND at >= %s AND at < %s) AS spent,"
        " (SELECT count(*) FROM news WHERE created_at >= %s) AS today,"
        " (SELECT count(*) FROM news WHERE analysis_status = 'pending') AS waiting",
        (start, start + timedelta(days=1), start)).fetchone()
    feed_card = {"key": "feed", "label": "News feed",
                 "description": "Where the headlines come from: Alpaca's news service.", **_service(feed, "Connected", "Not connected")}
    model = getattr(analyst, "model", "") if analyst is not None else ""
    analyst_card = {"key": "analyst", "label": "Claude, the analyst",
                    "description": "The AI that reads each article and predicts the move.",
                    **_service(analyst, "Ready", "Not ready")}
    if analyst is not None and getattr(analyst, "connected", False) and model:
        analyst_card["note"] = f"Model: {model}"
    cap_today = float(getattr(ai, "news_daily_cap_usd", 0.0))
    cap_month = float(getattr(ai, "monthly_cap_usd", 0.0))
    return [
        feed_card,
        analyst_card,
        _cap_card("spend-today", "Claude spend today", float(row["spent"]), cap_today,
                  "What reading the news cost today (a new day starts at midnight UTC), against the daily cap."),
        _cap_card("spend-month", "Claude spend this month", haiku.month_spend(conn, now), cap_month,
                  "Everything Claude cost this month, all features together, against the monthly cap."),
        {"key": "articles", "label": "Articles today", "value": f"{int(row['today']):,}", "tone": "plain",
         "description": "Headlines saved today that name a stock or coin we trade.", "note": ""},
        {"key": "waiting", "label": "Waiting for analysis", "value": f"{int(row['waiting']):,}",
         "tone": "warn" if int(row["waiting"]) else "plain",
         "description": "Articles saved but not yet read by Claude.", "note": ""},
    ]


# ------------------------------------------------------------------ scores


def _score_cards(s: dict[str, Any], market: str) -> list[dict[str, Any]]:
    n = s["n"]
    none = n == 0
    edge = s["edge_pct"]
    wobble = s["edge_se_pct"]
    if edge is None:
        edge_text = "-"
    else:
        edge_text = pct(edge, signed=True, digits=2) + (f" ± {pct(wobble, digits=2)}" if wobble is not None else "")
    return [
        {"key": "hit-rate", "label": "How often Claude was right", "value": share(s["hit_rate"]), "tone": "plain",
         "description": f"A call is right when the price moved the way Claude said, compared with {MARKET_BENCH[market]}. "
                        "Luck alone gives about 50%.",
         "note": "No scored predictions yet." if none else ""},
        {"key": "result", "label": "Average result after costs", "value": pct(s["mean_signed_pct"], signed=True, digits=2),
         "tone": _tone(s["mean_signed_pct"]),
         "description": "What each call would have made on average if traded, after slippage and fees.", "note": ""},
        {"key": "twin", "label": "Coin-flip twin's average", "value": pct(s["mean_twin_pct"], signed=True, digits=2),
         "tone": "muted",
         "description": "The coin-flip twin makes the same trades but picks up or down at random; Claude has to beat it.",
         "note": ""},
        {"key": "edge", "label": "Edge over the coin flip", "value": edge_text,
         "tone": "gain" if s["beats_twin"] else "plain",
         "description": f"Claude's result minus the twin's, both measured against {MARKET_BENCH[market]} so "
                        "riding the market earns nothing, plus or minus the wobble luck alone would cause. "
                        "It is real only when it is more than twice the wobble.",
         "note": "Beats the twin." if s["beats_twin"] else ""},
        {"key": "scored", "label": "Predictions scored", "value": f"{n:,}", "tone": "plain",
         "description": "Checked against what the price really did.",
         "note": f"{s['pending']:,} waiting for their time to pass · {s['none_count']:,} where Claude saw nothing to trade"
                 + (f" · {s['fallback_excluded']:,} replay{'s' if s['fallback_excluded'] != 1 else ''} left out"
                    " (answered by a fallback model)"
                    if s.get("fallback_excluded") else "")},
    ]


def _detail_rows(s: dict[str, Any]) -> dict[str, Any]:
    directions = []
    for key, word in (("up", "Up calls"), ("down", "Down calls")):
        d = s["by_direction"][key]
        directions.append({"label": word, "n": d["n"], "hit": share(d["hit_rate"]),
                           "result": pct(d["mean_signed_pct"], signed=True, digits=2)})
    calibration = [{"label": c["label"], "n": c["n"], "hit": share(c["hit_rate"])} for c in s["calibration"]]
    return {"directions": directions, "calibration": calibration}


def scores(conn: psycopg.Connection, limits: Any) -> list[dict[str, Any]]:
    out = []
    for market in MARKETS:
        s = news_scoring.stats(conn, market=market)
        ok, reason = news_scoring.gate(conn, market, limits)
        live = news_scoring.stats(conn, market=market, source="live")["n"]
        replay = news_scoring.stats(conn, market=market, source="replay")["n"]
        out.append({"market": market, "name": MARKET_NAME[market], "cards": _score_cards(s, market),
                    "gate_ok": ok, "gate_reason": reason, "live": live, "replay": replay, "detail": _detail_rows(s),
                    # The gate counts only the calls the strategy would trade, one per symbol
                    # and day, so its numbers can differ from the cards below, which count
                    # every call.
                    "gate_basis": gate_basis(market, limits)})
    return out


def gate_basis(market: str, limits: Any) -> str:
    """What the gate line counts, in plain words (news_scoring.gate and would_trade)."""
    cost = news_scoring.round_trip_cost_pct(market)
    which = "never SPY or QQQ" if market == "stocks" else "rises only, since coins cannot be shorted"
    return (f"Counted on the calls it would trade (confidence {limits.min_confidence:g} or more, not already "
            f"priced in, an expected move of at least {limits.cost_multiple:g} times the {cost:.2f}% round-trip "
            f"cost, {which}), one per symbol and day, against {MARKET_BENCH[market]}. Replays answered by a "
            "different model than the one asked never count.")


# ------------------------------------------------------------------ paper trading


def _position_row(row: dict[str, Any]) -> dict[str, Any]:
    """One open position. news_trading.open_positions gives "price", "value" (negative for a
    short: shown as a plain amount, the Bet column says which way) and "open_pnl"."""
    side, qty, entry = row.get("side"), float(row.get("qty") or 0.0), row.get("entry_price")
    price, value, change = row.get("price"), row.get("value"), row.get("open_pnl")
    return {
        "symbol": row.get("symbol", ""), "market": MARKET_NAME.get(row.get("market") or "", row.get("market") or ""),
        "side": "Short" if side == "short" else "Long", "qty": f"{qty:,.4g}",
        "entry": money(entry) if entry is not None else "-",
        "now": money(price) if price is not None else "-",
        "value": money(abs(value)) if value is not None else "-",
        "change": money(change, signed=True) if change is not None else "-", "tone": _tone(change),
        "stop": money(row.get("stop_price")) if row.get("stop_price") is not None else "-",
        "due": when(row.get("exit_due_at")), "opened": when(row.get("opened_at")),
        "headline": row.get("headline") or "", "status": row.get("status", ""),
    }


def _book_cards(b: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"key": "equity", "label": "Book value", "value": money(b.get("equity")),
         "description": "Cash plus what the open positions are worth now."},
        {"key": "cash", "label": "Cash", "value": money(b.get("cash")), "description": "Money not tied up in a position."},
        {"key": "gross", "label": "In positions", "value": money(b.get("gross_exposure")),
         "description": "Everything held, longs plus shorts, counted as a positive number."},
        {"key": "short", "label": "Of which shorts", "value": money(b.get("short_exposure")),
         "description": "Bets that a price will fall."},
        {"key": "positions", "label": "Open positions", "value": f"{int(b.get('open_positions') or 0):,}",
         "description": "Trades running right now."},
        {"key": "trades", "label": "Trades today", "value": f"{int(b.get('trades_today') or 0):,}",
         "description": "New positions opened today; there is a daily limit."},
    ]


def paper(conn: psycopg.Connection, score_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Both news books and the open positions. `available` is False until news_trading exists."""
    try:
        from coordinator import news_trading
    except ImportError:
        return {"available": False, "books": [], "positions": []}
    summary = news_trading.book_summary(conn)
    positions = [_position_row(r) for r in news_trading.open_positions(conn)]
    gates = {r["market"]: r for r in score_rows}
    books = []
    for market in MARKETS:
        b = summary.get(market) or {}
        status = b.get("status") or "not_started"
        gate = gates[market]
        running = status in ("active", "closing")
        if running:
            start_reason = "Already running." if status == "active" else "Still selling its positions."
        else:
            start_reason = "" if gate["gate_ok"] else gate["gate_reason"]
        books.append({"market": market, "name": MARKET_NAME[market], "note": MARKET_NOTE[market], "status": status,
                      "status_text": BOOK_TEXT.get(status, status), "cards": _book_cards(b),
                      "waiting": b.get("waiting") or "", "start_disabled": bool(start_reason),
                      "start_reason": start_reason, "stop_disabled": status != "active"})
    return {"available": True, "books": books, "positions": positions}


# ------------------------------------------------------------------ feed


def _call(p: dict[str, Any]) -> dict[str, Any]:
    arrow, word = DIRECTION[p["direction"]]
    call: dict[str, Any] = {"symbol": p["symbol"], "arrow": arrow, "word": word, "direction": p["direction"],
                            "source": p["source"], "rationale": p["rationale"], "priced_in": bool(p["already_priced_in"]),
                            "id": p["id"]}
    if p["direction"] != "none":
        days = int(p["horizon_days"])
        call["move"] = f"about {p['expected_move_pct']:.1f}% in {days} day{'s' if days != 1 else ''}"
        call["confidence"] = f"{p['confidence'] * 100:.0f}% sure"
    score = None
    if p["scored_at"] is not None and p["signed_pct"] is not None:
        score = {"result": pct(p["signed_pct"], signed=True, digits=2), "tone": _tone(p["signed_pct"]),
                 "hit": "Right" if p["hit"] else "Wrong",
                 "twin": pct(p["twin_signed_pct"], signed=True, digits=2) if p["twin_signed_pct"] is not None else "-"}
    call["score"] = score
    call["score_note"] = p["score_note"] or "" if score is None and p["direction"] != "none" else ""
    return call


def feed(conn: psycopg.Connection, now: datetime) -> list[dict[str, Any]]:
    articles = conn.execute("SELECT * FROM news ORDER BY created_at DESC, id DESC LIMIT %s", (FEED_LIMIT,)).fetchall()
    calls: dict[int, list[dict[str, Any]]] = {}
    if articles:
        for p in conn.execute("SELECT * FROM news_predictions WHERE news_id = ANY(%s) ORDER BY news_id, symbol, id",
                              ([a["id"] for a in articles],)).fetchall():
            calls.setdefault(p["news_id"], []).append(_call(p))
    out = []
    for a in articles:
        out.append({"id": a["id"], "headline": a["headline"], "url": safe_url(a["url"]), "source": a["source"],
                    "time": when(a["created_at"]), "ago": ago((now - a["created_at"]).total_seconds()),
                    "symbols": list(a["ours"] or []), "status": a["analysis_status"],
                    "status_text": ANALYSIS_TEXT.get(a["analysis_status"], a["analysis_status"]),
                    "note": a["analysis_note"] or "", "calls": calls.get(a["id"], [])})
    return out


# ------------------------------------------------------------------ the page


def news_page(conn: psycopg.Connection, request_state: Any, broker_status: Any, limits: Any,
              now: datetime | None = None) -> dict[str, Any]:
    """Everything the News screen and its fragment show. `request_state` is app.state: the
    feed and the analyst are read defensively (None shows "Not set up")."""
    now = now or datetime.now(timezone.utc)
    ai = request_state.ai
    score_rows = scores(conn, limits)
    return {
        "header": header(conn, broker_status, now),
        "enabled": bool(getattr(ai, "news_enabled", True)),
        "status": status_cards(conn, getattr(request_state, "news_feed", None), getattr(request_state, "analyst", None), ai, now),
        "scores": score_rows,
        "paper": paper(conn, score_rows),
        "feed": feed(conn, now),
        "demo": bool(getattr(broker_status.broker, "fake", False)),
    }
