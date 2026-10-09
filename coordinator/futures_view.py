"""The Models screen's Futures view, as plain dicts (the templates only format them).

Ranking: futures models are ranked by expected net dollars per Combine attempt on the
held-out period, at their best contract size. Only a model that beats its coin-flip
twin (a higher pass rate and more money per attempt) gets a rank, and only while the
fees and the payout cap are set in config/topstep.toml; until then nothing is ranked
and the screen says why. A model tested under other Topstep rules than the current
file's is not ranked until it is backtested again.

"Ready for a Combine" needs every item of the checklist: real (Databento) prices, the
held-out tests, the once-only lockbox Final check, and 20 days of shadow trading on
live prices. Shadow trading is not built yet, so no model is ready in this build, and a
model tested on proxy prices can never be.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import psycopg

from coordinator import futures_data, futures_models, futures_trading, safety
from coordinator.topstep_broker import CONFIRM_PHRASE, NOT_SET
from coordinator.models import STATUS_TEXT, list_models
from coordinator.models_view import SPARK_POINTS, hold_text
from fleet2.models.futures import tries_bucket
from fleet2.sim import futures_stats, topstep
from fleet2.sim.cme_session import as_date

FEED_TAG = {"databento": "Databento prices", "proxy": "Proxy prices", "synthetic": "Synthetic prices"}
PAPER_NOTE = ("A futures model first paper trades on Alpaca (SPY or QQQ shares standing in for its contracts). "
              "Only once its checklist is complete can it trade on Topstep.")


def dollars(value: float | None, signed: bool = True) -> str:
    if value is None:
        return "-"
    text = f"${abs(value):,.0f}"
    return (("+" if value >= 0 else "−") + text) if signed else text


def share(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def day_text(yyyymmdd: int | None) -> str:
    if not yyyymmdd:
        return "-"
    d = as_date(int(yyyymmdd))
    return f"{d:%b} {d.day}, {d.year}"


def _epoch(yyyymmdd: int) -> int:
    d = as_date(int(yyyymmdd))
    return int((date(d.year, d.month, d.day) - date(1970, 1, 1)).days * 86400)


def chosen(held: dict[str, Any], rules: topstep.Rules) -> dict[str, Any]:
    """The size whose numbers are shown: the best one when the fees are set, else 1."""
    best = topstep.best_size(held, rules)
    if best is not None:
        return best
    first = (held.get("sizes") or [{}])[0]
    return {**first, "net": None, "twin_net": None}


def beats_twin(entry: dict[str, Any], rules: topstep.Rules) -> bool | None:
    """Passes more often than the coin-flip twin, and (when it can be counted) makes more
    money per attempt. None when the period was too short for any attempt."""
    mine, twin = (entry.get("sim") or {}).get("pass_rate"), (entry.get("twin") or {}).get("pass_rate")
    if mine is None or twin is None:
        return None
    if mine <= twin:
        return False
    net, twin_net = topstep.net_per_attempt(entry["sim"], rules), topstep.net_per_attempt(entry["twin"], rules)
    return True if net is None or twin_net is None else net > twin_net


def luck(m: dict[str, Any], tries: dict[str, dict[str, float]]) -> dict[str, Any]:
    """The "chance this is luck" figure and how many tries it counts."""
    train = (m.get("metrics") or {}).get("train") or {}
    t = tries.get(tries_bucket(m["module"])) or {"n": 0, "sum": 0.0, "sq": 0.0}
    found = m["origin"] == "search"
    trials = max(1, int(t["n"])) if found else 1
    variance = futures_stats.variance_from_sums(int(t["n"]), t["sum"], t["sq"]) if found else 0.0
    if not train.get("n_days"):
        return {"value": None, "trials": trials}
    value = futures_stats.chance_of_luck(float(train.get("sr_day") or 0.0), int(train["n_days"]),
                                         float(train.get("skew") or 0.0), float(train.get("kurt") or 3.0),
                                         trials, variance)
    return {"value": value, "trials": trials}


def _status(m: dict[str, Any], rules: topstep.Rules) -> dict[str, Any]:
    """Everything the list and the detail share about one model's held-out results."""
    held = (m.get("metrics") or {}).get("held_out")
    out: dict[str, Any] = {"tested": bool(held), "stale": False, "pick": None, "beats": None, "net": None}
    if not held:
        return out
    out["stale"] = held.get("sim_key") != rules.sim_key()
    pick = chosen(held, rules)
    out.update(pick=pick, beats=beats_twin(pick, rules), net=pick.get("net"))
    return out


def list_row(m: dict[str, Any], rules: topstep.Rules) -> dict[str, Any]:
    st = _status(m, rules)
    metrics = m.get("metrics") or {}
    held = metrics.get("held_out") or {}
    pick = st["pick"] or {}
    sim, twin = pick.get("sim") or {}, pick.get("twin") or {}
    curve = ((held.get("numbers") or {}).get("curve") or {}).get("pnl") or []
    step = max(1, len(curve) // SPARK_POINTS)
    retired = m["status"] == "retired"
    note = topstep.money_note(rules)
    if not st["tested"]:
        money, tone = "-", "plain"
    elif st["net"] is not None:
        money, tone = dollars(st["net"]), "gain" if st["net"] > 0 else "loss" if st["net"] < 0 else "plain"
    else:
        money, tone = "Set the fee", "warn"
    rankable = bool(st["tested"] and not st["stale"] and not retired and st["beats"] and st["net"] is not None)
    feed = metrics.get("feed") or held.get("feed")
    return {
        "id": m["id"], "name": m["name"], "description": m["description"], "market": "Futures",
        "status": STATUS_TEXT[m["status"]], "status_key": m["status"] or "new", "origin": m["origin"],
        "money": money, "money_value": st["net"], "tone": tone, "money_note": note,
        "pass_line": (f"Passes {share(sim.get('pass_rate'))} · coin flip {share(twin.get('pass_rate'))}"
                      if st["tested"] else "Not tested yet"),
        "beats": st["beats"], "rankable": rankable, "retired": retired, "stale": st["stale"], "tested": st["tested"],
        "feed_tag": FEED_TAG.get(feed) if feed else None, "proxy": feed in ("proxy", "synthetic"),
        "spark": curve[::step][:SPARK_POINTS] if curve else [],
    }


def ranked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ranked models first, by expected net per attempt; then, unranked: models that beat
    their twin (fees unset), those that do not, tested under old rules, untested, retired."""
    def key(r: dict[str, Any]) -> tuple[int, float, str]:
        if r["rankable"]:
            group = 0
        elif r["retired"]:
            group = 5
        elif not r["tested"]:
            group = 4
        elif r["stale"]:
            group = 3
        else:
            group = 1 if r["beats"] else 2
        return group, -(r["money_value"] or 0.0), r["name"]

    out = sorted(rows, key=key)
    rank = 0
    for r in out:
        if r["rankable"]:
            rank += 1
            r["rank"] = rank
        else:
            r["rank"] = None
    return out


def _card(key: str, label: str, value: str, description: str, note: str | None = None, tone: str = "plain") -> dict[str, Any]:
    return {"key": key, "label": label, "value": value, "description": description, "note": note, "tone": tone}


def cards(m: dict[str, Any], rules: topstep.Rules, pick: dict[str, Any], tries: dict[str, Any]) -> list[dict[str, Any]]:
    """The numbers of the detail panel, each with a plain-words explanation."""
    held = m["metrics"]["held_out"]
    numbers = held.get("numbers") or {}
    sim, twin = pick.get("sim") or {}, pick.get("twin") or {}
    note = topstep.money_note(rules)
    split = rules.profit_split
    payout = None if sim.get("mean_paid") is None else split * sim["mean_paid"]
    twin_payout = None if twin.get("mean_paid") is None else split * twin["mean_paid"]
    beat = (sim.get("pass_rate") or 0) > (twin.get("pass_rate") or 0)
    lk = luck(m, tries)
    out = [
        _card("pass_rate", "Pass rate", share(sim.get("pass_rate")),
              "Share of Combine attempts on the held-out prices that reached the profit target without touching "
              "the loss floor. One attempt starts on each day; one still going after 60 trading days is given up.",
              f"Coin-flip twin: {share(twin.get('pass_rate'))} · {sim.get('attempts', 0)} attempts",
              "gain" if beat else "loss"),
        _card("days_to_pass", "Median days to pass",
              "-" if sim.get("median_days_to_pass") is None else f"{sim['median_days_to_pass']:.0f} days",
              "Trading days from the start of an attempt to passing, for the attempts that passed.",
              None if twin.get("median_days_to_pass") is None else f"Coin-flip twin: {twin['median_days_to_pass']:.0f} days"),
        _card("payout", "Expected payout", dollars(payout, signed=False) if payout is not None else "Set the payout cap",
              "Your share of the Express Funded payouts, averaged over every attempt, including the many that "
              "never get there.",
              f"Coin-flip twin: {dollars(twin_payout, signed=False)}" if twin_payout is not None else sim.get("payout_note"),
              "plain" if payout is not None else "warn"),
        _card("net", "Expected net per attempt", dollars(pick.get("net")) if pick.get("net") is not None else note or "-",
              "The expected payout minus the monthly Combine fees and the activation fee. Below zero means paying "
              "for Combines would lose money on average.",
              f"Coin-flip twin: {dollars(pick.get('twin_net'))}" if pick.get("twin_net") is not None else None,
              ("gain" if pick["net"] > 0 else "loss") if pick.get("net") is not None else "warn"),
        _card("contracts", "Contract size", f"{pick.get('contracts', 1)} micro{'s' if pick.get('contracts', 1) != 1 else ''}",
              "How many micro contracts the numbers above are for: the size with the most money per attempt, of "
              "the sizes whose worst training stretch stays within half the loss limit.",
              ("Even one contract's worst stretch uses more than half the loss limit" if held.get("too_risky")
               else f"Tried 1 to {held.get('max_size', 1)}") if pick.get("net") is not None
              else f"{note}: until then the numbers are for 1",
              "warn" if held.get("too_risky") or pick.get("net") is None else "plain"),
        _card("worst_day", "Worst day", dollars(numbers.get("worst_day")),
              "The biggest loss of one held-out day at one contract. The note gives the worst moment of any day, "
              "open trades included: that is what the loss limit watches.",
              f"Worst moment: {dollars(numbers.get('worst_dip'))}", "loss" if (numbers.get("worst_day") or 0) < 0 else "plain"),
        _card("best_day_share", "Best-day share", share(numbers.get("best_day_share")),
              "The best day's share of the total profit. Topstep's consistency rule limits it, and a high share means "
              "one lucky day did most of the work.",
              None if numbers.get("best_day_share") is not None else "No profit to share"),
        _card("trades_per_day", "Trades per day",
              "-" if numbers.get("trades_per_day") is None else f"{numbers['trades_per_day']:.1f}",
              "Average trades on the days the model traded.",
              f"Traded on {numbers.get('days_traded', 0)} of {numbers.get('days', 0)} days"),
        _card("avg_hold", "Average hold",
              hold_text(None if numbers.get("avg_hold_minutes") is None else numbers["avg_hold_minutes"] * 60),
              "How long a trade usually stays open. Every trade is closed by the end of the day."),
        _card("double", "At double slippage", dollars(held.get("double_slippage_pnl")),
              "Held-out profit at one contract with twice the slippage. A model that only works with perfect fills "
              "is not ready.", f"At normal slippage: {dollars(numbers.get('net_pnl'))}",
              "gain" if (held.get("double_slippage_pnl") or 0) > 0 else "loss"),
        _card("luck", "Chance this is luck", share(lk["value"]),
              "How likely a training result this good would turn up from settings with no real edge, given how many "
              "settings model search tried for this model file (the deflated Sharpe ratio). Lower is better; "
              "around 50% is a coin toss.",
              f"Counting {lk['trials']:,} setting{'s' if lk['trials'] != 1 else ''} tried",
              "plain" if lk["value"] is None else "gain" if lk["value"] < 0.05 else "warn" if lk["value"] < 0.5 else "loss"),
    ]
    return out


def verdict(m: dict[str, Any], rules: topstep.Rules, pick: dict[str, Any], check: dict[str, Any] | None,
            paper: dict[str, Any] | None = None) -> dict[str, Any]:
    """The "ready for a Combine" checklist. Ready only when every item holds. `paper` is
    the model's Alpaca paper record (futures_trading.paper_record)."""
    held = m["metrics"]["held_out"]
    feed = m["metrics"].get("feed") or held.get("feed")
    sim, twin = pick.get("sim") or {}, pick.get("twin") or {}
    items = []

    def item(label: str, state: str, note: str | None = None) -> None:
        items.append({"label": label, "state": state, "note": note})

    real = feed == "databento"
    item("Real futures prices (Databento)", "ok" if real else "no",
         None if real else f"{FEED_TAG.get(feed, 'These prices')} never count toward a verdict")
    net = pick.get("net")
    item("Held-out: expected net per attempt above zero",
         "unset" if net is None else "ok" if net > 0 else "no", topstep.money_note(rules) if net is None else None)
    edge = None if sim.get("pass_rate") is None or twin.get("pass_rate") is None else sim["pass_rate"] - twin["pass_rate"]
    item(f"Held-out: passes at least {rules.min_pass_rate_edge * 100:.0f} points more often than its coin flip",
         "no" if edge is None or edge < rules.min_pass_rate_edge else "ok",
         None if edge is None else f"{edge * 100:+.0f} points")
    item("Held-out: profitable at double slippage", "ok" if (held.get("double_slippage_pnl") or 0) > 0 else "no")
    if check is None:
        item("Lockbox Final check: profitable, pass rate close to held-out", "pending", "Not run yet")
    else:
        result = check["result"]
        size = int(result.get("contracts") or 1)
        entry = (result.get("sizes") or [{}])[size - 1]
        lock_rate = (entry.get("sim") or {}).get("pass_rate")
        held_rate = sim.get("pass_rate") or 0.0
        ok = (entry.get("net_pnl") or 0) > 0 and lock_rate is not None and lock_rate >= held_rate - rules.max_lockbox_drop
        item("Lockbox Final check: profitable, pass rate close to held-out", "ok" if ok else "no",
             f"Lockbox passes {share(lock_rate)}, held-out {share(sim.get('pass_rate'))}")
    label = (f"Alpaca paper trading: {rules.min_shadow_days}+ days, at most {rules.max_days_outside * 100:.0f}% of them "
             "outside the backtest's range, and a profit")
    if paper is None or (paper["days"] == 0 and paper["outside"] is not None):
        item(label, "pending", "Not started yet")
    else:
        state = futures_trading.paper_ok(paper, rules)
        if paper["outside"] is None:
            note = "Run backtest again: this model was tested before its range of daily results was kept"
        else:
            note = (f"{paper['days']} day{'s' if paper['days'] != 1 else ''}, {dollars(paper['total'])}, "
                    f"{paper['outside']} outside {dollars(paper['low'])} to {dollars(paper['high'])}")
        item(label, state, note)
    if held.get("sim_key") != rules.sim_key():
        item("Tested under the current rules in config/topstep.toml", "no", "Run backtest again")
    ready = real and all(i["state"] == "ok" for i in items)
    first = next((i for i in items if i["state"] != "ok"), None)
    return {"ready": ready, "items": items,
            "text": "Ready for a Combine" if ready else f"Not ready for a Combine. First missing: {first['label']}."}


def _settings_text(params: dict[str, Any]) -> str:
    shown = []
    for k, v in params.items():
        if k == "recipe":  # shown in plain words in the description instead
            continue
        label = k.replace("_", " ")
        shown.append(f"{label} {v:g}" if isinstance(v, float) else f"{label} {v}")
    return " · ".join(shown)


def final_check_view(m: dict[str, Any], check: dict[str, Any] | None, running: bool, st: dict[str, Any],
                     rules: topstep.Rules, feed: str | None) -> dict[str, Any]:
    if check is not None:
        result = check["result"]
        size = int(result.get("contracts") or 1)
        entry = (result.get("sizes") or [{}])[size - 1]
        sim, twin = entry.get("sim") or {}, entry.get("twin") or {}
        when = check["created_at"]
        return {"stored": True, "button": False,
                "text": (f"Final check, kept forever: the lockbox passes {share(sim.get('pass_rate'))} "
                         f"(coin flip {share(twin.get('pass_rate'))}), {dollars(entry.get('net_pnl'))} at {size} "
                         f"contract{'s' if size != 1 else ''}, run {when:%b} {when.day}, {when.year} on "
                         f"{FEED_TAG.get(check['feed'], check['feed']).lower()}.")}
    reason = None
    if running:
        reason = "The Final check is running."
    elif not st["tested"]:
        reason = "Run a backtest first."
    elif st["stale"]:
        reason = "Topstep's rules changed since this model was tested: run a backtest first."
    elif feed in (None, "proxy", "mixed"):
        reason = "The Final check needs real futures prices (a Databento key). Proxy prices never count."
    elif rules.missing_trading_fees():
        reason = f"{topstep.FEE_MESSAGE} first."
    return {"stored": False, "button": True, "disabled": reason is not None, "reason": reason,
            "text": "Opens the lockbox, the last 15% of the prices, once. The result is kept forever and can never be "
                    "run again."}


def paper_of(conn: psycopg.Connection, m: dict[str, Any], rules: topstep.Rules, now: datetime) -> dict[str, Any]:
    """The model's Alpaca paper record, compared with its held-out daily range at the
    contracts it trades."""
    held = (m.get("metrics") or {}).get("held_out") or {}
    book = conn.execute("SELECT contracts FROM futures_books WHERE model_id = %s AND venue = 'alpaca_paper' "
                        "ORDER BY id DESC LIMIT 1", (m["id"],)).fetchone()
    contracts = int(book["contracts"]) if book else 1
    return futures_trading.paper_record(conn, m["id"], held.get("daily_band"), contracts, rules, futures_trading.trading_day(now))


def model_verdict(conn: psycopg.Connection, model_id: str, rules: topstep.Rules,
                  now: datetime | None = None) -> dict[str, Any]:
    """The checklist of one model, as the screen shows it ({"ready": False} when untested)."""
    now = now or datetime.now(timezone.utc)
    m = conn.execute("SELECT * FROM models WHERE id = %s", (model_id,)).fetchone()
    if m is None or m["market"] != "futures" or not (m["metrics"] or {}).get("held_out"):
        return {"ready": False, "items": [], "text": "Not tested yet"}
    pick = chosen(m["metrics"]["held_out"], rules)
    return verdict(m, rules, pick, futures_models.final_check(conn, model_id), paper_of(conn, m, rules, now))


def trading_view(conn: psycopg.Connection, m: dict[str, Any], rules: topstep.Rules, venues: Any, st: dict[str, Any],
                 ready: bool, now: datetime) -> dict[str, Any]:
    """The Alpaca paper and Topstep buttons of one model, and what each book is doing."""
    out: dict[str, Any] = {}
    fees = rules.missing_trading_fees()
    for venue in futures_trading.VENUES:
        book = futures_trading.open_book(conn, m["id"], venue)
        active = book is not None and book["status"] == "active"
        reason = None
        if not active:
            if m["status"] == "retired":
                reason = "This model is retired."
            elif not st["tested"]:
                reason = "Run a backtest first."
            elif fees:
                reason = f"{topstep.FEE_MESSAGE} first."
            elif book is not None:
                reason = "Stopping: closing its position."
            elif venue == "alpaca_paper" and venues is not None:
                broker = venues.alpaca.broker
                if not broker.connected:
                    reason = broker.problem
                elif broker.mode != "paper":
                    reason = "The coordinator is in live mode: futures models only paper trade on Alpaca."
            elif venue == "topstep":
                if not ready:
                    reason = "Only a model ready for a Combine (every line of its checklist ticked) trades on Topstep."
                elif venues is None or not venues.topstep.on:
                    reason = venues.topstep.problem if venues is not None else "Topstep is not connected."
                elif safety.topstep_paused(conn):
                    reason = safety.topstep_paused(conn)
        word = "paper trading on Alpaca" if venue == "alpaca_paper" else "trading on Topstep"
        out[venue] = {
            "active": active,
            "button": ("Stop " if active else "Start ") + word,
            "action": "futures-stop" if active else "futures-start",
            "disabled": reason is not None and not active,
            "reason": reason,
            "book": futures_trading.book_line(conn, book, now) if book else None,
        }
    return out


def detail(m: dict[str, Any], rules: topstep.Rules, tries: dict[str, Any], check: dict[str, Any] | None,
           running: bool, feed: str | None, conn: psycopg.Connection | None = None, venues: Any = None,
           now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    metrics = m.get("metrics") or {}
    st = _status(m, rules)
    model_feed = metrics.get("feed") or (metrics.get("held_out") or {}).get("feed")
    tags = ["Futures", STATUS_TEXT[m["status"]]]
    if model_feed in FEED_TAG:
        tags.append(FEED_TAG[model_feed])
    if st["stale"]:
        tags.append("Rules changed: backtest again")
    origin = "Starter model"
    if m["origin"] == "search":
        origin = f"Found by model search · seed {metrics.get('seed') or '-'}"
        if metrics.get("recipe_by"):
            maker = "Claude Haiku" if metrics["recipe_by"] == "haiku" else "a random mix of building blocks"
            origin += f" · recipe put together by {maker}"
    out: dict[str, Any] = {
        "id": m["id"], "name": m["name"], "tags": tags, "warn_tags": {"Proxy prices", "Synthetic prices",
                                                                      "Rules changed: backtest again"},
        "description": m["description"], "how_it_works": m["how_it_works"], "origin": origin,
        "settings": _settings_text(m["params"] or {}), "retired": m["status"] == "retired",
        "tested": st["tested"], "paper_note": PAPER_NOTE,
        "final_check": final_check_view(m, check, running, st, rules, feed),
    }
    paper = paper_of(conn, m, rules, now) if conn is not None and st["tested"] else None
    ready = False
    if st["tested"]:
        out["verdict"] = verdict(m, rules, st["pick"], check, paper)
        ready = out["verdict"]["ready"]
    if conn is not None:
        out["trading"] = trading_view(conn, m, rules, venues, st, ready, now)
    if not st["tested"]:
        out["untested"] = "Not tested yet. Press Run backtest to see how it would have done under Topstep's rules."
        return out
    held = metrics["held_out"]
    pick = st["pick"]
    numbers = held.get("numbers") or {}
    curve = numbers.get("curve") or {}
    twin_curve = held.get("twin_curve") or {}
    out["chart"] = {
        "title": "Held-out profit and loss, added up",
        "t": [_epoch(d) for d in curve.get("d") or []],
        "model": curve.get("pnl") or [],
        "benchmark": twin_curve.get("pnl") or [],
        "model_label": f"{m['name']}, 1 contract",
        "benchmark_label": "Coin-flip twin",
        "ref": 0.0,
    }
    out["period"] = (f"Held-out period {day_text(numbers.get('start'))} to {day_text(numbers.get('end'))}: "
                     "prices model search never saw")
    train = metrics.get("train") or {}
    if train:
        score = train.get("score")
        out["training_line"] = (f"Training score {score:.2f} (worst of four parts, double slippage)"
                                if score is not None else "Training score -") + \
            f" · traded on {train.get('days_traded', 0):,} days"
        if train.get("neighbour_median") is not None:
            out["training_line"] += f" · settings 10% either way score {train['neighbour_median']:.2f}"
    out["metrics"] = cards(m, rules, pick, tries)
    return out


def topstep_panel(conn: psycopg.Connection, venues: Any) -> dict[str, Any]:
    """What the screen says about the Topstep connection."""
    if venues is None:
        return {"state": "off", "text": NOT_SET, "confirm": False}
    link = venues.topstep
    stopped = safety.topstep_paused(conn)
    if link.client is None:
        return {"state": "off", "text": link.problem or NOT_SET, "confirm": False}
    if not link.on:
        return {"state": "unconfirmed", "text": link.problem, "confirm": True, "phrase": CONFIRM_PHRASE}
    acc = venues.topstep_state or {}
    text = "Topstep: connected" + (" (demo data)" if getattr(link.client, "fake", False) else "")
    if acc.get("account"):
        text += (f" to {acc['account']} · balance ${acc['balance']:,.0f} · loss floor ${acc['floor']:,.0f} · "
                 f"${acc['room']:,.0f} of room")
    elif acc.get("error"):
        text += f" · account check failed: {acc['error']}"
    return {"state": "paused" if stopped else "on", "text": text, "paused": stopped, "confirm": False}


def live_signals(conn: psycopg.Connection, now: datetime) -> list[dict[str, Any]]:
    """Every trading model's latest signal, for trading by hand on TopstepX."""
    out = []
    for book in conn.execute("SELECT b.*, m.name AS model_name FROM futures_books b JOIN models m ON m.id = b.model_id "
                             "WHERE b.status = 'active' ORDER BY m.name").fetchall():
        line = futures_trading.book_line(conn, book, now)
        if line["signal"]:
            out.append({"model": book["model_name"], "venue": futures_trading.VENUE_TEXT[book["venue"]],
                        "signal": line["signal"].removeprefix("Live signal: ")})
    return out


def futures_page(conn: psycopg.Connection, rules: topstep.Rules, selected_id: str | None = None,
                 search: dict[str, Any] | None = None, venues: Any = None,
                 now: datetime | None = None) -> dict[str, Any]:
    """The futures list (ranked), the selected model and the prices and search panel."""
    now = now or datetime.now(timezone.utc)
    models = [m for m in list_models(conn, include_retired=True) if m["market"] == "futures"]
    rows = ranked([list_row(m, rules) for m in models])
    by_id = {m["id"]: m for m in models}
    chosen_id = selected_id if selected_id in by_id else (rows[0]["id"] if rows else None)
    for r in rows:
        r["selected"] = r["id"] == chosen_id
    prices = futures_data.status_line(conn)
    tries = futures_models.tries(conn)
    notice = None
    note = topstep.money_note(rules)
    if rows and not any(r["tested"] for r in rows):
        notice = "No futures model is tested yet: press Run backtest on one, or start a model search."
    elif rows and note:
        notice = f"{note} to rank futures models: they are ranked by expected money per Combine attempt."
    elif rows and not any(r["rankable"] for r in rows):
        notice = "No futures model beats its coin-flip twin on the held-out prices yet, so none is ranked."
    missing = rules.missing_trading_fees()
    total_tries = sum(int(t["n"]) for t in tries.values())
    selected = None
    if chosen_id:
        running = futures_models.final_check_running(conn, chosen_id)
        selected = detail(by_id[chosen_id], rules, tries, futures_models.final_check(conn, chosen_id), running,
                          prices["feed"], conn, venues, now)
    return {
        "view": "futures",
        "notice": notice,
        "models": rows,
        "selected": selected,
        "prices": prices,
        "fee_note": (f"{topstep.FEE_MESSAGE} ({', '.join(missing)}): futures backtests and model search need it."
                     if missing else None),
        "tries_line": (f"{total_tries:,} futures settings tried so far"
                       if total_tries and not (search or {}).get("running") else None),
        "search": search or {"running": False, "button": "Start model search", "detail": None},
        "topstep": topstep_panel(conn, venues),
        "signals": live_signals(conn, now),
    }
