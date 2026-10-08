"""Everything the Fleet screen and the shared header show, as plain dicts.

The dashboard templates only format what these functions return, so the wording of
every number lives here in one place and the tests can check it without a browser.
Money is shown with a plus or minus sign, never colour alone.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import psycopg

from coordinator.broker import LIVE, BrokerStatus
from coordinator.limits import Limits
from coordinator.models import list_models
from coordinator.scheduling import can_run_again
from coordinator.settings import get_int_setting, get_setting

TZ = ZoneInfo("America/New_York")
JOB_LABELS = {
    "sleep": "Test job",
    "data_refresh": "Data refresh",
    "backtest": "Backtest",
    "paper_trade": "Paper trade",
    "model_search": "Model search",
    "final_check": "Final check",
}
JOB_CHOICES = (
    ("backtest", "Backtest", "Test one model on past prices and save its results."),
    ("paper_trade", "Paper trade", "Run one model live against the paper account."),
    ("model_search", "Model search", "Invent new models, test them, keep the good ones, repeat until stopped."),
    ("data_refresh", "Data refresh", "Download the latest prices to the coordinator."),
)
NEEDS_MODEL = ("backtest", "paper_trade")
# Job kinds the workers can run in this build; the others are listed but not offered yet.
AVAILABLE_KINDS: tuple[str, ...] = ("sleep", "data_refresh", "backtest", "paper_trade", "model_search")
PAUSED_BANNER = "All trading is paused. No model will place orders until you resume. Backtests keep running."


# ------------------------------------------------------------------ formatting


def money(value: float | None, signed: bool = False) -> str:
    """12345.6 -> "$12,345.60"; signed: "+$12.00" / "-$12.00" (a real minus sign)."""
    if value is None:
        return "-"
    text = f"${abs(value):,.2f}"
    if not signed:
        return ("-" if value < 0 else "") + text
    return ("+" if value >= 0 else "−") + text


def pct(value: float | None, signed: bool = False, digits: int = 1) -> str:
    """2.345 -> "2.3%"; signed: "+2.3%" / "−2.3%"."""
    if value is None:
        return "-"
    text = f"{abs(value):.{digits}f}%"
    if not signed:
        return ("-" if value < 0 else "") + text
    return ("+" if value >= 0 else "−") + text


def ago(seconds: float | None) -> str:
    """Seconds -> "12 s ago" / "3 min ago" / "4 h ago" / "2 days ago" / "never"."""
    if seconds is None:
        return "never"
    s = max(0, int(seconds))
    if s < 60:
        return f"{s} s ago"
    if s < 3600:
        return f"{s // 60} min ago"
    if s < 86400:
        return f"{s // 3600} h ago"
    days = s // 86400
    return f"{days} day{'s' if days != 1 else ''} ago"


def duration(seconds: float | None) -> str:
    """Seconds -> "45 s" / "3 min 20 s" / "2 h 5 min"."""
    if seconds is None:
        return "-"
    s = max(0, int(round(seconds)))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60} s" if s % 60 else f"{s // 60} min"
    return f"{s // 3600} h {(s % 3600) // 60} min"


def clock_time(when: datetime) -> str:
    """A time of day in New York time: "4:00 PM ET"."""
    local = when.astimezone(TZ)
    return local.strftime("%I:%M %p").lstrip("0") + " ET"


def short_when(when: datetime | None, now: datetime) -> str:
    """Today: "3:42 PM"; this week: "Tue 3:42 PM"; older: "Oct 3"."""
    if when is None:
        return "-"
    local, today = when.astimezone(TZ), now.astimezone(TZ)
    hm = local.strftime("%I:%M %p").lstrip("0")
    if local.date() == today.date():
        return hm
    if (today.date() - local.date()).days < 7:
        return local.strftime("%a ") + hm
    return local.strftime("%b ") + str(local.day)


# ------------------------------------------------------------------ header


def header(conn: psycopg.Connection, status: BrokerStatus, now: datetime) -> dict[str, Any]:
    """Pills (mode, stocks, crypto), the pause switch and its banner."""
    paused = get_setting(conn, "trading_paused", False) is True
    reason = get_setting(conn, "paused_reason", None)
    clock = status.clock_info
    if clock is None:
        stocks = {"text": "Stocks: clock unavailable", "state": "unknown"}
    elif clock.is_open:
        stocks = {"text": f"Stocks open · closes {clock_time(clock.next_close)}", "state": "open"}
    else:
        opens = clock.next_open.astimezone(TZ)
        day = "today" if opens.date() == now.astimezone(TZ).date() else opens.strftime("%a")
        stocks = {"text": f"Stocks closed · opens {day} {clock_time(clock.next_open)}", "state": "closed"}
    live = status.broker.mode == LIVE
    return {
        "mode": {"text": "Live" if live else "Paper", "state": "live" if live else "paper"},
        "stocks": stocks,
        "crypto": {"text": "Crypto 24/7", "state": "open"},
        "paused": paused,
        "pause_button": "Resume trading" if paused else "Pause all trading",
        "banner": PAUSED_BANNER if paused else None,
        "paused_reason": reason if paused else None,
        "demo": bool(getattr(status.broker, "fake", False)),
    }


# ------------------------------------------------------------------ tiles


def tiles(conn: psycopg.Connection, status: BrokerStatus, limits: Limits, workers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The four summary tiles, each with a one-line explanation under the number."""
    account = status.account_info
    demo = " (demo data)" if getattr(status.broker, "fake", False) else ""
    if account is None:
        why = status.error or status.broker.problem or "Alpaca has not answered yet"
        value_tile = {"key": "account", "label": "Account value", "value": "-", "note": why, "tone": "muted"}
        pnl_tile = {"key": "pnl", "label": "Today's profit/loss", "value": "-",
                    "note": f"Trading pauses if the account falls {pct(limits.daily_loss_limit_pct)} in a day", "tone": "muted"}
    else:
        mode = "live" if status.broker.mode == LIVE else "paper"
        value_tile = {"key": "account", "label": "Account value", "value": money(account.equity),
                      "note": f"Everything in the Alpaca {mode} account, cash and positions{demo}", "tone": "plain"}
        limit_dollars = account.last_equity * limits.daily_loss_limit_pct / 100.0
        change = account.day_change
        pnl_tile = {
            "key": "pnl", "label": "Today's profit/loss",
            "value": f"{money(change, signed=True)} ({pct(account.day_change_pct, signed=True, digits=2)})",
            "note": f"Daily loss limit {pct(-limits.daily_loss_limit_pct, signed=True)} ({money(-limit_dollars, signed=True)}): "
                    "trading pauses itself past it",
            "tone": "gain" if change > 0 else "loss" if change < 0 else "plain",
        }
    online = [w for w in workers if w["online"]]
    offline = [w["name"] for w in workers if not w["online"]]
    hot = [w["name"] for w in online if w["hot"]]
    notes = []
    if offline:
        notes.append("Offline: " + ", ".join(offline))
    if hot:
        notes.append("Hot: " + ", ".join(hot))
    workers_tile = {
        "key": "workers", "label": "Workers online", "value": f"{len(online)} of {len(workers)}",
        "note": " · ".join(notes) if notes else ("All workers online" if workers else "No workers enrolled yet"),
        "tone": "warn" if offline or hot else "plain",
    }
    counts = conn.execute(
        "SELECT count(*) FILTER (WHERE status IN ('leased','cancel_requested')) AS running,"
        " count(*) FILTER (WHERE status = 'queued') AS queued FROM jobs"
    ).fetchone()
    jobs_tile = {
        "key": "jobs", "label": "Jobs running", "value": str(counts["running"]),
        "note": f"{counts['queued']} queued, waiting for a free worker" if counts["queued"] else "Nothing queued",
        "tone": "plain",
    }
    return [value_tile, pnl_tile, workers_tile, jobs_tile]


# ------------------------------------------------------------------ workers


def _progress(job: dict[str, Any]) -> dict[str, Any]:
    if job["kind"] == "paper_trade" or job["progress"] is None:
        return {"text": "Live", "pct": None}
    p = max(0.0, min(1.0, float(job["progress"] or 0)))
    return {"text": f"{int(p * 100)}%", "pct": round(p * 100, 1)}


def workers(conn: psycopg.Connection, limits: Limits, now: datetime) -> list[dict[str, Any]]:
    """One card per worker, sorted by name (w1, w2, ... w10 in number order)."""
    online_after = get_int_setting(conn, "online_after_seconds", 20)
    rows = conn.execute(
        """
        SELECT w.*, EXTRACT(EPOCH FROM (now() - w.last_heartbeat_at)) AS silent_s,
               j.id AS job_id, j.kind AS job_kind, j.progress AS job_progress, j.detail AS job_detail,
               m.name AS job_model_name, j.status AS job_status
          FROM workers w
          LEFT JOIN LATERAL (
            SELECT * FROM jobs WHERE lease_worker_id = w.id AND status IN ('leased', 'cancel_requested')
             ORDER BY started_at DESC LIMIT 1) j ON true
          LEFT JOIN models m ON m.id = j.model_id
        """
    ).fetchall()
    cards = []
    for r in rows:
        silent = r["silent_s"]
        online = silent is not None and float(silent) <= online_after
        temp = r["temp_c"]
        hot = online and temp is not None and float(temp) >= limits.hot_temp_c
        card: dict[str, Any] = {
            "id": r["id"], "name": r["name"], "online": online, "enabled": r["enabled"], "hot": hot,
            "temp": {"text": f"{float(temp):.0f}°C" if temp is not None else "no sensor", "hot": hot,
                     "known": temp is not None},
            "cpu_pct": round(float(r["cpu_pct"]), 1) if online and r["cpu_pct"] is not None else None,
            "ram_pct": round(float(r["ram_pct"]), 1) if online and r["ram_pct"] is not None else None,
        }
        if not online:
            card.update(state="offline", dot="offline", task="Offline",
                        detail=f"Last seen {ago(float(silent)) if silent is not None else 'never'}", progress=None)
        elif r["job_id"] is None:
            card.update(state="idle", dot="hot" if hot else "idle", task="Idle", detail="No job assigned", progress=None)
        else:
            job = {"kind": r["job_kind"], "progress": r["job_progress"]}
            task = JOB_LABELS.get(r["job_kind"], r["job_kind"])
            if r["job_model_name"]:
                task += f" · {r['job_model_name']}"
            detail = r["job_detail"] or ("Stopping" if r["job_status"] == "cancel_requested" else "Starting")
            card.update(state="busy", dot="hot" if hot else "busy", task=task, detail=detail,
                        progress=_progress(job), job_id=str(r["job_id"]))
        cards.append(card)
    cards.sort(key=lambda c: _natural(c["name"]))
    return cards


def _natural(name: str) -> tuple[Any, ...]:
    """w2 before w10."""
    head = name.rstrip("0123456789")
    tail = name[len(head):]
    return (head, int(tail) if tail else -1, name)


# ------------------------------------------------------------------ jobs


def _result_text(row: dict[str, Any]) -> str:
    result = row["result"] if isinstance(row["result"], dict) else {}
    if row["status"] == "failed":
        return row["error"] or "Failed"
    if row["status"] == "cancelled":
        return "Cancelled before it finished"
    return str(result.get("summary") or "Done")


def previous_jobs(conn: psycopg.Connection, now: datetime, limit: int = 50) -> list[dict[str, Any]]:
    """Finished jobs, newest first: Job, Model, Worker, Finished, Took, Result, Status."""
    rows = conn.execute(
        """
        SELECT j.*, w.name AS worker_name, m.name AS model_name,
               EXTRACT(EPOCH FROM (j.finished_at - COALESCE(j.started_at, j.created_at))) AS took_s
          FROM jobs j LEFT JOIN workers w ON w.id = j.last_worker_id
          LEFT JOIN models m ON m.id = j.model_id
         WHERE j.status IN ('succeeded', 'failed', 'cancelled')
         ORDER BY j.finished_at DESC NULLS LAST LIMIT %s
        """,
        (limit,),
    ).fetchall()
    out = []
    for r in rows:
        status = {"succeeded": "Succeeded", "failed": "Failed", "cancelled": "Cancelled"}[r["status"]]
        out.append({
            "id": str(r["id"]),
            "job": JOB_LABELS.get(r["kind"], r["kind"]),
            "model": r["model_name"] or "-",
            "worker": r["worker_name"] or "-",
            "finished": short_when(r["finished_at"], now),
            "took": duration(float(r["took_s"])) if r["took_s"] is not None and r["started_at"] else "-",
            "result": _result_text(r),
            "status": status,
            "state": r["status"],
            "can_run_again": can_run_again(r),
        })
    return out


def assign_options(conn: psycopg.Connection, worker_cards: list[dict[str, Any]], models: list[dict[str, Any]]) -> dict[str, Any]:
    """The three dropdowns of the "Assign a job" panel."""
    idle = [w for w in worker_cards if w["state"] == "idle" and w["enabled"]]
    return {
        "jobs": [{"kind": k, "label": label, "help": help_text, "needs_model": k in NEEDS_MODEL,
                  "available": k in AVAILABLE_KINDS} for k, label, help_text in JOB_CHOICES],
        "models": models,
        "workers": [{"value": "auto", "label": "Auto — pick the least busy"}]
        + [{"value": w["id"], "label": w["name"]} for w in idle]
        + [{"value": "all_idle", "label": "All idle workers"}],
    }


def fleet_page(conn: psycopg.Connection, status: BrokerStatus, limits: Limits, now: datetime | None = None) -> dict[str, Any]:
    """Everything the Fleet screen needs in one dict (also served as JSON at /api/fleet)."""
    now = now or datetime.now(timezone.utc)
    cards = workers(conn, limits, now)
    return {
        "header": header(conn, status, now),
        "tiles": tiles(conn, status, limits, cards),
        "workers": cards,
        "assign": assign_options(conn, cards, [
            {"id": m["id"], "name": m["name"], "market": m["market"]} for m in list_models(conn)]),
        "previous_jobs": previous_jobs(conn, now),
        "server_time": now.isoformat(),
    }
