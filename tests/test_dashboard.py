"""The dashboard pages: shared header, Fleet screen, its refreshable fragment, Models placeholder.

Tests look for data-* hooks (not CSS classes) in the rendered HTML."""
from __future__ import annotations

import re
from dataclasses import replace
from html import unescape

from coordinator import fleet_view, queue
from tests.conftest import enroll, heartbeat

BANNER = "All trading is paused. No model will place orders until you resume. Backtests keep running."


def attr_tags(html: str, attr: str) -> list[str]:
    """Every opening tag that carries the data-* attribute `attr`."""
    return re.findall(r"<[^>]*\s" + re.escape(attr) + r"(?![\w-])[^>]*>", html)


def element(html: str, hook: str) -> str:
    """The text content of the first element that has the hook attribute (tags stripped)."""
    m = re.search(r"<(\w+)[^>]*\s" + re.escape(hook) + r"(?![\w-])[^>]*>(.*?)</\1>", html, re.S)
    assert m, f"no element with {hook}"
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m.group(2))).strip()


def text(html: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", html)))


def failed_job(client, conn, worker):
    """A sleep job that fails because its worker went offline; returns the job id."""
    client.post("/api/jobs", json={"kind": "sleep", "params": {"seconds": 60}, "target": worker["worker_id"]})
    job = heartbeat(client, worker, want_job=True)["claimed"][0]
    conn.execute("UPDATE jobs SET lease_expires_at = now() - interval '1 second' WHERE id = %s", (job["id"],))
    queue.reap(conn)
    return job["id"]


def test_home_redirects_to_fleet(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/fleet"


def test_fleet_header_has_nav_pills_and_pause_button(client):
    html = client.get("/fleet").text
    assert "fleet-v2" in html
    assert element(html, 'data-nav="fleet"') == "Fleet" and element(html, 'data-nav="models"') == "Models"
    assert 'data-nav="fleet" class="current" aria-current="page"' in html
    assert 'data-nav="models" class="current"' not in html
    assert element(html, 'data-pill="mode"') == "Paper"
    assert element(html, 'data-pill="stocks"').startswith("Stocks")
    assert element(html, 'data-pill="crypto"') == "Crypto 24/7"
    assert element(html, 'data-action="pause"') == "Pause all trading"
    assert "data-banner" not in html
    assert element(html, "data-demo") == "demo data"  # the test broker is the fake one


def test_fleet_shows_four_tiles_with_notes(client):
    html = client.get("/fleet").text
    tiles = attr_tags(html, "data-tile")
    assert [re.search(r'data-tile="(\w+)"', t).group(1) for t in tiles] == ["account", "pnl", "workers", "jobs"]
    body = text(html)
    for label in ("Account value", "Today's profit/loss", "Workers online", "Jobs running"):
        assert label in body
    assert "+$1,000.00" in body  # signed, from the fake broker: 101,000 vs 100,000
    assert 'data-tone="gain"' in html


def test_worker_cards_show_name_temp_cpu_ram_and_idle_text(client, conn):
    w = enroll(client, conn, "w1")
    heartbeat(client, w, cpu_pct=12.0, ram_pct=34.0, temp_c=62.0)
    html = client.get("/fleet").text
    assert len(attr_tags(html, "data-worker")) == 1
    card = html[html.index("data-worker="):]
    assert 'data-name="w1"' in card and 'data-state="idle"' in card
    assert element(card, "data-temp") == "62°C"
    assert element(card, 'data-meter="cpu"') == "CPU 12%"
    assert element(card, 'data-meter="ram"') == "RAM 34%"
    assert element(card, "data-task") == "Idle — No job assigned"
    assert "data-progress" not in card


def test_hot_worker_temperature_is_flagged_with_words_too(client, conn):
    w = enroll(client, conn, "w1")
    heartbeat(client, w, temp_c=88.0)
    html = client.get("/fleet").text
    assert element(html, "data-temp") == "88°C · Hot"
    assert 'data-dot="hot"' in html


def test_worker_without_sensor_says_so(client, conn):
    w = enroll(client, conn, "w1")
    heartbeat(client, w, temp_c=None)
    assert element(client.get("/fleet").text, "data-temp") == "no sensor"


def test_offline_worker_says_last_seen(client, conn):
    w = enroll(client, conn, "w2")
    heartbeat(client, w)
    conn.execute("UPDATE workers SET last_heartbeat_at = now() - interval '5 minutes'")
    html = client.get("/fleet").text
    assert 'data-state="offline"' in html
    assert element(html, "data-task") == "Offline — Last seen 5 min ago"


def test_busy_worker_shows_job_detail_and_progress(client, conn):
    w = enroll(client, conn, "w3")
    heartbeat(client, w)
    client.post("/api/jobs", json={"kind": "sleep", "params": {"seconds": 5}, "target": w["worker_id"]})
    job = heartbeat(client, w, want_job=True)["claimed"][0]
    heartbeat(client, w, jobs=[{"id": job["id"], "lease_token": job["lease_token"], "progress": 0.4,
                                "detail": "Test job: 2 of 5 seconds"}])
    html = client.get("/fleet").text
    assert 'data-state="busy"' in html
    assert element(html, "data-task") == "Test job — Test job: 2 of 5 seconds"
    assert element(html, "data-progress-text") == "40%"
    assert 'style="width:40.0%"' in html


def test_live_progress_shows_live_label_and_full_bar():
    from coordinator import web
    html = web.env.get_template("_workers.html").render(workers=[{
        "id": "x", "name": "w1", "online": True, "enabled": True, "hot": False, "state": "busy", "dot": "busy",
        "temp": {"text": "50°C", "hot": False, "known": True}, "cpu_pct": 10.0, "ram_pct": 20.0,
        "task": "Paper trade · momentum", "detail": "Trading", "progress": {"text": "Live", "pct": None}}])
    assert element(html, "data-progress-text") == "Live"
    assert 'class="bar live"' in html and "aria-valuenow" not in html


def test_pause_shows_banner_and_flips_button(client):
    assert client.post("/api/trading/pause").json() == {"paused": True}
    html = client.get("/fleet").text
    assert element(html, "data-banner") .startswith(BANNER)
    assert f"<p>{BANNER}</p>" in html
    assert element(html, 'data-action="resume"') == "Resume trading"
    assert 'data-action="pause"' not in html
    client.post("/api/trading/resume")
    html = client.get("/fleet").text
    assert "data-banner" not in html and element(html, 'data-action="pause"') == "Pause all trading"


def test_previous_jobs_table_shows_failure_reason_and_run_again(client, conn):
    w = enroll(client, conn, "w3")
    heartbeat(client, w)
    job_id = failed_job(client, conn, w)
    html = client.get("/fleet").text
    table = html[html.index('data-table="previous-jobs"'):]
    for col in ("Job", "Model", "Worker", "Finished", "Took", "Result", "Status"):
        assert f"<th>{col}</th>" in table
    assert f'data-job="{job_id}" data-state="failed"' in table
    assert element(table, "data-result") == "Worker w3 went offline"
    assert element(table, 'data-status="failed"') == "Failed"
    assert re.search(rf'data-action="run-again" data-job-id="{job_id}">Run again</button>', table)
    assert 'class="table-wrap"' in html  # the table scrolls sideways inside its own box


def test_succeeded_job_has_no_run_again_button(client, conn):
    w = enroll(client, conn, "w1")
    heartbeat(client, w)
    client.post("/api/jobs", json={"kind": "sleep", "target": w["worker_id"]})
    job = heartbeat(client, w, want_job=True)["claimed"][0]
    done = client.post(f"/api/v1/jobs/{job['id']}/complete", json={"lease_token": job["lease_token"], "result": {"summary": "Slept 5 s"}},
                       headers={"Authorization": "Bearer " + w["worker_token"]})
    assert done.status_code == 200, done.text
    html = client.get("/fleet").text
    assert element(html, 'data-status="succeeded"') == "Succeeded"
    assert element(html, "data-result") == "Slept 5 s"
    assert 'data-action="run-again"' not in html


def test_empty_states(client):
    html = client.get("/fleet").text
    assert element(html, 'data-empty="workers"') == "No workers enrolled yet."
    assert element(html, 'data-empty="jobs"') == "No finished jobs yet."


def test_assign_panel_dropdowns(client, conn):
    heartbeat(client, enroll(client, conn, "w1"))
    html = client.get("/fleet").text
    kind = html[html.index('data-field="kind"'):html.index('data-field="model"')]
    options = re.findall(r"<option [^>]*>([^<]*)</option>", kind)
    assert options == ["Backtest (coming soon)", "Paper trade (coming soon)",
                       "Model search (coming soon)", "Data refresh (coming soon)"]  # none are available in stage 1
    assert kind.count("disabled") == 4
    assert "Test one model on past prices and save its results." in html
    assert 'data-field-wrap="model"' in html and 'value="" disabled selected>No models yet' in html
    worker = html[html.index('data-field="worker"'):]
    assert re.findall(r"<option [^>]*>([^<]*)</option>", worker)[:3] == ["Auto — pick the least busy", "w1", "All idle workers"]
    assert re.search(r'<button[^>]*data-action="assign"[^>]*disabled', html)  # nothing to assign yet


def test_assign_panel_enables_available_jobs_and_hides_model_when_not_needed(client, monkeypatch):
    monkeypatch.setattr(fleet_view, "AVAILABLE_KINDS", ("sleep", "data_refresh"))
    html = client.get("/fleet").text
    assert re.search(r'<option value="data_refresh"[^>]*selected>Data refresh</option>', html)
    assert "Download the latest prices to the coordinator." in html
    assert re.search(r'data-field-wrap="model" hidden', html)  # data_refresh needs no model
    assert not re.search(r'<button[^>]*data-action="assign"[^>]*disabled', html)


def test_fragment_returns_the_refreshable_regions(client, conn):
    w = enroll(client, conn, "w1")
    heartbeat(client, w)
    client.post("/api/trading/pause")
    html = client.get("/fragments/fleet").text
    regions = re.findall(r'data-region="([\w-]+)"', html)
    assert regions == ["status", "banner", "tiles", "workers", "jobs", "worker-options", "model-options"]
    assert element(html, 'data-pill="mode"') == "Paper"
    assert element(html, 'data-action="resume"') == "Resume trading"
    assert f"<p>{BANNER}</p>" in html
    assert len(attr_tags(html, "data-tile")) == 4 and 'data-worker="' in html
    assert '<option value="auto">Auto — pick the least busy</option>' in html
    assert "<html" not in html and "Assign a job" not in html  # the panel itself is never swapped


def test_models_placeholder_has_the_same_header(client):
    html = client.get("/models").text
    assert element(html, 'data-placeholder="models"') == "The Models screen arrives in stage 3."
    assert 'data-nav="models" class="current" aria-current="page"' in html
    assert element(html, 'data-pill="crypto"') == "Crypto 24/7"
    assert element(html, 'data-action="pause"') == "Pause all trading"
    client.post("/api/trading/pause")
    assert f"<p>{BANNER}</p>" in client.get("/models").text


def test_static_files_are_served(client):
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/fonts/manrope-latin-400-normal.woff2").status_code == 200
    assert client.get("/static/fonts/jetbrains-mono-latin-400-normal.woff2").status_code == 200


def test_pages_need_the_owner_login(config, broker):
    from fastapi.testclient import TestClient
    from coordinator.api.app import create_app
    from coordinator.broker import BrokerStatus
    strict = replace(config, dev=False, owner_login="owner@example.com")
    status = BrokerStatus(broker, ttl=0)
    status.refresh(force=True)
    with TestClient(create_app(strict, status)) as c:
        assert c.get("/fleet").status_code == 401
        assert c.get("/fragments/fleet").status_code == 401
        ok = c.get("/fleet", headers={"Tailscale-User-Login": "owner@example.com"})
        assert ok.status_code == 200
