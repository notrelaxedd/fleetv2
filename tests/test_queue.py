"""Queue behaviour: claim, one job per worker, dead worker -> failed, run again, cancel, Auto pick."""
from __future__ import annotations

import psycopg

from coordinator import queue
from tests.conftest import enroll, heartbeat


def test_auto_job_goes_to_least_busy_worker_and_reports_progress(client, conn):
    busy = enroll(client, conn, "w1")
    calm = enroll(client, conn, "w2")
    heartbeat(client, busy, cpu_pct=90.0)
    heartbeat(client, calm, cpu_pct=5.0)
    resp = client.post("/api/jobs", json={"kind": "sleep", "params": {"seconds": 5}, "target": "auto"})
    assert resp.status_code == 201, resp.text
    assert resp.json()["message"] == "Test job sent to w2"
    reply = heartbeat(client, calm, want_job=True)
    assert len(reply["claimed"]) == 1
    job = reply["claimed"][0]
    heartbeat(client, calm, jobs=[{"id": job["id"], "lease_token": job["lease_token"], "progress": 0.4,
                                   "detail": "Test job: 2 of 5 seconds"}])
    fleet = client.get("/api/fleet").json()
    card = next(w for w in fleet["workers"] if w["name"] == "w2")
    assert card["state"] == "busy"
    assert card["progress"] == {"text": "40%", "pct": 40.0}
    assert card["detail"] == "Test job: 2 of 5 seconds"
    assert heartbeat(client, busy, want_job=True)["claimed"] == []


def test_worker_that_goes_offline_fails_its_job_which_can_run_again(client, conn):
    w = enroll(client, conn, "w3")
    heartbeat(client, w)
    client.post("/api/jobs", json={"kind": "sleep", "params": {"seconds": 60}, "target": w["worker_id"]})
    job = heartbeat(client, w, want_job=True)["claimed"][0]
    conn.execute("UPDATE jobs SET lease_expires_at = now() - interval '1 second' WHERE id = %s", (job["id"],))
    queue.reap(conn)
    row = conn.execute("SELECT status, error FROM jobs WHERE id = %s", (job["id"],)).fetchone()
    assert row == {"status": "failed", "error": "Worker w3 went offline"}
    prev = client.get("/api/fleet").json()["previous_jobs"][0]
    assert prev["status"] == "Failed" and prev["result"] == "Worker w3 went offline" and prev["can_run_again"]
    again = client.post(f"/api/jobs/{job['id']}/run-again")
    assert again.status_code == 201
    new = conn.execute("SELECT kind, status, params, retry_of FROM jobs WHERE id = %s", (again.json()["jobs"][0]["id"],)).fetchone()
    assert new["kind"] == "sleep" and new["status"] == "queued" and str(new["retry_of"]) == job["id"]


def test_cancel_running_job_is_stopped_through_heartbeat(client, conn):
    w = enroll(client, conn, "w4")
    heartbeat(client, w)
    client.post("/api/jobs", json={"kind": "sleep", "params": {"seconds": 60}, "target": w["worker_id"]})
    job = heartbeat(client, w, want_job=True)["claimed"][0]
    client.post(f"/api/jobs/{job['id']}/cancel")
    reply = heartbeat(client, w, jobs=[{"id": job["id"], "lease_token": job["lease_token"], "progress": 0.1}])
    assert reply["cancel"] == [job["id"]]
    heartbeat(client, w, released=[{"id": job["id"], "lease_token": job["lease_token"], "reason": "cancel"}])
    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job["id"],)).fetchone()["status"] == "cancelled"


def test_all_idle_makes_one_job_per_idle_worker(client, conn):
    for name in ("w5", "w6"):
        heartbeat(client, enroll(client, conn, name))
    resp = client.post("/api/jobs", json={"kind": "sleep", "target": "all_idle"})
    assert resp.status_code == 201
    assert len(resp.json()["jobs"]) == 2


def test_offline_worker_card_says_last_seen(client, conn):
    w = enroll(client, conn, "w7")
    heartbeat(client, w)
    conn.execute("UPDATE workers SET last_heartbeat_at = now() - interval '5 minutes'")
    card = client.get("/api/fleet").json()["workers"][0]
    assert card["state"] == "offline" and card["detail"] == "Last seen 5 min ago"


def test_two_workers_never_claim_the_same_job(client, conn):
    a, b = enroll(client, conn, "w8"), enroll(client, conn, "w9")
    conn.execute("INSERT INTO jobs (kind) VALUES ('sleep')")
    got = heartbeat(client, a, want_job=True)["claimed"] + heartbeat(client, b, want_job=True)["claimed"]
    assert len(got) == 1


def test_pause_and_resume_switch(client, conn):
    assert client.post("/api/trading/pause").json() == {"paused": True}
    header = client.get("/api/fleet").json()["header"]
    assert header["pause_button"] == "Resume trading"
    assert header["banner"] == ("All trading is paused. No model will place orders until you resume. "
                                "Backtests keep running.")
    client.post("/api/trading/resume")
    assert client.get("/api/fleet").json()["header"]["banner"] is None
    actions = [r["action"] for r in conn.execute("SELECT action FROM audit_log ORDER BY id")]
    assert "trading_paused" in actions and "trading_resumed" in actions


def test_coordinator_restart_keeps_running_jobs(client, conn):
    from coordinator import recovery

    w = enroll(client, conn, "w10")
    heartbeat(client, w)
    client.post("/api/jobs", json={"kind": "sleep", "params": {"seconds": 60}, "target": w["worker_id"]})
    job = heartbeat(client, w, want_job=True)["claimed"][0]
    conn.execute("UPDATE jobs SET lease_expires_at = now() - interval '1 second'")
    assert recovery.startup_grace(conn, 30) == 1
    queue.reap(conn)
    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job["id"],)).fetchone()["status"] == "leased"


def test_backtest_job_carries_model_and_limits_and_stores_result(client, conn):
    w = enroll(client, conn, "w11")
    heartbeat(client, w)
    resp = client.post("/api/jobs", json={"kind": "backtest", "model_id": "momentum", "target": "auto"})
    assert resp.status_code == 201, resp.text
    assert resp.json()["message"] == "Backtest sent to w11"
    job = heartbeat(client, w, want_job=True)["claimed"][0]
    assert job["params"]["module"] == "momentum" and job["params"]["market"] == "stocks"
    assert job["params"]["limits"] == {"money": 10000.0, "max_per_position": 1000.0, "max_per_model": 10000.0}
    assert job["params"]["held_out_fraction"] == 0.25
    metrics = {"held_out": {"roi": 0.05}, "train": {"roi": 0.1}}
    r = client.post("/api/v1/models/momentum/backtest", json={"job_id": job["id"], "backtest_metrics": metrics},
                    headers={"Authorization": "Bearer " + w["worker_token"]})
    assert r.status_code == 200 and r.json()["status"] == "backtested"
    row = conn.execute("SELECT status, metrics FROM models WHERE id = 'momentum'").fetchone()
    assert row["metrics"]["held_out"]["roi"] == 0.05


def test_backtest_needs_a_model_and_paper_trade_is_not_ready(client):
    assert client.post("/api/jobs", json={"kind": "backtest"}).status_code == 400
    assert client.post("/api/jobs", json={"kind": "paper_trade", "model_id": "momentum"}).status_code == 400
