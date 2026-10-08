"""Shared fixtures (pattern copied from v1): per-test databases cloned from a migrated
template, an app client with the fake broker, and helpers to enroll fake workers."""
from __future__ import annotations

import hashlib
import os
import uuid
from typing import Any, Iterator

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row

from coordinator import auth, db
from coordinator.api.app import create_app
from coordinator.broker import BrokerStatus, FakeBroker
from coordinator.config import Config

ADMIN_URL = os.environ.get("FLEET_TEST_DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:5432/postgres")
TEMPLATE_LOCK_KEY = 7_402_002


def _digest() -> str:
    h = hashlib.sha256()
    for path in db.migration_files():
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:12]


TEMPLATE_DB = "fleet2_test_template_" + _digest()


def db_url(name: str) -> str:
    return make_conninfo(ADMIN_URL, dbname=name)


def _admin() -> psycopg.Connection:
    return psycopg.connect(ADMIN_URL, autocommit=True, row_factory=dict_row)


@pytest.fixture(scope="session")
def template_db() -> str:
    """Create the migrated template database once per session."""
    with _admin() as admin:
        admin.execute("SELECT pg_advisory_lock(%s)", (TEMPLATE_LOCK_KEY,))
        try:
            if not admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEMPLATE_DB,)).fetchone():
                admin.execute(f'CREATE DATABASE "{TEMPLATE_DB}"')
                db.migrate(db_url(TEMPLATE_DB))
        finally:
            admin.execute("SELECT pg_advisory_unlock(%s)", (TEMPLATE_LOCK_KEY,))
    return TEMPLATE_DB


@pytest.fixture
def test_db_url(template_db: str) -> Iterator[str]:
    """A fresh database cloned from the template, dropped afterwards."""
    name = f"fleet2_test_{uuid.uuid4().hex[:12]}"
    with _admin() as admin:
        admin.execute(f'CREATE DATABASE "{name}" TEMPLATE "{template_db}"')
    try:
        yield db_url(name)
    finally:
        with _admin() as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture
def conn(test_db_url: str) -> Iterator[psycopg.Connection]:
    """A dict-row connection to the test database (autocommit, so each call sticks)."""
    with psycopg.connect(test_db_url, row_factory=dict_row, autocommit=True) as c:
        c.execute("SET TIME ZONE 'UTC'")
        yield c


@pytest.fixture
def config(test_db_url: str) -> Config:
    return Config(database_url=test_db_url, dev=True, public_url="http://127.0.0.1:8090")


@pytest.fixture
def broker() -> FakeBroker:
    return FakeBroker(equity=101_000.0, last_equity=100_000.0)


@pytest.fixture
def client(config: Config, broker: FakeBroker) -> Iterator[TestClient]:
    status = BrokerStatus(broker, ttl=0)
    status.refresh(force=True)
    app = create_app(config, status)
    with TestClient(app) as c:
        yield c


def enroll(client: TestClient, conn: psycopg.Connection, name: str) -> dict[str, Any]:
    """Enroll a fake worker named `name`; returns {"worker_id", "worker_token"}."""
    token, _ = auth.create_enroll_token(conn)
    resp = client.post("/api/v1/workers/register", json={"enroll_token": token, "name": name, "hostname": name})
    assert resp.status_code == 200, resp.text
    return resp.json()


def heartbeat(client: TestClient, worker: dict[str, Any], **body: Any) -> dict[str, Any]:
    """One heartbeat from a fake worker; returns the reply."""
    payload = {"cpu_pct": 10.0, "ram_pct": 30.0, "temp_c": 55.0, "jobs": [], "released": [], "want_job": False}
    payload.update(body)
    resp = client.post(f"/api/v1/workers/{worker['worker_id']}/heartbeat", json=payload,
                       headers={"Authorization": "Bearer " + worker["worker_token"]})
    assert resp.status_code == 200, resp.text
    return resp.json()
