"""Connection pool helpers and the SQL migration runner."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

log = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
MIGRATION_LOCK_KEY = 7_401_001


def _configure(conn: psycopg.Connection) -> None:
    """Session setup applied to every pooled connection."""
    conn.execute("SET TIME ZONE 'UTC'")
    conn.commit()


def make_pool(conn_url: str, min_size: int = 1, max_size: int = 16) -> ConnectionPool:
    """Create and open a pool whose connections return dict rows."""
    pool = ConnectionPool(
        conn_url,
        min_size=min_size,
        max_size=max_size,
        kwargs={"row_factory": dict_row},
        configure=_configure,
        open=False,
    )
    pool.open()
    return pool


@contextmanager
def connection(pool: ConnectionPool) -> Iterator[psycopg.Connection]:
    """Borrow a connection; commits on success and rolls back on error."""
    with pool.connection() as conn:
        yield conn


@contextmanager
def connect(conn_url: str, autocommit: bool = False) -> Iterator[psycopg.Connection]:
    """Open a standalone dict-row connection outside the pool."""
    conn = psycopg.connect(conn_url, row_factory=dict_row, autocommit=autocommit)
    try:
        _configure(conn)
        if not autocommit:
            conn.commit()
        yield conn
        if not autocommit:
            conn.commit()
    except BaseException:
        if not autocommit:
            conn.rollback()
        raise
    finally:
        conn.close()


def migration_files(directory: Path = MIGRATIONS_DIR) -> list[Path]:
    """All *.sql migrations sorted by filename."""
    return sorted(p for p in directory.glob("*.sql") if p.is_file())


def applied_versions(conn: psycopg.Connection) -> set[str]:
    """Versions recorded in schema_migrations (empty if the table is missing)."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
    )
    rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    return {row["version"] for row in rows}


def migrate(conn_url: str, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply pending migrations in order, each in its own transaction.

    Returns the versions applied by this call. Safe to run repeatedly.
    """
    applied: list[str] = []
    with connect(conn_url) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK_KEY,))
        conn.commit()
        done = applied_versions(conn)
        conn.commit()
        for path in migration_files(directory):
            version = path.stem
            if version in done:
                continue
            log.info("applying migration %s", version)
            conn.execute(path.read_text(encoding="utf-8"))
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            conn.commit()
            applied.append(version)
        conn.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK_KEY,))
    return applied
