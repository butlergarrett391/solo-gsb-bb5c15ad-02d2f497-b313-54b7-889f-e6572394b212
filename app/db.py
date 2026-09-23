from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

# One process-wide lock serializes every read/write transaction. The API is
# I/O-latency friendly enough that a coarse lock is preferable to subtle
# scheduling races (a batch decision must be atomic w.r.t. submit/cancel).
_lock = threading.RLock()

_conn: sqlite3.Connection | None = None


def get_lock() -> threading.RLock:
    return _lock


def connect(db_path: str) -> sqlite3.Connection:
    """Open (or reopen) the module-level SQLite connection and create tables."""
    global _conn
    if _conn is not None:
        return _conn

    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        db_path,
        timeout=30,
        check_same_thread=False,
        isolation_level=None,  # explicit BEGIN/COMMIT
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    _init_schema(conn)
    _conn = conn
    return _conn


def get_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("database not initialized; call connect() first")
    return _conn


def reset_for_tests() -> None:
    """Drop the module-level connection (test isolation helper)."""
    global _conn
    if _conn is not None:
        _conn.close()
    _conn = None


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS tenants (
            id               TEXT PRIMARY KEY,
            weight           REAL NOT NULL,
            deficit          REAL NOT NULL DEFAULT 0,
            -- Wall-clock time up to which aging has already been compensated.
            aging_settled_at TEXT,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id              TEXT PRIMARY KEY,
            tenant_id       TEXT NOT NULL REFERENCES tenants(id),
            gpus            INTEGER NOT NULL,
            duration_min    INTEGER NOT NULL,
            cost_gpu_min    REAL NOT NULL,
            status          TEXT NOT NULL CHECK (status IN
                                ('QUEUED','RUNNING','COMPLETED','CANCELLED')),
            queued_at       TEXT NOT NULL,
            scheduled_at    TEXT,
            completed_at    TEXT,
            batch_id        INTEGER REFERENCES batches(id),
            run_index       INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        CREATE INDEX IF NOT EXISTS idx_jobs_tenant_status
            ON jobs(tenant_id, status);

        -- DAG edges: job_id may only run after every depends_on job is
        -- COMPLETED. CANCELLED prerequisites stay unmet forever, leaving the
        -- dependent queued (and explicitly blocked_by the cancelled job).
        CREATE TABLE IF NOT EXISTS job_dependencies (
            job_id     TEXT NOT NULL REFERENCES jobs(id),
            depends_on TEXT NOT NULL REFERENCES jobs(id),
            PRIMARY KEY (job_id, depends_on)
        );
        CREATE INDEX IF NOT EXISTS idx_job_deps_target
            ON job_dependencies(depends_on);

        CREATE TABLE IF NOT EXISTS batches (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at       TEXT NOT NULL,
            now_ref          TEXT NOT NULL,
            capacity_gpus    INTEGER NOT NULL,
            used_gpus_before INTEGER NOT NULL,
            available_gpus   INTEGER NOT NULL,
            rounds           INTEGER NOT NULL,
            jobs_selected    INTEGER NOT NULL
        );

        -- One row per credit/debit event: aging grants, DRR quantum grants,
        -- job dispatch debits. Makes every per-tenant balance reproducible.
        CREATE TABLE IF NOT EXISTS ledger (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id   TEXT NOT NULL REFERENCES tenants(id),
            batch_id    INTEGER REFERENCES batches(id),
            job_id      TEXT REFERENCES jobs(id),
            kind        TEXT NOT NULL CHECK (kind IN
                          ('AGING','QUANTUM','DISPATCH','RESET')),
            amount      REAL NOT NULL,
            balance_after REAL NOT NULL,
            detail      TEXT,
            created_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ledger_tenant ON ledger(tenant_id, id);
        """
    )
