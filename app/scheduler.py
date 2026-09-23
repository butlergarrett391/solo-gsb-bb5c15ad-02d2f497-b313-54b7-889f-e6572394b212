"""Fair-queuing core: weighted deficit round-robin with wait-time aging.

The module is deliberately framework-free; FastAPI only translates HTTP into
calls here. All functions are serialized by ``app.db.get_lock()`` and use
explicit SQLite transactions so that a scheduling batch is atomic with
respect to submissions, cancellations and completions.

Cost unit
=========
A job's cost is its resource footprint:

    cost_gpu_min = gpus * duration_min

A tenant's *balance* (deficit counter) is measured in the same unit.

Dependencies
============
A job may be submitted with ``depends_on``: a list of prerequisite job ids
(any tenant — dependencies cross tenant boundaries). Prerequisites must exist
at submit time, and adding the edges must not create a cycle. A dependent job
is eligible only when **all** prerequisites are COMPLETED; a CANCELLED
prerequisite can never complete, so the dependent stays blocked forever and
the reason is visible in ``blocked_by``. A dependency-blocked head-of-line
job receives neither DRR quantum nor aging credit and does not consume any
capacity; the tenant's queue cannot advance past it.

WDRR per batch
==============
1. Each active tenant (one with an *eligible* QUEUED head job, tenant id
   order) receives ``base_quantum * weight`` GPU-minutes of credit every DRR
   round.
2. Visiting tenants in fixed order, the head-of-line job is dispatched when
   ``balance >= job_cost`` AND enough free GPUs exist AND all prerequisites
   are COMPLETED.
3. A full round in which nothing is dispatched ends the batch: blocked jobs
   keep their credit, and a later batch (after completions or aging) retries.

Aging compensation
==================
While a tenant has an eligible queued job, elapsed wall-clock time since its
last settlement is converted at ``AGING_RATE`` into extra credit (optionally
capped per batch). When the tenant's queue drains the aging clock resets, so
a later freshly submitted small job starts a clean wait window. Aging lets
small jobs behind large, high-weight tenants accumulate credit until they
become affordable, bounding starvation. Time spent purely dependency-blocked
does not age: the block is caused by the submitter's own graph, not by the
scheduler's fairness policy.

Reproducibility
===============
Tenant visit order is tenant-id ascending, jobs are FIFO per tenant, job ids
are sequential, and ``run_batch`` accepts an injectable ``now``/seed-free
logic: identical inputs always produce identical selection order.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from . import db
from .config import Settings, get_settings

QUEUED, RUNNING, COMPLETED, CANCELLED = "QUEUED", "RUNNING", "COMPLETED", "CANCELLED"

# Float comparison tolerance for the affordability test.
_EPS = 1e-9


class SchedulerError(Exception):
    """Base class for expected, HTTP-mappable service errors."""

    status_code = 400


class NotFound(SchedulerError):
    status_code = 404


class Conflict(SchedulerError):
    status_code = 409


def _now_iso(now: datetime | None = None) -> str:
    if now is None:
        now = datetime.now(timezone.utc)
    return now.astimezone(timezone.utc).isoformat()


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _job_id(conn: sqlite3.Connection) -> str:
    # Deterministic, human-readable sequential ids (J000001, ...).
    row = conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()
    return f"J{row['n'] + 1:06d}"


def _credit(
    conn: sqlite3.Connection,
    tenant_id: str,
    amount: float,
    kind: str,
    created_at: str,
    batch_id: int | None = None,
    job_id: str | None = None,
    detail: str | None = None,
) -> float:
    """Apply a signed ledger entry and return the resulting balance."""
    conn.execute(
        "UPDATE tenants SET deficit = deficit + ?, updated_at = ? WHERE id = ?",
        (amount, created_at, tenant_id),
    )
    row = conn.execute(
        "SELECT deficit FROM tenants WHERE id = ?", (tenant_id,)
    ).fetchone()
    conn.execute(
        """INSERT INTO ledger
               (tenant_id, batch_id, job_id, kind, amount, balance_after,
                detail, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (tenant_id, batch_id, job_id, kind, amount, row["deficit"],
         detail, created_at),
    )
    return row["deficit"]


def _has_queued(conn: sqlite3.Connection, tenant_id: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE tenant_id = ? AND status = ?",
        (tenant_id, QUEUED),
    ).fetchone()
    return row["n"] > 0


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def _depends_on_ids(conn: sqlite3.Connection, job_id: str) -> list[str]:
    """Prerequisite job ids in submission (edge) order."""
    return [
        r["depends_on"]
        for r in conn.execute(
            """SELECT depends_on FROM job_dependencies
                WHERE job_id = ? ORDER BY id""",
            (job_id,),
        ).fetchall()
    ]


def _unmet_dependencies(
    conn: sqlite3.Connection, job_id: str
) -> list[dict[str, Any]]:
    """Prerequisites that have not COMPLETED, with the blocking reason.

    COMPLETED prerequisites are omitted. A CANCELLED prerequisite blocks
    permanently (it can never complete); QUEUED/RUNNING ones block until
    they finish.
    """
    rows = conn.execute(
        """SELECT j.id, j.status
             FROM job_dependencies d
             JOIN jobs j ON j.id = d.depends_on
            WHERE d.job_id = ? AND j.status != ?
            ORDER BY d.id""",
        (job_id, COMPLETED),
    ).fetchall()
    blocked: list[dict[str, Any]] = []
    for r in rows:
        reason = (
            f"prerequisite {r['id']} was CANCELLED and can never complete"
            if r["status"] == CANCELLED
            else f"prerequisite {r['id']} is {r['status']}, not COMPLETED"
        )
        blocked.append({"job_id": r["id"], "status": r["status"],
                        "reason": reason})
    return blocked


def _dependencies_satisfied(conn: sqlite3.Connection, job_id: str) -> bool:
    row = conn.execute(
        """SELECT COUNT(*) AS n
             FROM job_dependencies d
             JOIN jobs j ON j.id = d.depends_on
            WHERE d.job_id = ? AND j.status != ?""",
        (job_id, COMPLETED),
    ).fetchone()
    return row["n"] == 0


def _would_create_cycle(
    conn: sqlite3.Connection, job_id: str, prereq_ids: list[str]
) -> list[str] | None:
    """Check whether adding ``job_id -> prereq`` edges would close a cycle.

    Returns the offending cycle path (ending back at its first node) or
    ``None``. The new edges are not yet in the table, so ``job_id`` is treated
    as if it already pointed at ``prereq_ids``. Self-dependence is a direct
    cycle. Pre-existing cycles reachable from the prerequisites (possible via
    out-of-band writes) are reported too.
    """
    if job_id in prereq_ids:
        return [job_id, job_id]

    def successors(node: str) -> list[str]:
        if node == job_id:
            return list(prereq_ids)
        return [
            r["depends_on"]
            for r in conn.execute(
                "SELECT depends_on FROM job_dependencies WHERE job_id = ?",
                (node,),
            ).fetchall()
        ]

    # Iterative DFS with white/gray/black colouring; a back edge to a gray
    # node closes a cycle.
    GRAY, BLACK = 1, 2
    color: dict[str, int] = {}
    for root in prereq_ids:
        if color.get(root) == BLACK:
            continue
        # stack frames: (node, child index)
        stack: list[tuple[str, int]] = [(root, 0)]
        path = [root]
        color[root] = GRAY
        while stack:
            node, idx = stack[-1]
            children = successors(node)
            if idx < len(children):
                stack[-1] = (node, idx + 1)
                child = children[idx]
                if color.get(child) == GRAY:
                    start = path.index(child)
                    return path[start:] + [child]
                if color.get(child) != BLACK:
                    color[child] = GRAY
                    path.append(child)
                    stack.append((child, 0))
            else:
                color[node] = BLACK
                stack.pop()
                path.pop()
    return None


# ---------------------------------------------------------------------------
# Tenants
# ---------------------------------------------------------------------------

def upsert_tenant(
    tenant_id: str,
    weight: float,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Create a tenant or update its weight. Existing balances are retained."""
    if not tenant_id or not tenant_id.strip():
        raise SchedulerError("tenant_id must be a non-empty string")
    if weight <= 0:
        raise SchedulerError("weight must be > 0")
    settings = settings or get_settings()
    now = _now_iso()
    with db.get_lock(), db.get_conn():
        conn = db.get_conn()
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT id FROM tenants WHERE id = ?", (tenant_id,)
        ).fetchone()
        if existing is None:
            conn.execute(
                """INSERT INTO tenants
                       (id, weight, deficit, aging_settled_at,
                        created_at, updated_at)
                   VALUES (?, ?, 0, NULL, ?, ?)""",
                (tenant_id, float(weight), now, now),
            )
        else:
            conn.execute(
                "UPDATE tenants SET weight = ?, updated_at = ? WHERE id = ?",
                (float(weight), now, tenant_id),
            )
        return get_tenant(tenant_id)


def get_tenant(tenant_id: str) -> dict[str, Any]:
    with db.get_lock():
        conn = db.get_conn()
        row = conn.execute(
            _TENANT_SELECT + " HAVING t.id = ?", (tenant_id,)
        ).fetchone()
        if row is None:
            raise NotFound(f"tenant '{tenant_id}' not found")
        return _tenant_dict(conn, row)


_TENANT_SELECT = """
    SELECT t.id, t.weight, t.deficit, t.aging_settled_at,
           SUM(CASE WHEN j.status = 'QUEUED'   THEN 1 ELSE 0 END) AS queued,
           SUM(CASE WHEN j.status = 'RUNNING'  THEN 1 ELSE 0 END) AS running
      FROM tenants t
      LEFT JOIN jobs j ON j.tenant_id = t.id
     GROUP BY t.id
"""


def _tenant_dict(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "weight": round(row["weight"], 6),
        "balance_gpu_min": round(row["deficit"], 6),
        "queued_jobs": row["queued"],
        "running_jobs": row["running"],
        "aging_settled_at": row["aging_settled_at"],
    }


def _list_tenants(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(_TENANT_SELECT + " ORDER BY t.id").fetchall()
    return [_tenant_dict(conn, r) for r in rows]


def list_tenants() -> list[dict[str, Any]]:
    with db.get_lock():
        return _list_tenants(db.get_conn())


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------

def submit_job(
    tenant_id: str,
    gpus: int,
    duration_min: int,
    weight: float = 1.0,
    depends_on: list[str] | None = None,
) -> dict[str, Any]:
    if not tenant_id or not tenant_id.strip():
        raise SchedulerError("tenant_id must be a non-empty string")
    if not isinstance(gpus, int) or gpus <= 0:
        raise SchedulerError("gpus must be a positive integer")
    if not isinstance(duration_min, int) or duration_min <= 0:
        raise SchedulerError("duration_min must be a positive integer")
    if weight <= 0:
        raise SchedulerError("weight must be > 0")

    prereq_ids = list(depends_on or [])
    for dep in prereq_ids:
        if not isinstance(dep, str) or not dep.strip():
            raise SchedulerError(
                "depends_on entries must be non-empty job id strings"
            )
    # Preserve the caller's order while ignoring duplicate edges.
    seen: set[str] = set()
    unique_prereqs: list[str] = []
    for dep in prereq_ids:
        if dep not in seen:
            seen.add(dep)
            unique_prereqs.append(dep)
    prereq_ids = unique_prereqs

    settings = get_settings()
    if gpus > settings.cluster_gpus:
        raise SchedulerError(
            f"job requests {gpus} GPUs which exceeds cluster capacity "
            f"{settings.cluster_gpus}; it could never run"
        )

    now = _now_iso()
    cost = float(gpus * duration_min)
    with db.get_lock(), db.get_conn():
        conn = db.get_conn()
        conn.execute("BEGIN IMMEDIATE")

        # Every prerequisite must exist (in any tenant).
        missing = [
            dep for dep in prereq_ids
            if conn.execute(
                "SELECT 1 FROM jobs WHERE id = ?", (dep,)
            ).fetchone() is None
        ]
        if missing:
            raise SchedulerError(
                f"unknown depends_on job id(s): {', '.join(missing)}"
            )

        # Auto-register unknown tenants so a single POST /jobs is enough.
        if conn.execute(
            "SELECT 1 FROM tenants WHERE id = ?", (tenant_id,)
        ).fetchone() is None:
            conn.execute(
                """INSERT INTO tenants
                       (id, weight, deficit, aging_settled_at,
                        created_at, updated_at)
                   VALUES (?, ?, 0, ?, ?, ?)""",
                (tenant_id, float(weight), now, now, now),
            )
        else:
            # Latest submission may adjust the tenant's weight.
            conn.execute(
                "UPDATE tenants SET weight = ?, updated_at = ? WHERE id = ?",
                (float(weight), now, tenant_id),
            )

        job_id = _job_id(conn)

        # Refuse edges that would close a dependency cycle.
        cycle = _would_create_cycle(conn, job_id, prereq_ids)
        if cycle is not None:
            raise SchedulerError(
                "depends_on creates a dependency cycle: "
                + " -> ".join(cycle)
            )

        conn.execute(
            """INSERT INTO jobs
                   (id, tenant_id, gpus, duration_min, cost_gpu_min, status,
                    queued_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (job_id, tenant_id, gpus, duration_min, cost, QUEUED, now),
        )
        conn.executemany(
            """INSERT INTO job_dependencies (job_id, depends_on, created_at)
               VALUES (?, ?, ?)""",
            [(job_id, dep, now) for dep in prereq_ids],
        )
        # aging_settled_at is left untouched: a non-empty queue keeps accruing
        # wait time, and a drained queue already has it reset to NULL.
        return _get_job_or_404(conn, job_id)


def _get_job_or_404(conn: sqlite3.Connection, job_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise NotFound(f"job '{job_id}' not found")
    return _job_dict(conn, row)


def _job_dict(
    conn: sqlite3.Connection, row: sqlite3.Row
) -> dict[str, Any]:
    blocked_by = (
        _unmet_dependencies(conn, row["id"])
        if row["status"] == QUEUED
        else []
    )
    return {
        "id": row["id"],
        "tenant_id": row["tenant_id"],
        "gpus": row["gpus"],
        "duration_min": row["duration_min"],
        "cost_gpu_min": round(row["cost_gpu_min"], 6),
        "status": row["status"],
        "depends_on": _depends_on_ids(conn, row["id"]),
        "blocked_by": blocked_by,
        "queued_at": row["queued_at"],
        "scheduled_at": row["scheduled_at"],
        "completed_at": row["completed_at"],
        "batch_id": row["batch_id"],
    }


def get_job(job_id: str) -> dict[str, Any]:
    with db.get_lock():
        return _get_job_or_404(db.get_conn(), job_id)


def cancel_job(job_id: str) -> dict[str, Any]:
    now = _now_iso()
    with db.get_lock(), db.get_conn():
        conn = db.get_conn()
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise NotFound(f"job '{job_id}' not found")
        if row["status"] in (COMPLETED, CANCELLED):
            raise Conflict(
                f"job '{job_id}' is already {row['status'].lower()}"
            )
        conn.execute(
            "UPDATE jobs SET status = ?, completed_at = ? WHERE id = ?",
            (CANCELLED, now, job_id),
        )
        if not _has_queued(conn, row["tenant_id"]):
            conn.execute(
                "UPDATE tenants SET aging_settled_at = NULL WHERE id = ?",
                (row["tenant_id"],),
            )
        return _get_job_or_404(conn, job_id)


def complete_job(job_id: str) -> dict[str, Any]:
    """Mark a RUNNING job COMPLETED, releasing its GPUs for future batches."""
    now = _now_iso()
    with db.get_lock(), db.get_conn():
        conn = db.get_conn()
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise NotFound(f"job '{job_id}' not found")
        if row["status"] != RUNNING:
            raise Conflict(
                f"job '{job_id}' is {row['status'].lower()}, only RUNNING "
                "jobs can complete"
            )
        conn.execute(
            "UPDATE jobs SET status = ?, completed_at = ? WHERE id = ?",
            (COMPLETED, now, job_id),
        )
        return _get_job_or_404(conn, job_id)


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def run_batch(
    now: datetime | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Schedule the next batch of jobs.

    Returns the selection order, per-tenant balances after the batch, capacity
    usage and an explainable ledger of credit/debit events.
    """
    settings = settings or get_settings()
    now_str = _now_iso(now)
    with db.get_lock(), db.get_conn():
        conn = db.get_conn()
        conn.execute("BEGIN IMMEDIATE")

        used = conn.execute(
            "SELECT COALESCE(SUM(gpus), 0) AS n FROM jobs WHERE status = ?",
            (RUNNING,),
        ).fetchone()["n"]
        available = settings.cluster_gpus - used

        cur = conn.execute(
            """INSERT INTO batches
                   (created_at, now_ref, capacity_gpus, used_gpus_before,
                    available_gpus, rounds, jobs_selected)
               VALUES (?, ?, ?, ?, ?, 0, 0)""",
            (now_str, now_str, settings.cluster_gpus, used,
             max(available, 0)),
        )
        batch_id = cur.lastrowid

        active = conn.execute(
            """SELECT DISTINCT tenant_id FROM jobs
                WHERE status = ? ORDER BY tenant_id""",
            (QUEUED,),
        ).fetchall()
        active_ids = [t["tenant_id"] for t in active]

        if not active_ids:
            conn.execute(
                "UPDATE batches SET rounds = 0, jobs_selected = 0 WHERE id = ?",
                (batch_id,),
            )
            conn.commit()
            return _batch_result(conn, batch_id, settings)

        def head_ready(tenant_id: str):
            """FIFO head job iff it is QUEUED and every prerequisite COMPLETED."""
            row = conn.execute(
                """SELECT * FROM jobs
                    WHERE tenant_id = ? AND status = ?
                    ORDER BY queued_at, id LIMIT 1""",
                (tenant_id, QUEUED),
            ).fetchone()
            if row is None or not _dependencies_satisfied(conn, row["id"]):
                return None
            return row

        # Aging is settled even when the cluster is full, so waiting tenants
        # keep accumulating starvation compensation across blocked batches.
        # --- Aging settlement at batch start -------------------------------
        # Compensation covers the full wait since the oldest queued job's
        # arrival: on first observation we settle from queued_at, afterwards
        # from the previous settlement instant. Same-instant retries grant
        # nothing, so the endpoint cannot be farmed via aging either.
        # Dependency-blocked head jobs accrue nothing: the wait is imposed by
        # the submitter's own dependency graph; advancing the settlement
        # instant ensures the blocked interval never pays out retroactively.
        for tenant_id in active_ids:
            head = conn.execute(
                """SELECT * FROM jobs
                    WHERE tenant_id = ? AND status = ?
                    ORDER BY queued_at, id LIMIT 1""",
                (tenant_id, QUEUED),
            ).fetchone()
            if head is not None and not _dependencies_satisfied(
                conn, head["id"]
            ):
                conn.execute(
                    "UPDATE tenants SET aging_settled_at = ?, updated_at = ? "
                    "WHERE id = ?",
                    (now_str, now_str, tenant_id),
                )
                continue
            t = conn.execute(
                "SELECT weight, deficit, aging_settled_at FROM tenants "
                "WHERE id = ?",
                (tenant_id,),
            ).fetchone()
            oldest = conn.execute(
                "SELECT MIN(queued_at) AS ts FROM jobs "
                "WHERE tenant_id = ? AND status = ?",
                (tenant_id, QUEUED),
            ).fetchone()["ts"]
            base = t["aging_settled_at"] or oldest
            waited_min = (
                _parse_iso(now_str) - _parse_iso(base)
            ).total_seconds() / 60.0
            if waited_min < 0:
                waited_min = 0.0
            bonus = settings.aging_rate_gpu_min_per_min * waited_min
            if settings.aging_cap_gpu_min is not None:
                bonus = min(bonus, settings.aging_cap_gpu_min)
            if bonus >= _EPS:
                _credit(
                    conn, tenant_id, bonus, "AGING", now_str,
                    batch_id=batch_id,
                    detail=f"waited {waited_min:.3f} min",
                )
            conn.execute(
                "UPDATE tenants SET aging_settled_at = ?, updated_at = ? "
                "WHERE id = ?",
                (now_str, now_str, tenant_id),
            )

        if available <= 0:
            conn.execute(
                "UPDATE batches SET rounds = 0, jobs_selected = 0 WHERE id = ?",
                (batch_id,),
            )
            conn.commit()
            return _batch_result(conn, batch_id, settings)

        # --- WDRR rounds ----------------------------------------------------
        # A tenant is visited in a round only while its dependency-ready head
        # job FITS the currently free capacity. Fitting-but-unaffordable jobs
        # keep accumulating quantum across rounds until affordable. A tenant
        # whose head job cannot physically fit (capacity-blocked) or whose
        # prerequisites are not all COMPLETED (dependency-blocked) accrues
        # nothing: waiting for completions is compensated by aging only in the
        # former case, and this prevents callers farming quantum by repeatedly
        # invoking the endpoint.
        #
        # Each round's grants are tentative above a commit watermark. The
        # moment a round produces a dispatch, every grant up to that point is
        # committed (it is ordinary DRR accrual). When the batch ends without
        # a trailing dispatch (e.g. max_rounds exhausted), the uncommitted
        # grants are rolled back so blocked calls leave balances unchanged.
        selected: list[str] = []
        run_index = 0
        rounds = 0
        free = available
        # Only tenants with a dependency-ready head job participate in DRR.
        round_tenants = [
            tid for tid in active_ids if head_ready(tid) is not None
        ]
        commit_mark = conn.execute(
            "SELECT COALESCE(MAX(id), 0) AS m FROM ledger"
        ).fetchone()["m"]

        def rollback_uncommitted() -> None:
            rows = conn.execute(
                "SELECT tenant_id, amount FROM ledger WHERE id > ?",
                (commit_mark,),
            ).fetchall()
            for r in rows:
                conn.execute(
                    "UPDATE tenants SET deficit = deficit - ? WHERE id = ?",
                    (r["amount"], r["tenant_id"]),
                )
            conn.execute("DELETE FROM ledger WHERE id > ?", (commit_mark,))

        while rounds < settings.max_rounds:
            fitting = [
                tid for tid in round_tenants
                if (h := head_ready(tid)) is not None and h["gpus"] <= free
            ]
            if not fitting:
                break  # every ready job is capacity-blocked; no grants

            rounds += 1
            dispatched_this_round = False

            for tenant_id in fitting:
                if free <= 0:
                    break
                head = head_ready(tenant_id)
                if head is None or head["gpus"] > free:
                    # Became ineligible after earlier tenants in this round.
                    continue

                weight = conn.execute(
                    "SELECT weight FROM tenants WHERE id = ?", (tenant_id,)
                ).fetchone()["weight"]
                quantum = settings.base_quantum_gpu_min * float(weight)
                _credit(
                    conn, tenant_id, quantum, "QUANTUM", now_str,
                    batch_id=batch_id, detail=f"round {rounds}",
                )

                bal = conn.execute(
                    "SELECT deficit FROM tenants WHERE id = ?", (tenant_id,)
                ).fetchone()["deficit"]
                cost = head["cost_gpu_min"]

                if bal + _EPS < cost:
                    continue  # keep accruing in later rounds
                if head["gpus"] > free:
                    continue  # capacity invariant: never crossed

                run_index += 1
                conn.execute(
                    """UPDATE jobs
                          SET status = ?, scheduled_at = ?, batch_id = ?,
                              run_index = ?
                        WHERE id = ?""",
                    (RUNNING, now_str, batch_id, run_index, head["id"]),
                )
                _credit(
                    conn, tenant_id, -cost, "DISPATCH", now_str,
                    batch_id=batch_id, job_id=head["id"],
                    detail=f"{head['gpus']} GPU x {head['duration_min']} min",
                )
                selected.append(head["id"])
                free -= head["gpus"]
                dispatched_this_round = True

            if dispatched_this_round:
                # All accrual that led to this dispatch is now permanent.
                commit_mark = conn.execute(
                    "SELECT COALESCE(MAX(id), 0) AS m FROM ledger"
                ).fetchone()["m"]
            # Idle rounds are normal while tenants are still accruing toward
            # affordability (positive weights guarantee they get there); the
            # loop continues until dispatch or no job fits any longer.

        # Grants above the commit watermark never led to a dispatch (idle
        # trailing rounds, or a batch cut off by max_rounds): roll them back
        # so that invoking the endpoint while blocked leaves balances and the
        # ledger untouched.
        rollback_uncommitted()

        # Close aging windows for tenants whose queue drained; others keep
        # accruing wait time from this batch's reference instant.
        for tenant_id in active_ids:
            if _has_queued(conn, tenant_id):
                conn.execute(
                    "UPDATE tenants SET aging_settled_at = ? WHERE id = ?",
                    (now_str, tenant_id),
                )
            else:
                conn.execute(
                    "UPDATE tenants SET aging_settled_at = NULL WHERE id = ?",
                    (tenant_id,),
                )

        conn.execute(
            "UPDATE batches SET rounds = ?, jobs_selected = ? WHERE id = ?",
            (rounds, len(selected), batch_id),
        )
        conn.commit()
        return _batch_result(conn, batch_id, settings)


def _batch_result(
    conn: sqlite3.Connection,
    batch_id: int,
    settings: Settings,
) -> dict[str, Any]:
    batch = conn.execute(
        "SELECT * FROM batches WHERE id = ?", (batch_id,)
    ).fetchone()
    jobs = conn.execute(
        """SELECT * FROM jobs WHERE batch_id = ? ORDER BY run_index""",
        (batch_id,),
    ).fetchall()
    used = conn.execute(
        "SELECT COALESCE(SUM(gpus), 0) AS n FROM jobs WHERE status = ?",
        (RUNNING,),
    ).fetchone()["n"]
    return {
        "batch_id": batch["id"],
        "created_at": batch["created_at"],
        "capacity_gpus": batch["capacity_gpus"],
        "used_gpus_before": batch["used_gpus_before"],
        "available_gpus_at_start": batch["available_gpus"],
        "used_gpus_after": used,
        "free_gpus_after": settings.cluster_gpus - used,
        "rounds": batch["rounds"],
        "jobs_selected": len(jobs),
        "selection_order": [j["id"] for j in jobs],
        "selected_jobs": [_job_dict(conn, j) | {"run_index": j["run_index"]}
                          for j in jobs],
        "tenants": _list_tenants(conn),
        "events": _list_ledger(conn, batch_id=batch_id),
    }


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def list_jobs(
    status: str | None = None,
    tenant_id: str | None = None,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM jobs WHERE 1=1"
    params: list[Any] = []
    if status:
        valid = {QUEUED, RUNNING, COMPLETED, CANCELLED}
        if status not in valid:
            raise SchedulerError(f"status must be one of {sorted(valid)}")
        sql += " AND status = ?"
        params.append(status)
    if tenant_id:
        sql += " AND tenant_id = ?"
        params.append(tenant_id)
    sql += " ORDER BY queued_at, id"
    with db.get_lock():
        conn = db.get_conn()
        return [_job_dict(conn, r) for r in conn.execute(sql, params)]


def queue_overview() -> dict[str, Any]:
    """Return per-tenant FIFO queues with positions, weights and balances."""
    with db.get_lock():
        conn = db.get_conn()
        settings = get_settings()
        used = conn.execute(
            "SELECT COALESCE(SUM(gpus), 0) AS n FROM jobs WHERE status = ?",
            (RUNNING,),
        ).fetchone()["n"]

        tenants = conn.execute(
            _TENANT_SELECT + " ORDER BY t.id"
        ).fetchall()
        groups = []
        for t in tenants:
            queued = conn.execute(
                """SELECT * FROM jobs WHERE tenant_id = ? AND status = ?
                   ORDER BY queued_at, id""",
                (t["id"], QUEUED),
            ).fetchall()
            if not queued:
                continue
            groups.append({
                "tenant_id": t["id"],
                "weight": round(t["weight"], 6),
                "balance_gpu_min": round(t["deficit"], 6),
                "oldest_queued_at": queued[0]["queued_at"],
                "head_blocked_by": _unmet_dependencies(
                    conn, queued[0]["id"]
                ),
                "jobs": [
                    _job_dict(conn, j) | {"position": pos}
                    for pos, j in enumerate(queued, start=1)
                ],
            })

        total_queued = conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status = ?", (QUEUED,)
        ).fetchone()["n"]
        total_running = conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status = ?", (RUNNING,)
        ).fetchone()["n"]
        return {
            "cluster_gpus": settings.cluster_gpus,
            "used_gpus": used,
            "free_gpus": settings.cluster_gpus - used,
            "queued_jobs": total_queued,
            "running_jobs": total_running,
            "queues": groups,
        }


def scheduler_state() -> dict[str, Any]:
    with db.get_lock():
        conn = db.get_conn()
        settings = get_settings()
        used = conn.execute(
            "SELECT COALESCE(SUM(gpus), 0) AS n FROM jobs WHERE status = ?",
            (RUNNING,),
        ).fetchone()["n"]
        rows = conn.execute(_TENANT_SELECT + " ORDER BY t.id").fetchall()
        return {
            "cluster_gpus": settings.cluster_gpus,
            "used_gpus": used,
            "free_gpus": settings.cluster_gpus - used,
            "settings": {
                "base_quantum_gpu_min": settings.base_quantum_gpu_min,
                "aging_rate_gpu_min_per_min":
                    settings.aging_rate_gpu_min_per_min,
                "aging_cap_gpu_min": settings.aging_cap_gpu_min,
            },
            "tenants": [_tenant_dict(conn, r) for r in rows],
        }


def _list_ledger(
    conn: sqlite3.Connection,
    tenant_id: str | None = None,
    batch_id: int | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM ledger WHERE 1=1"
    params: list[Any] = []
    if tenant_id:
        sql += " AND tenant_id = ?"
        params.append(tenant_id)
    if batch_id is not None:
        sql += " AND batch_id = ?"
        params.append(batch_id)
    sql += " ORDER BY id ASC LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [
        {
            "id": r["id"],
            "tenant_id": r["tenant_id"],
            "batch_id": r["batch_id"],
            "job_id": r["job_id"],
            "kind": r["kind"],
            "amount": round(r["amount"], 6),
            "balance_after": round(r["balance_after"], 6),
            "detail": r["detail"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def list_ledger(
    tenant_id: str | None = None,
    batch_id: int | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    if limit <= 0 or limit > 10_000:
        raise SchedulerError("limit must be between 1 and 10000")
    with db.get_lock():
        return _list_ledger(
            db.get_conn(), tenant_id=tenant_id, batch_id=batch_id, limit=limit
        )
