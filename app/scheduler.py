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

WDRR per batch
==============
1. Each active tenant (one with QUEUED jobs, tenant id order) receives
   ``base_quantum * weight`` GPU-minutes of credit every DRR round.
2. Visiting tenants in fixed order, the head-of-line job is dispatched when
   ``balance >= job_cost`` AND enough free GPUs exist.
3. A full round in which nothing is dispatched ends the batch: blocked jobs
   keep their credit, and a later batch (after completions or aging) retries.

Aging compensation
==================
While a tenant has a queued job, elapsed wall-clock time since its last
settlement is converted at ``AGING_RATE`` into extra credit (optionally
capped per batch). When the tenant's queue drains the aging clock resets, so
a later freshly submitted small job starts a clean wait window. Aging lets
small jobs behind large, high-weight tenants accumulate credit until they
become affordable, bounding starvation.

Reproducibility
===============
Tenant visit order is tenant-id ascending, jobs are FIFO per tenant, job ids
are sequential, and ``run_batch`` accepts an injectable ``now``/seed-free
logic: identical inputs always produce identical selection order.

Job dependencies
================
``POST /jobs`` accepts ``depends_on``: a list of job ids (from ANY tenant)
that must be COMPLETED before the new job becomes schedulable. Unknown ids
and dependency cycles (including self-dependence) are rejected at submit.
Submission therefore never closes a cycle, but the whole graph is rechecked
defensively. During a batch, a tenant is visited only when its head-of-line
job's prerequisites are all COMPLETED; a dependency-blocked head receives
neither quantum nor aging credit (its aging checkpoint is merely advanced),
so blocked dependencies can never be farmed into spendable balance. A
CANCELLED prerequisite can never complete: dependents stay QUEUED forever
and ``blocked_by`` (on the job object and in ``GET /queue``) reports the
prerequisite id together with its current status.
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
# Job dependencies (DAG)
# ---------------------------------------------------------------------------

def _depends_on(conn: sqlite3.Connection, job_id: str) -> list[str]:
    """Prerequisite job ids in submission order (id ascending)."""
    rows = conn.execute(
        """SELECT depends_on FROM job_dependencies
            WHERE job_id = ? ORDER BY depends_on""",
        (job_id,),
    ).fetchall()
    return [r["depends_on"] for r in rows]


def _unmet_prerequisites(
    conn: sqlite3.Connection, job_id: str
) -> list[dict[str, str]]:
    """Prerequisites that have not COMPLETED, with the reason (their status).

    QUEUED/RUNNING prerequisites may still release the job; a CANCELLED one
    can never complete, so the dependent stays queued forever and the entry
    here explains why.
    """
    rows = conn.execute(
        """SELECT j.id AS id, j.status AS status
             FROM job_dependencies d
             JOIN jobs j ON j.id = d.depends_on
            WHERE d.job_id = ? AND j.status != ?
            ORDER BY j.id""",
        (job_id, COMPLETED),
    ).fetchall()
    return [{"id": r["id"], "status": r["status"]} for r in rows]


def _find_cycle(
    conn: sqlite3.Connection, extra_edges: dict[str, list[str]] | None = None
) -> list[str] | None:
    """Return a job-id cycle in the dependency graph, or None when it is a DAG.

    ``extra_edges`` maps a (possibly not-yet-inserted) job id to its
    prerequisites, so a candidate submission can be validated against the
    graph it would create. Iterative DFS with WHITE/GRAY/BLACK coloring.
    """
    extra_edges = extra_edges or {}
    graph: dict[str, list[str]] = {}
    rows = conn.execute(
        "SELECT job_id, depends_on FROM job_dependencies"
    ).fetchall()
    for r in rows:
        graph.setdefault(r["job_id"], []).append(r["depends_on"])
        graph.setdefault(r["depends_on"], [])
    for node, prereqs in extra_edges.items():
        graph.setdefault(node, []).extend(prereqs)
        for p in prereqs:
            graph.setdefault(p, [])

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {node: WHITE for node in graph}
    for start in sorted(graph):
        if color.get(start, BLACK) != WHITE:
            continue
        # Stack of (node, next-neighbour-index); parallel pos[] stack records
        # the path so the GRAY edge closing a cycle names the exact nodes.
        stack: list[str] = [start]
        pos: list[int] = [0]
        color[start] = GRAY
        while stack:
            node = stack[-1]
            neighbours = graph[node]
            i = pos[-1]
            if i < len(neighbours):
                nxt = neighbours[i]
                pos[-1] = i + 1
                if color.get(nxt, WHITE) == GRAY:
                    cycle = stack[stack.index(nxt):] + [nxt]
                    return cycle
                if color.get(nxt, WHITE) == WHITE:
                    color[nxt] = GRAY
                    stack.append(nxt)
                    pos.append(0)
            else:
                color[node] = BLACK
                stack.pop()
                pos.pop()
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
    if depends_on is None:
        depends_on = []
    if not isinstance(depends_on, list) or any(
        not isinstance(d, str) or not d.strip() for d in depends_on
    ):
        raise SchedulerError(
            "depends_on must be a list of non-empty job id strings"
        )
    depends_on = sorted(set(depends_on))

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
        # Auto-register unknown tenants so a single POST /jobs is enough.
        if conn.execute(
            "SELECT 1 FROM tenants WHERE id = ?", (tenant_id,)
        ).fetchone() is None:
            conn.execute(
                """INSERT INTO tenants
                       (id, weight, deficit, aging_settled_at,
                        created_at, updated_at)
                   VALUES (?, ?, 0, NULL, ?, ?)""",
                (tenant_id, float(weight), now, now),
            )
        else:
            # Latest submission may adjust the tenant's weight.
            conn.execute(
                "UPDATE tenants SET weight = ?, updated_at = ? WHERE id = ?",
                (float(weight), now, tenant_id),
            )

        job_id = _job_id(conn)

        # Reject unknown prerequisites. Prerequisites may belong to other
        # tenants: dependencies are cluster-global, not tenant-scoped.
        if depends_on:
            placeholders = ",".join("?" for _ in depends_on)
            found = {
                r["id"] for r in conn.execute(
                    f"SELECT id FROM jobs WHERE id IN ({placeholders})",
                    depends_on,
                ).fetchall()
            }
            missing = [
                d for d in depends_on
                if d not in found and d != job_id
            ]
            if missing:
                raise SchedulerError(
                    f"unknown depends_on job id(s): {', '.join(missing)}"
                )

        # Reject anything that would close a cycle (including self
        # dependencies such as depends_on == the new job's own id).
        cycle = _find_cycle(conn, {job_id: depends_on})
        if cycle is not None:
            raise SchedulerError(
                "dependency cycle detected: " + " -> ".join(cycle)
            )

        conn.execute(
            """INSERT INTO jobs
                   (id, tenant_id, gpus, duration_min, cost_gpu_min, status,
                    queued_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (job_id, tenant_id, gpus, duration_min, cost, QUEUED, now),
        )
        for prereq in depends_on:
            conn.execute(
                "INSERT INTO job_dependencies (job_id, depends_on) VALUES (?, ?)",
                (job_id, prereq),
            )
        # aging_settled_at is left untouched: a non-empty queue keeps accruing
        # wait time, and a drained queue already has it reset to NULL.
        return _get_job_or_404(conn, job_id)


def _get_job_or_404(conn: sqlite3.Connection, job_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise NotFound(f"job '{job_id}' not found")
    return _job_dict(conn, row)


def _job_dict(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "tenant_id": row["tenant_id"],
        "gpus": row["gpus"],
        "duration_min": row["duration_min"],
        "cost_gpu_min": round(row["cost_gpu_min"], 6),
        "status": row["status"],
        "depends_on": _depends_on(conn, row["id"]),
        # Present for QUEUED jobs: prerequisites that are not COMPLETED yet,
        # each annotated with its current status (QUEUED/RUNNING/CANCELLED).
        # Empty means every prerequisite is done and the job is schedulable.
        "blocked_by": (
            _unmet_prerequisites(conn, row["id"])
            if row["status"] == QUEUED else []
        ),
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
            """SELECT DISTINCT j.tenant_id, t.weight, t.deficit,
                              t.aging_settled_at
                 FROM jobs j JOIN tenants t ON t.id = j.tenant_id
                WHERE j.status = ?
                ORDER BY j.tenant_id""",
            (QUEUED,),
        ).fetchall()

        if not active:
            conn.execute(
                "UPDATE batches SET rounds = 0, jobs_selected = 0 WHERE id = ?",
                (batch_id,),
            )
            conn.commit()
            return _batch_result(conn, batch_id, settings)

        # Dependency readiness is recomputed from the *current* head of each
        # queue: only a COMPLETED prerequisite releases a job, and a tenant's
        # next head (revealed after a dispatch) may itself be blocked. A
        # tenant whose head-of-line job is blocked neither accrues aging nor
        # spends quantum: waiting on another job is not scheduler starvation,
        # and blocked batches must not hand out spendable credit.
        def head_of(tenant_id: str):
            return conn.execute(
                """SELECT * FROM jobs
                    WHERE tenant_id = ? AND status = ?
                    ORDER BY queued_at, id LIMIT 1""",
                (tenant_id, QUEUED),
            ).fetchone()

        def head_ready(head) -> bool:
            return head is not None and not _unmet_prerequisites(
                conn, head["id"]
            )

        active_ids = [t["tenant_id"] for t in active]
        ready_at_start = {
            tid for tid in active_ids if head_ready(head_of(tid))
        }

        # Aging is settled even when the cluster is full, so waiting tenants
        # keep accumulating starvation compensation across blocked batches.
        # --- Aging settlement at batch start -------------------------------
        # Compensation covers the full wait since the oldest queued job's
        # arrival: on first observation we settle from queued_at, afterwards
        # from the previous settlement instant. Same-instant retries grant
        # nothing, so the endpoint cannot be farmed via aging either.
        # Dependency-blocked tenants receive nothing, but their checkpoint is
        # advanced: once dependencies release them, waiting is measured from
        # the release point rather than the original submission.
        for t in active:
            if t["tenant_id"] not in ready_at_start:
                conn.execute(
                    "UPDATE tenants SET aging_settled_at = ?, updated_at = ? "
                    "WHERE id = ?",
                    (now_str, now_str, t["tenant_id"]),
                )
                continue
            oldest = conn.execute(
                "SELECT MIN(queued_at) AS ts FROM jobs "
                "WHERE tenant_id = ? AND status = ?",
                (t["tenant_id"], QUEUED),
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
                    conn, t["tenant_id"], bonus, "AGING", now_str,
                    batch_id=batch_id,
                    detail=f"waited {waited_min:.3f} min",
                )
            conn.execute(
                "UPDATE tenants SET aging_settled_at = ?, updated_at = ? "
                "WHERE id = ?",
                (now_str, now_str, t["tenant_id"]),
            )

        if available <= 0:
            conn.execute(
                "UPDATE batches SET rounds = 0, jobs_selected = 0 WHERE id = ?",
                (batch_id,),
            )
            conn.commit()
            return _batch_result(conn, batch_id, settings)

        # --- WDRR rounds ----------------------------------------------------
        # A tenant is visited in a round only while its head job FITS the
        # currently free capacity. Fitting-but-unaffordable jobs keep
        # accumulating quantum across rounds until affordable. A tenant whose
        # head job cannot physically fit (capacity-blocked) accrues nothing:
        # waiting for completions is compensated by aging, and this prevents
        # callers farming quantum by repeatedly invoking the endpoint.
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
            fitting = []
            for tid in active_ids:
                h = head_of(tid)
                if h is not None and head_ready(h) and h["gpus"] <= free:
                    fitting.append(tid)
            if not fitting:
                break  # every queued job is capacity/dependency-blocked

            rounds += 1
            dispatched_this_round = False

            for tenant_id in fitting:
                if free <= 0:
                    break
                head = head_of(tenant_id)
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
