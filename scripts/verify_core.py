#!/usr/bin/env python3
"""Dependency-free smoke test for the scheduling core (stdlib sqlite3 only).

Run:  python3 scripts/verify_core.py

It spins an in-memory database, exercises weighted DRR, aging, capacity
limits, cancellation and reproducibility, and asserts the exact outcomes.
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import db  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app import scheduler  # noqa: E402

EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def at(m: float) -> datetime:
    return EPOCH + timedelta(minutes=m)


def iso(m: float) -> str:
    return at(m).isoformat()


def main() -> int:
    os.environ.update(
        DB_PATH=":memory:",
        CLUSTER_GPUS="8",
        BASE_QUANTUM_GPU_MIN="60",
        AGING_RATE_GPU_MIN_PER_MIN="1",
    )
    get_settings.cache_clear()

    db.reset_for_tests()
    db.connect(":memory:")

    # --- Case 1: WDRR selection order -------------------------------------
    def submit(tid, gpus, mins, weight, when):
        job = scheduler.submit_job(tid, gpus, mins, weight)
        db.get_conn().execute(
            "UPDATE jobs SET queued_at = ? WHERE id = ?", (when, job["id"]))
        return job

    submit("a", 1, 30, 1.0, iso(0))
    for i in range(3):
        submit("b", 1, 30, 3.0, iso(1 + i))

    r = scheduler.run_batch(now=at(10))
    assert r["selection_order"] == ["J000001", "J000002", "J000003", "J000004"], r["selection_order"]
    bal = {t["id"]: t["balance_gpu_min"] for t in r["tenants"]}
    # a: aging 10 + quantum 60 - cost 30 = 40
    # b: aging ~9 + 3 quanta x 180 - cost 90 = 459
    assert abs(bal["a"] - 40.0) < 1e-9, bal
    assert abs(bal["b"] - 459.0) < 1e-9, bal
    aging = {e["tenant_id"]: e["amount"]
             for e in r["events"] if e["kind"] == "AGING"}
    assert abs(aging["a"] - 10.0) < 1e-9
    assert abs(aging["b"] - 9.0) < 1e-9
    print("case 1 weighted DRR order/balance OK")

    # fresh DB for capacity case
    db.reset_for_tests()
    db.connect(":memory:")

    # --- Case 2: capacity never exceeded; blocked keeps credit ------------
    submit("a", 6, 10, 1.0, iso(0))
    submit("b", 6, 10, 1.0, iso(1))
    r1 = scheduler.run_batch(now=at(5))
    assert r1["selection_order"] == ["J000001"]
    r2 = scheduler.run_batch(now=at(6))
    assert r2["selection_order"] == []  # b blocked on capacity
    b_bal_before = scheduler.get_tenant("b")["balance_gpu_min"]
    # no quantum farming on no-op repeated batch
    r2b = scheduler.run_batch(now=at(6))
    assert r2b["selection_order"] == []
    assert scheduler.get_tenant("b")["balance_gpu_min"] == b_bal_before
    scheduler.complete_job("J000001")
    r3 = scheduler.run_batch(now=at(7))
    assert r3["selection_order"] == ["J000002"]
    assert scheduler.scheduler_state()["used_gpus"] == 6
    print("case 2 capacity constraint + retained credit OK")

    # --- Case 3: multi-round accrual within one batch + cross-batch carry --
    db.reset_for_tests()
    db.connect(":memory:")
    # Two large 6-GPU jobs, cost 600 each (6 x 100), quantum 60/round ->
    # each needs 10 rounds before it can dispatch.
    submit("a", 6, 100, 1.0, iso(0))
    submit("b", 6, 100, 1.0, iso(1))
    r = scheduler.run_batch(now=at(10))
    # Aging applies from the jobs' arrivals too: a waited 10, b 9 min.
    assert r["selection_order"] == ["J000001"], r["selection_order"]
    assert r["rounds"] == 10, r["rounds"]
    assert abs(scheduler.get_tenant("a")["balance_gpu_min"] - 10.0) < 1e-9
    assert abs(scheduler.get_tenant("b")["balance_gpu_min"] - 549.0) < 1e-9

    # No farming while purely capacity-blocked (same instant -> no aging):
    blocked = scheduler.run_batch(now=at(10))
    assert blocked["selection_order"] == []
    assert abs(scheduler.get_tenant("b")["balance_gpu_min"] - 549.0) < 1e-9
    assert all(e["kind"] != "QUANTUM" for e in blocked["events"])

    # Capacity frees up: 549 + 2 more aging + 60 quantum - 600 cost = 11.
    scheduler.complete_job("J000001")
    r2 = scheduler.run_batch(now=at(12))
    assert r2["selection_order"] == ["J000002"]
    assert abs(scheduler.get_tenant("b")["balance_gpu_min"] - 11.0) < 1e-9
    aging = [e for e in r2["events"] if e["kind"] == "AGING"]
    assert abs(aging[0]["amount"] - 2.0) < 1e-9
    print("case 3 multi-round accrual + cross-batch carry OK")

    # --- Case 4: aging compensates a waiting small job ---------------------
    db.reset_for_tests()
    db.connect(":memory:")
    # Heavy tenant 'a' (sorts first) saturates the cluster with an 8-GPU job;
    # small tenant 'b' has a tiny job that cannot fit afterwards.
    submit("a", 8, 10, 4.0, iso(0))   # cost 80, quantum 240 -> runs now
    submit("b", 1, 60, 1.0, iso(0))  # cost 60, blocked by capacity
    r1 = scheduler.run_batch(now=at(10))
    assert r1["selection_order"] == ["J000001"], r1["selection_order"]
    assert scheduler.scheduler_state()["used_gpus"] == 8
    # a: aging 10 + quantum 240 - cost 80 = 170.
    # b got no quantum (capacity-blocked) but keeps aging credit 10.
    assert abs(scheduler.get_tenant("a")["balance_gpu_min"] - 170.0) < 1e-9
    assert abs(scheduler.get_tenant("b")["balance_gpu_min"] - 10.0) < 1e-9
    scheduler.complete_job("J000001")
    r2 = scheduler.run_batch(now=at(120))
    assert r2["selection_order"] == ["J000002"], r2["selection_order"]
    # b: 10 carried aging + 110 more + quantum 60 - cost 60 = 120.
    assert abs(scheduler.get_tenant("b")["balance_gpu_min"] - 120.0) < 1e-9
    aging = [e for e in r2["events"] if e["kind"] == "AGING"]
    assert aging and abs(aging[0]["amount"] - 110.0) < 1e-6, aging
    # aging window resets after drain
    assert scheduler.get_tenant("b")["aging_settled_at"] is None
    print("case 4 aging compensation + window reset OK")

    # --- Case 5: oversized jobs rejected ----------------------------------
    db.reset_for_tests()
    db.connect(":memory:")
    try:
        scheduler.submit_job("a", 99, 1, 1.0)
    except scheduler.SchedulerError:
        pass
    else:
        raise AssertionError("oversized job should be rejected")
    print("case 5 oversized rejection OK")

    # --- Case 6: queue + cancel -------------------------------------------
    submit("a", 2, 10, 1.0, iso(0))
    submit("a", 2, 10, 1.0, iso(1))
    r = scheduler.run_batch(now=at(2))
    assert r["selection_order"] == ["J000001", "J000002"]
    scheduler.cancel_job("J000002")
    assert scheduler.scheduler_state()["used_gpus"] == 2
    q = scheduler.queue_overview()
    assert q["running_jobs"] == 1 and q["queued_jobs"] == 0
    print("case 6 cancel/queue state OK")

    # --- Case 7: job dependencies (cross-tenant, blocking, release) --------
    db.reset_for_tests()
    db.connect(":memory:")
    pre = submit("a", 1, 10, 1.0, iso(0))
    dep = scheduler.submit_job("b", 1, 10, 1.0, depends_on=[pre["id"]])
    db.get_conn().execute(
        "UPDATE jobs SET queued_at = ? WHERE id = ?", (iso(1), dep["id"]))
    dep = scheduler._get_job_or_404(db.get_conn(), dep["id"])
    assert dep["depends_on"] == ["J000001"]
    assert dep["blocked_by"] == [{"id": "J000001", "status": "QUEUED"}]

    # Unknown prerequisite rejected.
    try:
        scheduler.submit_job("c", 1, 10, depends_on=["J000999"])
    except scheduler.SchedulerError as e:
        assert "unknown depends_on" in str(e)
    else:
        raise AssertionError("unknown dependency should be rejected")

    r = scheduler.run_batch(now=at(10))
    assert r["selection_order"] == ["J000001"], r["selection_order"]
    # Blocked tenant earns nothing (no aging, no quantum).
    assert scheduler.get_tenant("b")["balance_gpu_min"] == 0.0
    assert all(e["tenant_id"] != "b" for e in r["events"])
    assert scheduler.run_batch(now=at(20))["selection_order"] == []

    scheduler.complete_job("J000001")
    r2 = scheduler.run_batch(now=at(30))
    assert r2["selection_order"] == ["J000002"], r2["selection_order"]

    # Cancelled prerequisite blocks forever, with a queryable reason.
    p3 = scheduler.submit_job("a", 1, 10)
    d3 = scheduler.submit_job("b", 1, 10, depends_on=[p3["id"]])
    scheduler.cancel_job(p3["id"])
    assert scheduler.run_batch(now=at(40))["selection_order"] == []
    assert scheduler.get_job(d3["id"])["blocked_by"] == [
        {"id": p3["id"], "status": "CANCELLED"}]
    print("case 7 dependencies + blocked_by OK")

    print("\nALL CORE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
