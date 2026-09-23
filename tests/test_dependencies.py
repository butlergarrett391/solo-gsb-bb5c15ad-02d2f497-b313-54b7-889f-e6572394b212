"""Job dependency tests: validation, blocking, release, cancellation."""
import pytest

from app import db, scheduler
from tests.conftest import at, iso  # noqa: F401


def submit(tenant, gpus, minutes, weight=1.0, when=None, depends_on=None):
    conn = db.get_conn()
    prior = conn.execute(
        "SELECT aging_settled_at AS ts FROM tenants WHERE id = ?",
        (tenant,),
    ).fetchone()
    job = scheduler.submit_job(
        tenant, gpus, minutes, weight, depends_on=depends_on
    )
    if when is not None:
        conn.execute(
            "UPDATE jobs SET queued_at = ? WHERE id = ?", (when, job["id"])
        )
        if prior is None or prior["ts"] is None:
            conn.execute(
                "UPDATE tenants SET aging_settled_at = ? WHERE id = ?",
                (when, tenant),
            )
    return scheduler._get_job_or_404(conn, job["id"])


def batch(now_minutes=100):
    return scheduler.run_batch(now=at(now_minutes))


# ---------------------------------------------------------------------------
# Submission validation
# ---------------------------------------------------------------------------

def test_missing_dependency_rejected(conn):
    with pytest.raises(scheduler.SchedulerError) as ei:
        submit("a", 1, 5, depends_on=["J999999"])
    assert "J999999" in str(ei.value)
    # Nothing was created.
    assert scheduler.list_jobs() == []
    assert scheduler.list_tenants() == []


def test_self_dependency_rejected(conn):
    # Self-loops cannot arise through the id-montonic API (a new id never
    # exists yet), but the validator must still reject one, e.g. when a
    # self-edge was injected out of band and a later job reaches it.
    a = submit("a", 1, 5, when=iso(0))
    db.get_conn().execute(
        "INSERT INTO job_dependencies (job_id, depends_on, created_at) "
        "VALUES (?, ?, ?)",
        (a["id"], a["id"], iso(1)),
    )
    b = submit("b", 1, 5, when=iso(2))
    with pytest.raises(scheduler.SchedulerError) as ei:
        # Re-validate: a new edge reaching a self-loop closes a cycle.
        scheduler.submit_job("c", 1, 5, depends_on=[b["id"], a["id"]])
    assert "cycle" in str(ei.value).lower()


def test_duplicate_dependencies_deduplicated(conn):
    a = submit("a", 1, 5, when=iso(0))
    b = submit("b", 1, 5, when=iso(1), depends_on=[a["id"], a["id"]])
    assert b["depends_on"] == [a["id"]]


def test_empty_dependency_id_rejected(conn):
    submit("a", 1, 5)
    with pytest.raises(scheduler.SchedulerError):
        submit("b", 1, 5, depends_on=[""])


def test_cycle_detected_against_preexisting_graph(conn):
    # A -> B, then a third job C tries to depend on a node reachable through
    # a cycle injected out-of-band. This simulates a graph that would close
    # a cycle from the new edges' reachability.
    a = submit("a", 1, 5, when=iso(0))
    b = submit("b", 1, 5, when=iso(1), depends_on=[a["id"]])
    # Inject a back edge B ... nothing can point back through the API because
    # ids always increase, but a cycle in stored data must still be detected:
    # add edge A -> B directly.
    db.get_conn().execute(
        "INSERT INTO job_dependencies (job_id, depends_on, created_at) "
        "VALUES (?, ?, ?)",
        (a["id"], b["id"], iso(2)),
    )
    # C depends on A; walking A -> B -> A reaches a (gray) node.
    with pytest.raises(scheduler.SchedulerError) as ei:
        submit("c", 1, 5, depends_on=[a["id"]])
    assert "cycle" in str(ei.value).lower()


def test_chain_and_diamond_accepted(conn):
    a = submit("a", 1, 5, when=iso(0))
    b = submit("a", 1, 5, when=iso(1), depends_on=[a["id"]])
    c = submit("a", 1, 5, when=iso(2), depends_on=[a["id"]])
    d = submit("a", 1, 5, when=iso(3), depends_on=[b["id"], c["id"]])
    assert d["depends_on"] == [b["id"], c["id"]]


# ---------------------------------------------------------------------------
# Scheduling gating
# ---------------------------------------------------------------------------

def test_dependent_job_not_selected_until_prereq_completed(conn):
    a = submit("a", 1, 5, when=iso(0))
    b = submit("b", 1, 5, when=iso(1), depends_on=[a["id"]])

    r = batch(10)
    assert r["selection_order"] == [a["id"]]
    # b stays QUEUED and reports the blocker.
    bj = scheduler.get_job(b["id"])
    assert bj["status"] == "QUEUED"
    assert [x["job_id"] for x in bj["blocked_by"]] == [a["id"]]
    assert bj["blocked_by"][0]["status"] == "RUNNING"

    # A second batch still cannot pick b (prereq RUNNING), and must not
    # grant b any quantum.
    r2 = batch(11)
    assert r2["selection_order"] == []
    assert all(
        e["tenant_id"] != "b" for e in r2["events"]
    ), r2["events"]
    assert scheduler.get_tenant("b")["balance_gpu_min"] == 0.0

    # Once the prerequisite completes, the dependent becomes eligible.
    scheduler.complete_job(a["id"])
    r3 = batch(12)
    assert r3["selection_order"] == [b["id"]]


def test_dependent_does_not_consume_quantum_or_aging_while_blocked(conn):
    # Cheap prerequisite saturates the cluster? Use a large-ish dependent
    # whose tenant would otherwise accrue heavily.
    a = submit("a", 1, 5, when=iso(0))
    b = submit("b", 2, 100, when=iso(0), depends_on=[a["id"]])  # cost 200

    r = batch(60)
    assert r["selection_order"] == [a["id"]]
    # No QUANTUM/AGING for b while dependency-blocked, despite 60 min passing.
    b_events = [e for e in r["events"] if e["tenant_id"] == "b"]
    assert b_events == []
    assert scheduler.get_tenant("b")["balance_gpu_min"] == 0.0

    # Blocked interval must not pay out later: complete prereq at minute 120;
    # b's aging clock was advanced through the block, so only wait since the
    # last blocked batch counts.
    scheduler.complete_job(a["id"])
    r2 = scheduler.run_batch(now=at(120))
    assert r2["selection_order"] == [b["id"]]
    aging = [
        e for e in r2["events"]
        if e["tenant_id"] == "b" and e["kind"] == "AGING"
    ]
    # Wait counted only from the previous batch instant (60), not arrival (0).
    assert aging and aging[0]["amount"] == pytest.approx(60.0)


def test_dependency_head_blocks_rest_of_tenant_queue_fifo(conn):
    # Tenant t: head depends on unfinished p; the second t job is independent
    # but must wait behind the blocked head (strict per-tenant FIFO).
    p = submit("p", 1, 5, when=iso(0))
    submit("t", 1, 5, when=iso(1), depends_on=[p["id"]])   # head, blocked
    t2 = submit("t", 1, 5, when=iso(2))                    # behind head

    r = batch(10)
    assert r["selection_order"] == [p["id"]]
    assert all(j["tenant_id"] != "t" for j in r["selected_jobs"])

    scheduler.complete_job(p["id"])
    r2 = batch(11)
    # Both t jobs now run, in FIFO order (head first).
    ids = [j["id"] for j in r2["selected_jobs"] if j["tenant_id"] == "t"]
    assert ids == ["J000002", t2["id"]]


def test_cancelled_prerequisite_keeps_dependent_blocked_forever(conn):
    a = submit("a", 1, 5, when=iso(0))
    b = submit("b", 1, 5, when=iso(1), depends_on=[a["id"]])

    # Cancel the prerequisite before it ever runs.
    scheduler.cancel_job(a["id"])

    r = batch(10)
    assert r["selection_order"] == []
    bj = scheduler.get_job(b["id"])
    assert bj["status"] == "QUEUED"
    blocker = bj["blocked_by"][0]
    assert blocker["job_id"] == a["id"]
    assert blocker["status"] == "CANCELLED"
    assert "CANCELLED" in blocker["reason"]

    # Repeated batches never release it and never grant quantum.
    r2 = batch(200)
    assert r2["selection_order"] == []
    assert scheduler.get_tenant("b")["balance_gpu_min"] == 0.0

    # The dependent can then be cancelled itself, freeing the tenant queue.
    scheduler.cancel_job(b["id"])
    assert scheduler.queue_overview()["queued_jobs"] == 0


def test_prerequisite_cancelled_after_running_also_blocks(conn):
    a = submit("a", 1, 5, when=iso(0))
    b = submit("b", 1, 5, when=iso(1), depends_on=[a["id"]])
    batch(10)
    scheduler.cancel_job(a["id"])  # RUNNING -> CANCELLED

    r = batch(20)
    assert r["selection_order"] == []
    blocker = scheduler.get_job(b["id"])["blocked_by"][0]
    assert blocker["status"] == "CANCELLED"


# ---------------------------------------------------------------------------
# Cross-tenant dependencies
# ---------------------------------------------------------------------------

def test_cross_tenant_dependency(conn):
    # upstream tenant u holds the prerequisite; downstream tenant d waits.
    p = submit("u", 1, 5, when=iso(0))
    d1 = submit("d", 1, 5, when=iso(1), depends_on=[p["id"]])

    q = scheduler.queue_overview()
    groups = {g["tenant_id"]: g for g in q["queues"]}
    assert [x["job_id"] for x in groups["d"]["head_blocked_by"]] == [p["id"]]
    assert groups["d"]["jobs"][0]["blocked_by"][0]["job_id"] == p["id"]

    r = batch(10)
    assert r["selection_order"] == [p["id"]]
    # The downstream tenant appears with a queued job but got no dispatch.
    assert {t["id"]: t for t in r["tenants"]}["d"]["queued_jobs"] == 1
    assert {t["id"]: t for t in r["tenants"]}["d"]["balance_gpu_min"] == 0.0

    scheduler.complete_job(p["id"])
    r2 = batch(11)
    assert r2["selection_order"] == [d1["id"]]


def test_diamond_release_schedules_in_fifo_order(conn):
    # p1,p2 (any tenants) -> d waits on both; releases only after BOTH done.
    p1 = submit("u1", 1, 5, when=iso(0))
    p2 = submit("u2", 1, 5, when=iso(0))
    d = submit("d", 1, 5, when=iso(1), depends_on=[p1["id"], p2["id"]])

    r = batch(10)
    assert set(r["selection_order"]) == {p1["id"], p2["id"]}
    assert d["id"] not in r["selection_order"]

    scheduler.complete_job(p1["id"])
    r2 = batch(11)
    assert r2["selection_order"] == []
    assert [x["job_id"] for x in scheduler.get_job(d["id"])["blocked_by"]] == [
        p2["id"]
    ]

    scheduler.complete_job(p2["id"])
    r3 = batch(12)
    assert r3["selection_order"] == [d["id"]]


def test_completed_prerequisite_not_listed_as_blocker(conn):
    a = submit("a", 1, 5, when=iso(0))
    b = submit("b", 1, 5, when=iso(1), depends_on=[a["id"]])
    batch(10)
    scheduler.complete_job(a["id"])
    # Still QUEUED until the next batch, but now unblocked.
    bj = scheduler.get_job(b["id"])
    assert bj["status"] == "QUEUED"
    assert bj["blocked_by"] == []
    batch(11)
    running = scheduler.get_job(b["id"])
    assert running["status"] == "RUNNING"
    assert running["blocked_by"] == []
    assert running["depends_on"] == [a["id"]]
