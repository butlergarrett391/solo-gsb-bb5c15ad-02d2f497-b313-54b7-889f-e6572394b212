"""End-to-end HTTP tests for job dependencies."""


def test_dependency_lifecycle_over_http(client):
    # Prerequisite in one tenant.
    r = client.post("/jobs", json={
        "tenant_id": "up", "gpus": 1, "duration_min": 5})
    assert r.status_code == 201
    pre = r.json()["id"]

    # Dependent in a *different* tenant.
    r = client.post("/jobs", json={
        "tenant_id": "down", "gpus": 1, "duration_min": 5,
        "depends_on": [pre]})
    assert r.status_code == 201, r.text
    dep = r.json()["id"]
    assert dep != pre
    assert r.json()["depends_on"] == [pre]

    # First batch runs only the prerequisite.
    r = client.post("/scheduler/batch")
    assert r.json()["selection_order"] == [pre]

    # Dependent is blocked and reports why, on the job and the queue.
    r = client.get(f"/jobs/{dep}")
    body = r.json()
    assert body["status"] == "QUEUED"
    assert body["blocked_by"] == [{
        "job_id": pre, "status": "RUNNING",
        "reason": f"prerequisite {pre} is RUNNING, not COMPLETED",
    }]
    q = client.get("/queue").json()
    down_group = next(g for g in q["queues"] if g["tenant_id"] == "down")
    assert [x["job_id"] for x in down_group["head_blocked_by"]] == [pre]
    assert down_group["jobs"][0]["blocked_by"][0]["job_id"] == pre

    # Blocked batches select nothing.
    assert client.post("/scheduler/batch").json()["selection_order"] == []

    # Completing the prerequisite releases the dependent.
    assert client.post(f"/jobs/{pre}/complete").status_code == 200
    r = client.post("/scheduler/batch")
    assert r.json()["selection_order"] == [dep]
    assert client.get(f"/jobs/{dep}").json()["status"] == "RUNNING"


def test_unknown_dependency_rejected_http(client):
    r = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 5,
        "depends_on": ["J000404"]})
    assert r.status_code == 400
    assert "J000404" in r.json()["detail"]
    # No job was created.
    assert client.get("/jobs").json() == []


def test_malformed_dependency_body_http(client):
    # Non-list depends_on -> 422 from pydantic.
    r = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 5,
        "depends_on": "J000001"})
    assert r.status_code == 422
    # Empty-string id -> 400 from the service.
    r = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 5,
        "depends_on": [""]})
    assert r.status_code == 400


def test_cancelled_prerequisite_blocks_dependent_http(client):
    pre = client.post("/jobs", json={
        "tenant_id": "up", "gpus": 1, "duration_min": 5}).json()["id"]
    dep = client.post("/jobs", json={
        "tenant_id": "down", "gpus": 1, "duration_min": 5,
        "depends_on": [pre]}).json()["id"]

    assert client.post(f"/jobs/{pre}/cancel").status_code == 200
    r = client.post("/scheduler/batch")
    assert r.json()["selection_order"] == []

    body = client.get(f"/jobs/{dep}").json()
    blocker = body["blocked_by"][0]
    assert blocker["status"] == "CANCELLED"
    assert "CANCELLED" in blocker["reason"]

    # Never released, even on repeated batches.
    assert client.post("/scheduler/batch").json()["selection_order"] == []
    # The dependent itself can be cancelled to give up.
    r = client.post(f"/jobs/{dep}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "CANCELLED"


def test_diamond_releases_only_after_both_http(client):
    p1 = client.post("/jobs", json={
        "tenant_id": "u1", "gpus": 1, "duration_min": 5}).json()["id"]
    p2 = client.post("/jobs", json={
        "tenant_id": "u2", "gpus": 1, "duration_min": 5}).json()["id"]
    d = client.post("/jobs", json={
        "tenant_id": "d", "gpus": 1, "duration_min": 5,
        "depends_on": [p1, p2]}).json()["id"]

    r = client.post("/scheduler/batch")
    assert set(r.json()["selection_order"]) == {p1, p2}

    client.post(f"/jobs/{p1}/complete")
    assert client.post("/scheduler/batch").json()["selection_order"] == []
    assert [x["job_id"] for x in client.get(f"/jobs/{d}").json()["blocked_by"]] == [p2]

    client.post(f"/jobs/{p2}/complete")
    assert client.post("/scheduler/batch").json()["selection_order"] == [d]
