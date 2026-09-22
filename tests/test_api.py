"""End-to-end HTTP tests against the FastAPI app."""

import pytest


def test_full_lifecycle_via_http(client):
    # Submit jobs for two tenants.
    r = client.post("/jobs", json={
        "tenant_id": "team-a", "gpus": 2, "duration_min": 30, "weight": 1})
    assert r.status_code == 201, r.text
    j1 = r.json()["id"]

    r = client.post("/jobs", json={
        "tenant_id": "team-b", "gpus": 6, "duration_min": 10, "weight": 2})
    assert r.status_code == 201, r.text
    j2 = r.json()["id"]

    # 2 + 6 = 8 fits; both selected in DRR tenant order.
    r = client.post("/scheduler/batch")
    assert r.status_code == 200
    body = r.json()
    assert body["selection_order"] == [j1, j2]
    assert body["used_gpus_after"] == 8
    assert body["free_gpus_after"] == 0
    tenants = {t["id"]: t for t in body["tenants"]}
    # a: ~quantum 60 - cost 60 (~0 aging, sub-second elapsed)
    assert tenants["team-a"]["balance_gpu_min"] == pytest.approx(0.0, abs=0.5)
    # b: ~quantum 120 - cost 60
    assert tenants["team-b"]["balance_gpu_min"] == pytest.approx(60.0, abs=0.5)
    kinds = [e["kind"] for e in body["events"]]
    assert "QUANTUM" in kinds and "DISPATCH" in kinds

    # Third job cannot fit while cluster full.
    r = client.post("/jobs", json={
        "tenant_id": "team-c", "gpus": 4, "duration_min": 10, "weight": 1})
    j3 = r.json()["id"]
    r = client.post("/scheduler/batch")
    assert r.json()["selection_order"] == []

    # Complete j1 (2 GPUs freed) -> still only 2 free, j3 needs 4 -> blocked.
    r = client.post(f"/jobs/{j1}/complete")
    assert r.status_code == 200
    r = client.post("/scheduler/batch")
    assert r.json()["selection_order"] == []

    # Complete j2 -> 8 free -> j3 runs.
    client.post(f"/jobs/{j2}/complete")
    r = client.post("/scheduler/batch")
    assert r.json()["selection_order"] == [j3]

    # Queries.
    q = client.get("/queue").json()
    assert q["queued_jobs"] == 0
    assert q["used_gpus"] == 4
    jobs = client.get("/jobs?status=RUNNING").json()
    assert [j["id"] for j in jobs] == [j3]
    tenants_list = client.get("/tenants").json()
    assert {t["id"] for t in tenants_list} == {"team-a", "team-b", "team-c"}
    ledger = client.get("/ledger?tenant_id=team-b&limit=10").json()
    assert all(e["tenant_id"] == "team-b" for e in ledger)
    assert client.get("/health").json() == {"status": "ok"}


def test_validation_errors(client):
    r = client.post("/jobs", json={
        "tenant_id": "x", "gpus": 0, "duration_min": 10})
    assert r.status_code == 422
    r = client.post("/jobs", json={
        "tenant_id": "x", "gpus": 100, "duration_min": 10})
    assert r.status_code == 400  # exceeds cluster capacity
    r = client.get("/jobs/nope/complete" if False else "/jobs/nope")
    assert r.status_code == 404


def test_cancel_endpoint(client):
    j = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 5}).json()["id"]
    r = client.post(f"/jobs/{j}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "CANCELLED"
    assert client.post(f"/jobs/{j}/cancel").status_code == 409
