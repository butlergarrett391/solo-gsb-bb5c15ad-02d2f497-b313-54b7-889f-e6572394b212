from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from . import db, scheduler
from .config import get_settings
from .scheduler import SchedulerError


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    db.connect(settings.db_path)
    yield
    conn = db.get_conn()
    conn.close()


app = FastAPI(
    title="Multi-tenant GPU Fair-Queuing Scheduler",
    version="1.0.0",
    description=(
        "Weighted Deficit Round-Robin GPU job scheduler with wait-time aging. "
        "Cost = gpus * duration_min (GPU-minutes); capacity is never exceeded."
    ),
    lifespan=lifespan,
)


@app.exception_handler(SchedulerError)
async def scheduler_error_handler(request, exc: SchedulerError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc)},
    )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TenantIn(BaseModel):
    tenant_id: str = Field(..., min_length=1, examples=["team-a"])
    weight: float = Field(..., gt=0, examples=[2.0])


class JobIn(BaseModel):
    tenant_id: str = Field(..., min_length=1, examples=["team-a"])
    gpus: int = Field(..., gt=0, examples=[2])
    duration_min: int = Field(..., gt=0, examples=[60])
    weight: float = Field(
        default=1.0, gt=0,
        description="Tenant weight; auto-registers the tenant or updates it.",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description=(
            "Job ids that must reach COMPLETED before this job is schedulable. "
            "Dependencies are cluster-global (cross-tenant allowed); unknown "
            "ids and dependency cycles are rejected."
        ),
        examples=[["J000001"]],
    )

    @field_validator("depends_on")
    @classmethod
    def _non_empty_dependency_ids(cls, value: list[str]) -> list[str]:
        if any(not isinstance(d, str) or not d.strip() for d in value):
            raise ValueError("depends_on entries must be non-empty strings")
        return value


# ---------------------------------------------------------------------------
# Tenant / state
# ---------------------------------------------------------------------------

@app.post("/tenants", tags=["tenants"], status_code=201)
def create_or_update_tenant(body: TenantIn):
    return scheduler.upsert_tenant(body.tenant_id, body.weight)


@app.get("/tenants", tags=["tenants"])
def get_tenants():
    """All tenants with weights and current GPU-minute balances."""
    return scheduler.list_tenants()


@app.get("/tenants/{tenant_id}", tags=["tenants"])
def get_tenant(tenant_id: str):
    try:
        return scheduler.get_tenant(tenant_id)
    except SchedulerError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@app.get("/scheduler/state", tags=["scheduler"])
def scheduler_state():
    """Cluster capacity, free/used GPUs, tuning knobs and tenant balances."""
    return scheduler.scheduler_state()


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

@app.post("/jobs", tags=["jobs"], status_code=201)
def submit_job(body: JobIn):
    return scheduler.submit_job(
        tenant_id=body.tenant_id,
        gpus=body.gpus,
        duration_min=body.duration_min,
        weight=body.weight,
        depends_on=body.depends_on,
    )


@app.get("/jobs", tags=["jobs"])
def list_jobs(
    status: str | None = Query(
        None, description="QUEUED|RUNNING|COMPLETED|CANCELLED"),
    tenant_id: str | None = None,
):
    try:
        return scheduler.list_jobs(status=status, tenant_id=tenant_id)
    except SchedulerError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@app.get("/jobs/{job_id}", tags=["jobs"])
def get_job(job_id: str):
    try:
        return scheduler.get_job(job_id)
    except SchedulerError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@app.post("/jobs/{job_id}/cancel", tags=["jobs"])
def cancel_job(job_id: str):
    """Cancel a QUEUED (or RUNNING) job. Queued cancellation frees no GPU;
    RUNNING cancellation releases GPUs for subsequent batches."""
    try:
        return scheduler.cancel_job(job_id)
    except SchedulerError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@app.post("/jobs/{job_id}/complete", tags=["jobs"])
def complete_job(job_id: str):
    """Mark a RUNNING job COMPLETED, releasing its GPUs."""
    try:
        return scheduler.complete_job(job_id)
    except SchedulerError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


# ---------------------------------------------------------------------------
# Queue & scheduling
# ---------------------------------------------------------------------------

@app.get("/queue", tags=["scheduler"])
def get_queue():
    """FIFO per-tenant queues, positions, weights and balances."""
    return scheduler.queue_overview()


@app.post("/scheduler/batch", tags=["scheduler"])
def schedule_next_batch():
    """Run one WDRR+aging batch; returns selection order and balances."""
    return scheduler.run_batch()


@app.get("/ledger", tags=["scheduler"])
def get_ledger(
    tenant_id: str | None = None,
    batch_id: int | None = None,
    limit: int = Query(default=200, ge=1, le=10_000),
):
    """Credit/debit audit trail (AGING / QUANTUM / DISPATCH)."""
    return scheduler.list_ledger(
        tenant_id=tenant_id, batch_id=batch_id, limit=limit
    )


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok"}
