"""
GET  /api/v1/jobs          — list training jobs
POST /api/v1/train         — create a training job
POST /api/v1/jobs/{id}/cancel — cancel a running job
"""

import os
import signal
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

from .. import database as db
from ..config import DB_PATH
from ..schemas import JobListResponse, JobResponse, TrainRequest

router = APIRouter()

WORKER_SCRIPT = Path(__file__).resolve().parent.parent / "training_worker.py"


@router.get("/jobs", response_model=JobListResponse)
def list_jobs(status: str | None = None):
    jobs = db.list_jobs(status)
    return JobListResponse(
        jobs=[JobResponse(**j) for j in jobs],
        total=len(jobs),
    )


@router.post("/train", response_model=JobResponse)
def start_training(req: TrainRequest):
    # Validate dataset exists
    datasets = db.list_datasets("active")
    ds_ids = {d["id"] for d in datasets}
    if req.dataset_id not in ds_ids:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Validate base model exists if specified
    if req.base_model_id is not None:
        model = db.get_model(req.base_model_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Base model not found")

    # Create job record
    job = db.create_job(
        job_type=req.job_type,
        dataset_id=req.dataset_id,
        model_id=req.base_model_id,
        config=req.config.model_dump(),
    )

    # Spawn background subprocess
    proc = subprocess.Popen(
        [sys.executable, str(WORKER_SCRIPT),
         "--job_id", str(job["id"]),
         "--db_path", str(DB_PATH)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Store PID for cancellation
    db.update_job(job["id"], pid=proc.pid)
    job["pid"] = proc.pid

    return JobResponse(**job)


@router.post("/jobs/{job_id}/cancel", response_model=JobResponse)
def cancel_job(job_id: int):
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("pending", "running"):
        raise HTTPException(status_code=400, detail="Job is not running")

    # Try to kill the subprocess
    pid = job.get("pid")
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    updated = db.update_job(job_id, status="cancelled")
    return JobResponse(**updated)
