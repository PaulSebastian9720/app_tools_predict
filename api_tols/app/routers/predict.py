"""
POST /api/v1/predict/upload — single image prediction
POST /api/v1/predict/batch  — batch image prediction
"""

import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..models_store import store
from ..schemas import (
    BatchItemResult,
    BatchPredictionResult,
    PredictionResult,
)

router = APIRouter()


@router.post("/predict/upload", response_model=PredictionResult)
async def predict_upload(
    file: UploadFile = File(...),
    score_threshold: float = Form(0.5),
    model_id: int | None = Form(None),
):
    """Run prediction on a single uploaded image."""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        result = store.predict(contents, score_threshold, model_id)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PredictionResult(**result)


@router.post("/predict/batch", response_model=BatchPredictionResult)
async def predict_batch(
    files: list[UploadFile] = File(...),
    score_threshold: float = Form(0.5),
    model_id: int | None = Form(None),
):
    """Run prediction on multiple uploaded images."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []
    total_start = time.perf_counter()
    model_version = "unknown"

    for f in files:
        contents = await f.read()
        if not contents:
            continue
        try:
            pred = store.predict(contents, score_threshold, model_id)
            model_version = pred["model_version"]
            results.append(BatchItemResult(
                filename=f.filename or "unknown",
                detections=pred["detections"],
                inference_time_ms=pred["inference_time_ms"],
                annotated_image=pred["annotated_image"],
            ))
        except (RuntimeError, ValueError):
            results.append(BatchItemResult(
                filename=f.filename or "unknown",
                detections=[],
                inference_time_ms=0,
                annotated_image="",
            ))

    total_ms = (time.perf_counter() - total_start) * 1000

    return BatchPredictionResult(
        results=results,
        total_time_ms=round(total_ms, 1),
        model_version=model_version,
    )
