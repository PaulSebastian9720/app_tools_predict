"""
GET  /api/v1/models           — list models
POST /api/v1/models/{id}/promote — promote model to production
"""

from fastapi import APIRouter, HTTPException

from .. import database as db
from ..schemas import ModelListResponse, ModelResponse

router = APIRouter()


@router.get("/models", response_model=ModelListResponse)
def list_models(environment: str | None = None):
    models = db.list_models(environment)
    return ModelListResponse(
        models=[ModelResponse(**m) for m in models],
        total=len(models),
    )


@router.post("/models/{model_id}/promote", response_model=ModelResponse)
def promote_model(model_id: int):
    record = db.get_model(model_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Model not found")
    updated = db.promote_model(model_id)
    return ModelResponse(**updated)
