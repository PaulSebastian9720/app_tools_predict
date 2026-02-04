"""GET /api/v1/health — system health check."""

from fastapi import APIRouter

from ..models_store import store
from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    model, record = store.get_production_model()
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_version=record["version"] if record else None,
        device=store.get_device(),
    )
