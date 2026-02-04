"""GET /api/v1/classes — list all tool classes with colors."""

from fastapi import APIRouter

from ..config import CLASS_COLORS, CLASS_NAMES
from ..schemas import ClassInfo, ClassListResponse

router = APIRouter()


@router.get("/classes", response_model=ClassListResponse)
def get_classes():
    classes = [
        ClassInfo(id=i, name=name, color=CLASS_COLORS[name])
        for i, name in enumerate(CLASS_NAMES)
    ]
    return ClassListResponse(classes=classes)
