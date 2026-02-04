"""
GET  /api/v1/datasets — list datasets
POST /api/v1/datasets — upload dataset (images + labels)
"""

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .. import database as db
from ..config import STORAGE_DIR
from ..schemas import DatasetListResponse, DatasetResponse

router = APIRouter()


@router.get("/datasets", response_model=DatasetListResponse)
def list_datasets(status: str = "active"):
    datasets = db.list_datasets(status)
    return DatasetListResponse(
        datasets=[DatasetResponse(**d) for d in datasets],
        total=len(datasets),
    )


@router.post("/datasets", response_model=DatasetResponse)
async def create_dataset(
    name: str = Form(...),
    description: str = Form(""),
    images: list[UploadFile] = File(...),
    labels: list[UploadFile] | None = File(None),
    annotation_file: UploadFile | None = File(None),
):
    """
    Upload a dataset. Supports two modes:
    1. Images + YOLO label files (labels param)
    2. Images + COCO JSON annotation file (annotation_file param)
    """
    if not images:
        raise HTTPException(status_code=400, detail="No images provided")

    # Create a temporary ID-based directory
    # First insert to get the ID, then update paths
    record = db.create_dataset(
        name=name, description=description,
        image_dir="", label_dir="",
        num_images=0, num_annotations=0,
    )
    ds_id = record["id"]

    ds_dir = Path(STORAGE_DIR / "datasets" / str(ds_id))
    img_dir = ds_dir / "images"
    lbl_dir = ds_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Save images
    num_images = 0
    for img_file in images:
        if not img_file.filename:
            continue
        content = await img_file.read()
        if not content:
            continue
        dest = img_dir / img_file.filename
        dest.write_bytes(content)
        num_images += 1

    # Save labels
    num_annotations = 0
    if labels:
        for lbl_file in labels:
            if not lbl_file.filename:
                continue
            content = await lbl_file.read()
            dest = lbl_dir / lbl_file.filename
            dest.write_bytes(content)
            # Count annotation lines (each line = one bbox)
            num_annotations += len(content.decode("utf-8", errors="ignore").strip().splitlines())
    elif annotation_file:
        # Save raw annotation file for reference
        content = await annotation_file.read()
        ann_path = ds_dir / (annotation_file.filename or "annotations.json")
        ann_path.write_bytes(content)
        # For COCO JSON, count annotations later
        import json as _json
        try:
            coco = _json.loads(content)
            num_annotations = len(coco.get("annotations", []))
        except Exception:
            pass

    # Update record with final paths and counts
    updated = db.update_dataset(
        ds_id,
        image_dir=str(img_dir),
        label_dir=str(lbl_dir),
        num_images=num_images,
        num_annotations=num_annotations,
    )
    return DatasetResponse(**updated)
