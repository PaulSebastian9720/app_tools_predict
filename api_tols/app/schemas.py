"""Pydantic request/response models."""

from pydantic import BaseModel, Field


# ── Health ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None
    device: str


# ── Classes ────────────────────────────────────────────────────────────

class ClassInfo(BaseModel):
    id: int
    name: str
    color: str


class ClassListResponse(BaseModel):
    classes: list[ClassInfo]


# ── Prediction ─────────────────────────────────────────────────────────

class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    score: float
    bbox: list[float]
    mask_rle: str | None = None


class PredictionResult(BaseModel):
    detections: list[DetectionItem]
    inference_time_ms: float
    model_version: str
    image_size: list[int]
    annotated_image: str  # base64


class BatchItemResult(BaseModel):
    filename: str
    detections: list[DetectionItem]
    inference_time_ms: float
    annotated_image: str


class BatchPredictionResult(BaseModel):
    results: list[BatchItemResult]
    total_time_ms: float
    model_version: str


# ── Datasets ───────────────────────────────────────────────────────────

class DatasetResponse(BaseModel):
    id: int
    name: str
    description: str
    image_dir: str | None = None
    label_dir: str | None = None
    num_images: int
    num_annotations: int
    status: str
    created_at: str


class DatasetListResponse(BaseModel):
    datasets: list[DatasetResponse]
    total: int


# ── Training Jobs ──────────────────────────────────────────────────────

class TrainConfig(BaseModel):
    num_epochs: int = Field(default=30, ge=1)
    learning_rate: float = Field(default=0.01, gt=0)
    batch_size: int = Field(default=8, ge=1)
    freeze_epochs: int = Field(default=0, ge=0)
    device: str = "auto"


class TrainRequest(BaseModel):
    job_type: str = "fine_tune"
    dataset_id: int
    base_model_id: int | None = None
    config: TrainConfig = TrainConfig()


class JobResponse(BaseModel):
    id: int
    job_type: str
    dataset_id: int | None
    model_id: int | None = None
    status: str
    pid: int | None
    config: dict
    mlflow_run_id: str | None
    output_model_path: str | None
    error_message: str | None
    created_at: str
    started_at: str | None
    completed_at: str | None


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
    total: int


# ── Models ─────────────────────────────────────────────────────────────

class ModelResponse(BaseModel):
    id: int
    name: str
    version: str
    file_path: str
    environment: str
    training_job_id: int | None
    mlflow_run_id: str | None
    metrics: dict
    created_at: str
    promoted_at: str | None


class ModelListResponse(BaseModel):
    models: list[ModelResponse]
    total: int
