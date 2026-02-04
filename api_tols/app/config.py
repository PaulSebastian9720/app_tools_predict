"""
Shared configuration for the tools segmentation API.
Paths, class names, colors, and constants.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
API_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = API_ROOT.parent / "notebooks"
STORAGE_DIR = API_ROOT / "storage"
DB_PATH = STORAGE_DIR / "db.sqlite3"

RUNS_DIR = PROJECT_ROOT / "runs"
PRETRAINED_WEIGHTS = PROJECT_ROOT / "yolov8n.pt"

# ── MLflow ─────────────────────────────────────────────────────────────
MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_TRACKING_URI = f"file://{MLFLOW_TRACKING_DIR}"
MLFLOW_EXPERIMENT_NAME = "tools_detection"

# ── Class mapping ──────────────────────────────────────────────────────
CLASS_NAMES = [
    "Hammer", "Pliers", "Screwdriver", "Wrench",
]

NUM_CLASSES = len(CLASS_NAMES)
ID_TO_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}
NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


CLASS_COLORS = {
    "Hammer": "#e6194b",
    "Pliers": "#3cb44b",
    "Screwdriver": "#ffe119",
    "Wrench": "#4363d8",
}

# ── Training defaults ──────────────────────────────────────────────────
SEED = 42
DEFAULT_EPOCHS = 310
DEFAULT_BATCH_SIZE = 8
DEFAULT_IMG_SIZE = 640
DEFAULT_LR = 0.01

# ── Known pre-trained models to auto-register ─────────────────────────
KNOWN_MODELS = [
    {
        "name": "bootstrap_training",
        "version": "v1.0",
        "file_path": str(RUNS_DIR / "bootstrap_training" / "weights" / "best.pt"),
        "environment": "archived",
    },
    {
        "name": "dataset_1_finetune",
        "version": "v1.1",
        "file_path": str(RUNS_DIR / "dataset_1_finetune" / "weights" / "best.pt"),
        "environment": "archived",
    },
    {
        "name": "dataset_2_finetune",
        "version": "v1.2",
        "file_path": str(RUNS_DIR / "dataset_2_finetune" / "weights" / "best.pt"),
        "environment": "production",
    },
]
