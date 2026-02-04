"""
Shared configuration for the tools detection project.
Paths, class names, and constants used across notebooks.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
ORIGINAL_DATASET = DATA_DIR / "Tools.v1i.yolov8"
PREPROCESSED_DIR = DATA_DIR / "pre_process_2"

BOOTSTRAP_DIR = PREPROCESSED_DIR / "train" / "bootstrap"
DATASET_1_DIR = PREPROCESSED_DIR / "train" / "dataset_1"
DATASET_2_DIR = PREPROCESSED_DIR / "train" / "dataset_2"
DATASET_4_DIR = PREPROCESSED_DIR / "train" / "dataset_4"
TEST_DIR = PREPROCESSED_DIR / "test"

ORIGINAL_TRAIN_IMAGES = ORIGINAL_DATASET / "train" / "images"
ORIGINAL_TRAIN_LABELS = ORIGINAL_DATASET / "train" / "labels"
ORIGINAL_TEST_IMAGES = ORIGINAL_DATASET / "test" / "images"
ORIGINAL_TEST_LABELS = ORIGINAL_DATASET / "test" / "labels"

RUNS_DIR = PROJECT_ROOT / "runs"
BEST_MODEL_DIR = PROJECT_ROOT / "best_model"
PRETRAINED_WEIGHTS = PROJECT_ROOT / "yolov8n.pt"

# ── MLflow ─────────────────────────────────────────────────────────────
MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_TRACKING_URI_FILE = f"file://{MLFLOW_TRACKING_DIR}"

# Server launched by launch_mlflow_ui.sh
MLFLOW_SERVER_HOST = "127.0.0.1"
MLFLOW_SERVER_PORT = 5000
MLFLOW_TRACKING_URI = f"http://{MLFLOW_SERVER_HOST}:{MLFLOW_SERVER_PORT}"

MLFLOW_EXPERIMENT_NAME = "tools_detection"
MLFLOW_MODEL_NAME = "tools_detection_yolo"

# ── Class mapping ──────────────────────────────────────────────────────
CLASS_NAMES = [
    "Hammer", "Pliers", "Screwdriver", "Wrench",
]

NUM_CLASSES = len(CLASS_NAMES)
ID_TO_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}
NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# ── Training defaults ──────────────────────────────────────────────────
SEED = 42
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 8
DEFAULT_IMG_SIZE = 640
DEFAULT_LR = 0.01
