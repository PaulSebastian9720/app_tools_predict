"""
MLflow tracking utilities for the tools detection project.

Connects to the MLflow tracking server (launch_mlflow_ui.sh) when available,
falls back to the local file-based store otherwise.
"""

import mlflow
import pandas as pd
from pathlib import Path

from utils.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_TRACKING_URI_FILE,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
)


# ── Connection ───────────────────────────────────────────────────────


def setup_mlflow():
    """Connect to MLflow and set the experiment.

    Tries the tracking server first; if unreachable falls back to the
    local ``file://`` store so notebooks still work offline.

    Returns the experiment object.
    """
    uri = MLFLOW_TRACKING_URI
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.search_experiments()  # quick connectivity check
    except Exception:
        uri = MLFLOW_TRACKING_URI_FILE
        mlflow.set_tracking_uri(uri)
        print(f"  Server not available, using file store")

    experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"MLflow tracking URI : {uri}")
    print(f"Experiment          : {MLFLOW_EXPERIMENT_NAME}")
    print(f"Experiment ID       : {experiment.experiment_id}")
    return experiment


# ── Run listing ──────────────────────────────────────────────────────


def list_runs():
    """Return a DataFrame with all runs in the current experiment.

    Columns include run_id, run_name, parameters, and metrics.
    """
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        print("Experiment not found. Call setup_mlflow() first.")
        return pd.DataFrame()

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )
    if runs.empty:
        print("No runs found.")
    return runs


# ── Model loading (by run) ───────────────────────────────────────────


def load_model_from_run(run_id: str) -> Path:
    """Download the YOLO ``.pt`` artifact from an MLflow run.

    Parameters
    ----------
    run_id : str
        The MLflow run ID.

    Returns
    -------
    Path
        Local path to the downloaded ``.pt`` file.
    """
    client = mlflow.tracking.MlflowClient()

    # Look in "model/" first (log_artifact path), then root
    for artifact_dir in ("model", ""):
        artifacts = client.list_artifacts(run_id, path=artifact_dir or None)
        for a in artifacts:
            if a.path.endswith(".pt"):
                local_path = client.download_artifacts(run_id, a.path)
                print(f"Model downloaded: {local_path}")
                return Path(local_path)

    raise FileNotFoundError(f"No .pt model artifact found in run {run_id}")


# ── Model loading (by registry) ─────────────────────────────────────


def list_registered_model_versions(model_name: str = MLFLOW_MODEL_NAME):
    """Return a DataFrame with all versions of a registered model.

    Columns: version, run_id, creation_timestamp, current_stage, description.
    """
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as exc:
        print(f"Could not query model registry: {exc}")
        return pd.DataFrame()

    if not versions:
        print(f"No versions found for model '{model_name}'.")
        return pd.DataFrame()

    rows = []
    for mv in versions:
        rows.append({
            "version": mv.version,
            "run_id": mv.run_id,
            "stage": getattr(mv, "current_stage", "None"),
            "status": mv.status,
            "created": pd.Timestamp(mv.creation_timestamp, unit="ms"),
        })
    df = pd.DataFrame(rows).sort_values("version", ascending=False)
    return df.reset_index(drop=True)


def load_model_from_registry(
    model_name: str = MLFLOW_MODEL_NAME,
    version: str | None = None,
) -> Path:
    """Download the YOLO ``.pt`` weights from a registered model version.

    Parameters
    ----------
    model_name : str
        Registered model name (default from config).
    version : str or None
        Specific version number.  When *None* the latest version is used.

    Returns
    -------
    Path
        Local path to the downloaded ``.pt`` file.
    """
    client = mlflow.tracking.MlflowClient()

    if version is None:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(f"No versions for model '{model_name}'")
        # Latest by version number
        mv = max(versions, key=lambda v: int(v.version))
        print(f"Using latest version: {mv.version}")
    else:
        mv = client.get_model_version(model_name, version)

    print(f"  Registry version {mv.version}  |  run_id={mv.run_id}")
    return load_model_from_run(mv.run_id)


# ── YOLO <-> MLflow integration ─────────────────────────────────────


class YOLOPyfuncWrapper(mlflow.pyfunc.PythonModel):
    """Wraps a YOLO model so MLflow can serve it via the pyfunc interface."""

    def load_context(self, context):
        from ultralytics import YOLO
        self.model = YOLO(context.artifacts["weights"])

    def predict(self, context, model_input, params=None):
        conf = params.get("conf", 0.25) if params else 0.25
        if isinstance(model_input, pd.DataFrame):
            sources = model_input.iloc[:, 0].tolist()
        else:
            sources = list(model_input)

        results = self.model.predict(source=sources, conf=conf, verbose=False)
        predictions = []
        for r in results:
            boxes = r.boxes
            if len(boxes) > 0:
                predictions.append({
                    "boxes": boxes.xyxy.cpu().numpy().tolist(),
                    "confidences": boxes.conf.cpu().numpy().tolist(),
                    "classes": boxes.cls.cpu().int().numpy().tolist(),
                })
            else:
                predictions.append({"boxes": [], "confidences": [], "classes": []})
        return predictions


def log_yolo_run_artifacts(yolo_run_dir):
    """Log YOLO training output files (curves, images, CSV) to the active run.

    Skips weight files — those are handled separately.
    """
    yolo_run_dir = Path(yolo_run_dir)
    if not yolo_run_dir.exists():
        print(f"  YOLO run dir not found: {yolo_run_dir}")
        return

    allowed_ext = {".png", ".jpg", ".csv", ".yaml"}
    count = 0
    for f in sorted(yolo_run_dir.iterdir()):
        if f.is_file() and f.suffix in allowed_ext:
            mlflow.log_artifact(str(f), artifact_path="yolo_outputs")
            count += 1
    print(f"  Logged {count} YOLO artifacts from {yolo_run_dir.name}/")


def log_yolo_epoch_metrics(yolo_run_dir):
    """Log per-epoch training metrics from YOLO results.csv to the active run."""
    results_csv = Path(yolo_run_dir) / "results.csv"
    if not results_csv.exists():
        print(f"  results.csv not found in {yolo_run_dir}")
        return

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    col_map = {
        "train/box_loss": "train_box_loss",
        "train/cls_loss": "train_cls_loss",
        "train/dfl_loss": "train_dfl_loss",
        "metrics/precision(B)": "epoch_precision",
        "metrics/recall(B)": "epoch_recall",
        "metrics/mAP50(B)": "epoch_mAP50",
        "metrics/mAP50-95(B)": "epoch_mAP50_95",
        "val/box_loss": "val_box_loss",
        "val/cls_loss": "val_cls_loss",
        "val/dfl_loss": "val_dfl_loss",
    }

    for _, row in df.iterrows():
        epoch = int(row["epoch"])
        for yolo_col, mlflow_name in col_map.items():
            if yolo_col in df.columns:
                mlflow.log_metric(mlflow_name, float(row[yolo_col]), step=epoch)

    print(f"  Logged {len(df)} epochs of training metrics")


def register_yolo_model(best_pt, model_name=MLFLOW_MODEL_NAME):
    """Register a YOLO .pt model in the MLflow Model Registry via pyfunc.

    Parameters
    ----------
    best_pt : str or Path
        Path to the best.pt weights file.
    model_name : str
        Name in the Model Registry.

    Returns
    -------
    mlflow.models.model.ModelInfo
    """
    model_info = mlflow.pyfunc.log_model(
        artifact_path="yolo_registered",
        python_model=YOLOPyfuncWrapper(),
        artifacts={"weights": str(Path(best_pt).resolve())},
        registered_model_name=model_name,
        pip_requirements=["ultralytics", "torch", "pandas"],
    )
    print(f"  Model registered in: {model_name}")
    return model_info
