"""
Background training worker.
Runs as a standalone subprocess: python training_worker.py --job_id N --db_path PATH
"""

import argparse
import json
import sqlite3
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Add parent to path so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _update_job(db_path: str, job_id: int, **kwargs):
    conn = _get_conn(db_path)
    try:
        sets = []
        vals = []
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = json.dumps(v)
            sets.append(f"{k} = ?")
            vals.append(v)
        vals.append(job_id)
        conn.execute(f"UPDATE training_jobs SET {', '.join(sets)} WHERE id = ?", vals)
        conn.commit()
    finally:
        conn.close()


def _get_job(db_path: str, job_id: int) -> dict:
    conn = _get_conn(db_path)
    try:
        row = conn.execute("SELECT * FROM training_jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise ValueError(f"Job {job_id} not found")
        d = dict(row)
        for key in ("config",):
            if key in d and isinstance(d[key], str):
                d[key] = json.loads(d[key])
        return d
    finally:
        conn.close()


def _get_model(db_path: str, model_id: int) -> dict | None:
    conn = _get_conn(db_path)
    try:
        row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _get_dataset(db_path: str, dataset_id: int) -> dict | None:
    conn = _get_conn(db_path)
    try:
        row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _create_model(db_path: str, name: str, version: str, file_path: str,
                  environment: str, training_job_id: int,
                  metrics: dict | None = None):
    conn = _get_conn(db_path)
    try:
        conn.execute(
            "INSERT INTO models (name, version, file_path, environment, training_job_id, "
            "metrics, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, version, file_path, environment, training_job_id,
             json.dumps(metrics or {}), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def run_training(job_id: int, db_path: str):
    now = datetime.now(timezone.utc).isoformat()
    _update_job(db_path, job_id, status="running", started_at=now)

    job = _get_job(db_path, job_id)
    config = job["config"]

    # Resolve base model path
    base_model_path = None
    if job["model_id"]:
        base_model = _get_model(db_path, job["model_id"])
        if base_model:
            base_model_path = base_model["file_path"]

    if not base_model_path:
        # Use the production model or pretrained weights
        conn = _get_conn(db_path)
        try:
            row = conn.execute(
                "SELECT file_path FROM models WHERE environment = 'production' LIMIT 1"
            ).fetchone()
            if row:
                base_model_path = row["file_path"]
        finally:
            conn.close()

    if not base_model_path:
        # Fall back to pretrained weights
        from app.config import PRETRAINED_WEIGHTS
        base_model_path = str(PRETRAINED_WEIGHTS)

    # Get dataset info
    dataset = _get_dataset(db_path, job["dataset_id"])
    if not dataset:
        raise ValueError(f"Dataset {job['dataset_id']} not found")

    image_dir = dataset["image_dir"]
    label_dir = dataset["label_dir"]

    # Create YOLO data.yaml
    from app.config import CLASS_NAMES, STORAGE_DIR

    output_dir = Path(STORAGE_DIR) / "models" / f"job_{job_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_yaml_path = output_dir / "data.yaml"
    import yaml
    data_config = {
        "path": str(Path(image_dir).parent),
        "train": "images",
        "val": "images",  # Use same for val in fine-tuning
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_config, f)

    # Run training
    from ultralytics import YOLO

    model = YOLO(base_model_path)

    num_epochs = config.get("num_epochs", 30)
    lr = config.get("learning_rate", 0.01)
    batch_size = config.get("batch_size", 8)
    freeze_epochs = config.get("freeze_epochs", 0)
    device = config.get("device", "auto")

    if device == "auto":
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"

    # Optional: freeze backbone for initial epochs
    train_kwargs = dict(
        data=str(data_yaml_path),
        epochs=num_epochs,
        batch=batch_size,
        lr0=lr,
        imgsz=640,
        project=str(output_dir),
        name="train",
        exist_ok=True,
        device=device,
        verbose=True,
    )

    if freeze_epochs > 0:
        # Freeze backbone layers; use freeze_epochs as the layer count
        # YOLOv8n has 10 backbone layers (indices 0-9)
        train_kwargs["freeze"] = min(freeze_epochs, 10)

    results = model.train(**train_kwargs)

    # Extract metrics
    metrics = {}
    if hasattr(results, "results_dict"):
        rd = results.results_dict
        metrics = {
            "mAP50": round(rd.get("metrics/mAP50(B)", 0), 4),
            "mAP50-95": round(rd.get("metrics/mAP50-95(B)", 0), 4),
            "precision": round(rd.get("metrics/precision(B)", 0), 4),
            "recall": round(rd.get("metrics/recall(B)", 0), 4),
        }

    # Find output model
    best_path = output_dir / "train" / "weights" / "best.pt"
    if not best_path.exists():
        # Try alternate location
        for p in output_dir.rglob("best.pt"):
            best_path = p
            break

    output_path = str(best_path)

    # Register model in DB
    _create_model(
        db_path=db_path,
        name=f"finetune_job_{job_id}",
        version=f"v_job_{job_id}",
        file_path=output_path,
        environment="staging",
        training_job_id=job_id,
        metrics=metrics,
    )

    # ── MLflow logging ──────────────────────────────────────────────
    mlflow_run_id = None
    try:
        import logging
        import mlflow
        import mlflow.pyfunc
        import pandas as pd
        from app.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"api_finetune_job_{job_id}") as run:
            mlflow_run_id = run.info.run_id

            # 1) Parameters
            mlflow.log_params({
                "epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "img_size": 640,
                "freeze_epochs": freeze_epochs,
                "device": device,
                "base_model": str(base_model_path),
                "source": "api",
                "job_id": str(job_id),
            })

            # 2) Final metrics
            if metrics:
                mlflow.log_metrics(metrics)

            # 3) Per-epoch metrics from results.csv
            yolo_train_dir = output_dir / "train"
            results_csv = yolo_train_dir / "results.csv"
            if results_csv.exists():
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
                for _, row_data in df.iterrows():
                    epoch = int(row_data["epoch"])
                    for yolo_col, mlflow_name in col_map.items():
                        if yolo_col in df.columns:
                            mlflow.log_metric(
                                mlflow_name, float(row_data[yolo_col]), step=epoch
                            )

            # 4) YOLO training artifacts (curves, confusion matrices)
            if yolo_train_dir.exists():
                allowed_ext = {".png", ".jpg", ".csv", ".yaml"}
                for f in sorted(yolo_train_dir.iterdir()):
                    if f.is_file() and f.suffix in allowed_ext:
                        mlflow.log_artifact(str(f), artifact_path="yolo_outputs")

            # 5) Model artifact (consistent with notebooks convention)
            if best_path.exists():
                mlflow.log_artifact(str(best_path), artifact_path="model")

            # 6) Register model in MLflow Model Registry
            class _YOLOWrapper(mlflow.pyfunc.PythonModel):
                def load_context(self, context):
                    from ultralytics import YOLO
                    self.model = YOLO(context.artifacts["weights"])

                def predict(self, context, model_input, params=None):
                    conf = params.get("conf", 0.25) if params else 0.25
                    srcs = (model_input.iloc[:, 0].tolist()
                            if isinstance(model_input, pd.DataFrame)
                            else list(model_input))
                    results_list = self.model.predict(
                        source=srcs, conf=conf, verbose=False
                    )
                    preds = []
                    for r in results_list:
                        boxes = r.boxes
                        if len(boxes) > 0:
                            preds.append({
                                "boxes": boxes.xyxy.cpu().numpy().tolist(),
                                "confidences": boxes.conf.cpu().numpy().tolist(),
                                "classes": boxes.cls.cpu().int().numpy().tolist(),
                            })
                        else:
                            preds.append({
                                "boxes": [], "confidences": [], "classes": []
                            })
                    return preds

            if best_path.exists():
                mlflow.pyfunc.log_model(
                    artifact_path="yolo_registered",
                    python_model=_YOLOWrapper(),
                    artifacts={"weights": str(best_path.resolve())},
                    registered_model_name="tools_detection_yolo",
                    pip_requirements=["ultralytics", "torch", "pandas"],
                )

        _update_job(db_path, job_id, mlflow_run_id=mlflow_run_id)

        # Also update the models table with the run_id
        conn = _get_conn(db_path)
        try:
            conn.execute(
                "UPDATE models SET mlflow_run_id = ? WHERE training_job_id = ?",
                (mlflow_run_id, job_id),
            )
            conn.commit()
        finally:
            conn.close()

    except Exception as e:
        import logging
        logging.warning(f"MLflow logging failed for job {job_id}: {e}")
        if mlflow_run_id:
            try:
                _update_job(db_path, job_id, mlflow_run_id=mlflow_run_id)
            except Exception:
                pass

    # Mark complete
    completed_at = datetime.now(timezone.utc).isoformat()
    _update_job(
        db_path, job_id,
        status="completed",
        completed_at=completed_at,
        output_model_path=output_path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--db_path", type=str, required=True)
    args = parser.parse_args()

    try:
        run_training(args.job_id, args.db_path)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        _update_job(
            args.db_path, args.job_id,
            status="failed",
            error_message=error_msg[:2000],
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
