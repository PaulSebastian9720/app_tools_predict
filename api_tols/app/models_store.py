"""
Manages YOLO model loading and inference.
Keeps the production model loaded in memory for fast predictions.
"""

import base64
import io
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from . import database as db
from .config import CLASS_COLORS, ID_TO_NAME


class ModelStore:
    def __init__(self):
        self._models: dict[int, YOLO] = {}
        self._production_id: int | None = None

    def load_model(self, model_id: int) -> YOLO:
        """Load a model by its database ID, caching in memory."""
        if model_id in self._models:
            return self._models[model_id]
        record = db.get_model(model_id)
        if record is None:
            raise ValueError(f"Model {model_id} not found in database")
        path = record["file_path"]
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        model = YOLO(path)
        self._models[model_id] = model
        return model

    def load_production(self) -> YOLO | None:
        """Load the current production model."""
        record = db.get_production_model()
        if record is None:
            return None
        self._production_id = record["id"]
        return self.load_model(record["id"])

    def get_production_model(self) -> tuple[YOLO | None, dict | None]:
        """Return (model, db_record) for production model."""
        record = db.get_production_model()
        if record is None:
            return None, None
        model = self.load_model(record["id"])
        return model, record

    def get_device(self) -> str:
        """Return the device string of the production model."""
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def predict(self, image_bytes: bytes, score_threshold: float = 0.5,
                model_id: int | None = None) -> dict:
        """
        Run inference on a single image.
        Returns dict with detections, inference_time_ms, model_version,
        image_size, and annotated_image (base64).
        """
        if model_id is not None:
            model = self.load_model(model_id)
            record = db.get_model(model_id)
        else:
            model, record = self.get_production_model()

        if model is None or record is None:
            raise RuntimeError("No model available for prediction")

        # Decode image
        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        h, w = img.shape[:2]

        # Run inference
        start = time.perf_counter()
        results = model.predict(img, conf=score_threshold, verbose=False)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Parse detections
        detections = []
        result = results[0]
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                detections.append({
                    "class_id": cls_id,
                    "class_name": ID_TO_NAME.get(cls_id, f"class_{cls_id}"),
                    "score": round(float(box.conf[0]), 4),
                    "bbox": [round(float(x), 1) for x in box.xyxy[0].tolist()],
                    "mask_rle": None,
                })

        # Draw annotated image
        annotated = self._draw_annotations(img, detections)
        _, buf = cv2.imencode(".png", annotated)
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        return {
            "detections": detections,
            "inference_time_ms": round(elapsed_ms, 1),
            "model_version": record.get("version", "unknown"),
            "image_size": [w, h],
            "annotated_image": b64,
        }

    def _draw_annotations(self, img: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw bounding boxes and labels on the image."""
        annotated = img.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cls_name = det["class_name"]
            score = det["score"]

            # Get color
            hex_color = CLASS_COLORS.get(cls_name, "#00ff00")
            r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
            color = (b, g, r)  # BGR for OpenCV

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{cls_name} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated


# Singleton instance
store = ModelStore()
