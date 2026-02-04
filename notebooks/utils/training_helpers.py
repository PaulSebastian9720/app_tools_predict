"""
Training and evaluation utilities for YOLO object detection.
"""
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from ultralytics import YOLO

from utils.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    PREPROCESSED_DIR,
    TEST_DIR,
    DEFAULT_IMG_SIZE,
    ID_TO_NAME,
)


def get_device():
    """Detect and return the best available device for training."""
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Device : {'GPU' if device == '0' else 'CPU'}")
    if device == "0":
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM : {vram:.1f} GB")
    return device


def create_dataset_yaml(train_subdir, yaml_name=None):
    """
    Create a YOLO data YAML pointing to a specific training subset.

    Parameters
    ----------
    train_subdir : str
        Folder name under pre_process/train/ (e.g. 'bootstrap', 'dataset_1').
    yaml_name : str, optional
        Output filename. Defaults to '{train_subdir}_data.yaml'.

    Returns
    -------
    Path
        Absolute path to the created YAML file.
    """
    if yaml_name is None:
        yaml_name = f"{train_subdir}_data.yaml"

    yaml_path = PREPROCESSED_DIR / yaml_name
    config = {
        "path": str(PREPROCESSED_DIR),
        "train": f"train/{train_subdir}/images",
        "val": "test/images",
        "nc": NUM_CLASSES,
        "names": CLASS_NAMES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"YAML created: {yaml_path}")
    return yaml_path


def extract_metrics(val_results):
    """
    Extract detection metrics from YOLO validation results.

    Returns dict with mAP50, mAP50-95, precision, recall, f1.
    """
    metrics = {
        "mAP50": round(float(val_results.box.map50), 4),
        "mAP50-95": round(float(val_results.box.map), 4),
        "precision": round(float(val_results.box.mp), 4),
        "recall": round(float(val_results.box.mr), 4),
    }
    p, r = metrics["precision"], metrics["recall"]
    metrics["f1"] = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0
    return metrics


def evaluate_model(model, data_yaml=None, img_size=DEFAULT_IMG_SIZE, device=None):
    """
    Evaluate a YOLO model on the fixed test set.

    Parameters
    ----------
    model : YOLO
        Loaded YOLO model.
    data_yaml : str or Path, optional
        Data YAML path. Defaults to eval_data.yaml.
    img_size : int
        Image size for inference.
    device : str, optional
        Device for inference ('0' for GPU, 'cpu'). Auto-detected if None.

    Returns
    -------
    dict
        Detection metrics.
    """
    if data_yaml is None:
        data_yaml = PREPROCESSED_DIR / "eval_data.yaml"
        config = {
            "path": str(PREPROCESSED_DIR),
            "train": "test/images",
            "val": "test/images",
            "nc": NUM_CLASSES,
            "names": CLASS_NAMES,
        }
        with open(data_yaml, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    val_results = model.val(data=str(data_yaml), imgsz=img_size, device=device, verbose=False)
    return extract_metrics(val_results)


def visualize_predictions(model, images_dir, num_images=6,
                          img_size=DEFAULT_IMG_SIZE, conf=0.25):
    """Run inference on images and display annotated results."""
    images_dir = Path(images_dir)
    image_files = sorted(images_dir.glob("*.jpg"))[:num_images]
    if not image_files:
        print(f"No .jpg images in {images_dir}")
        return

    cols = min(3, len(image_files))
    rows = (len(image_files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for idx, img_path in enumerate(image_files):
        results = model.predict(str(img_path), imgsz=img_size, conf=conf, verbose=False)
        annotated = results[0].plot()
        axes[idx].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(img_path.name, fontsize=9)
        axes[idx].axis("off")

    for idx in range(len(image_files), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


def predict_single(model, image_path, img_size=DEFAULT_IMG_SIZE, conf=0.25):
    """
    Predict on a single image. Displays the annotated image and returns detections.

    Returns
    -------
    list[dict]
        Each dict has keys: class, confidence, bbox.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return []

    results = model.predict(str(image_path), imgsz=img_size, conf=conf, verbose=False)
    result = results[0]

    annotated = result.plot()
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title(image_path.name)
    plt.axis("off")
    plt.show()

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            detections.append({
                "class": ID_TO_NAME.get(cls_id, str(cls_id)),
                "confidence": round(float(box.conf.item()), 4),
                "bbox": box.xyxy[0].tolist(),
            })
    return detections


def predict_directory(model, images_dir, img_size=DEFAULT_IMG_SIZE, conf=0.25):
    """
    Run inference on all .jpg images in a directory.

    Returns
    -------
    dict
        {filename: [detection_dicts]}
    """
    images_dir = Path(images_dir)
    image_files = sorted(images_dir.glob("*.jpg"))
    if not image_files:
        print(f"No .jpg images in {images_dir}")
        return {}

    all_detections = {}
    for img_path in image_files:
        results = model.predict(str(img_path), imgsz=img_size, conf=conf, verbose=False)
        result = results[0]
        dets = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                dets.append({
                    "class": ID_TO_NAME.get(cls_id, str(cls_id)),
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox": box.xyxy[0].tolist(),
                })
        all_detections[img_path.name] = dets

    total = sum(len(v) for v in all_detections.values())
    print(f"Processed {len(image_files)} images — {total} detections")
    return all_detections
