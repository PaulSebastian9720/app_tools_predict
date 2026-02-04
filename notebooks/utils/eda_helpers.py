"""
Reusable helpers for exploratory data analysis on YOLO-format datasets.

Supports both YOLO label formats:
- Detection:    class_id x_center y_center width height
- Segmentation: class_id x1 y1 x2 y2 ... xn yn

All coordinates are normalized [0, 1].
"""

from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np


def _parse_label_line(line: str) -> dict | None:
    """
    Parse a single YOLO label line (detection or segmentation).
    Returns dict with class_id, x_center, y_center, width, height,
    or None if the line is invalid.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    class_id = int(parts[0])

    if len(parts) == 5:
        # Detection format: class_id x_center y_center width height
        return {
            "class_id": class_id,
            "x_center": float(parts[1]),
            "y_center": float(parts[2]),
            "width": float(parts[3]),
            "height": float(parts[4]),
        }
    else:
        # Segmentation format: class_id x1 y1 x2 y2 ... xn yn
        coords = list(map(float, parts[1:]))
        xs = coords[0::2]
        ys = coords[1::2]
        if not xs or not ys:
            return None
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return {
            "class_id": class_id,
            "x_center": (x_min + x_max) / 2,
            "y_center": (y_min + y_max) / 2,
            "width": x_max - x_min,
            "height": y_max - y_min,
        }


def count_class_distribution(labels_dir: Path) -> Counter:
    """Count annotations per class in a YOLO labels directory."""
    counter = Counter()
    for label_file in sorted(labels_dir.glob("*.txt")):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counter[int(parts[0])] += 1
    return counter


def count_images(images_dir: Path, extensions=("*.jpg", "*.png", "*.jpeg")) -> int:
    """Count images in a directory across common extensions."""
    total = 0
    for ext in extensions:
        total += len(list(images_dir.glob(ext)))
    return total


def get_annotations_per_image(labels_dir: Path) -> dict[str, list[int]]:
    """Return {image_stem: [class_id, ...]} for every label file."""
    data = {}
    for label_file in sorted(labels_dir.glob("*.txt")):
        classes = []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.append(int(parts[0]))
        data[label_file.stem] = classes
    return data


def get_bbox_stats(labels_dir: Path) -> list[dict]:
    """
    Parse all label lines and extract bounding boxes.
    Returns a list of dicts with:
    class_id, x_center, y_center, width, height (all normalized).
    """
    boxes = []
    for label_file in sorted(labels_dir.glob("*.txt")):
        with open(label_file) as f:
            for line in f:
                parsed = _parse_label_line(line)
                if parsed is not None:
                    boxes.append({
                        "class_id": parsed["class_id"],
                        "x_center": parsed["x_center"],
                        "y_center": parsed["y_center"],
                        "width": parsed["width"],
                        "height": parsed["height"],
                        "image": label_file.stem,
                    })
    return boxes


def get_image_dimensions(images_dir: Path, sample_limit: int = 0) -> list[dict]:
    """
    Read image dimensions. If sample_limit > 0, only read that many images.
    Returns list of {stem, width, height}.
    """
    files = sorted(images_dir.glob("*.jpg"))
    if sample_limit > 0:
        files = files[:sample_limit]
    dims = []
    for img_path in files:
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            dims.append({"stem": img_path.stem, "width": w, "height": h})
    return dims


def draw_bboxes(image_path: Path, label_path: Path,
                class_names: list[str], colors: np.ndarray) -> np.ndarray:
    """
    Draw bounding boxes on an image from YOLO label files.
    Returns BGR image with boxes and class labels drawn.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    h, w = img.shape[:2]

    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parsed = _parse_label_line(line)
                if parsed is None:
                    continue

                cid = parsed["class_id"]
                x1 = int((parsed["x_center"] - parsed["width"] / 2) * w)
                y1 = int((parsed["y_center"] - parsed["height"] / 2) * h)
                x2 = int((parsed["x_center"] + parsed["width"] / 2) * w)
                y2 = int((parsed["y_center"] + parsed["height"] / 2) * h)

                color = tuple(int(c) for c in (colors[cid % len(colors)] * 255)[:3])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                label = class_names[cid] if cid < len(class_names) else str(cid)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def dataset_summary(images_dir: Path, labels_dir: Path) -> dict:
    """
    Return a summary dict for a YOLO dataset split:
    num_images, num_labels, total_annotations, classes_present,
    annotations_per_image stats.
    """
    n_images = count_images(images_dir)
    n_labels = len(list(labels_dir.glob("*.txt")))
    dist = count_class_distribution(labels_dir)
    annots = get_annotations_per_image(labels_dir)
    counts_per_img = [len(v) for v in annots.values()]

    return {
        "num_images": n_images,
        "num_labels": n_labels,
        "total_annotations": sum(dist.values()),
        "classes_present": len(dist),
        "annotations_per_image_mean": np.mean(counts_per_img) if counts_per_img else 0,
        "annotations_per_image_std": np.std(counts_per_img) if counts_per_img else 0,
        "annotations_per_image_min": min(counts_per_img) if counts_per_img else 0,
        "annotations_per_image_max": max(counts_per_img) if counts_per_img else 0,
    }
