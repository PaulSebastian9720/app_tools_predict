"""
Utilities for converting bounding box annotations to YOLO format.
"""

from pathlib import Path


def bbox_to_yolo(bbox: list[float], img_width: int, img_height: int) -> list[float]:
    """
    Convert [x1, y1, x2, y2] pixel coordinates to YOLO format
    [x_center, y_center, width, height] normalized to [0, 1].
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / img_width
    cy = (y1 + y2) / 2.0 / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    return [
        round(max(0.0, min(1.0, cx)), 6),
        round(max(0.0, min(1.0, cy)), 6),
        round(max(0.0, min(1.0, w)), 6),
        round(max(0.0, min(1.0, h)), 6),
    ]


def save_yolo_labels(
    annotations: list[dict],
    output_dir: str | Path,
) -> int:
    """
    Write YOLO-format label files from a list of annotation dicts.

    Each annotation dict should have:
      - image_filename: str (e.g. "img001.jpg")
      - class_id: int
      - bbox: [x1, y1, x2, y2] in pixels
      - img_width: int
      - img_height: int

    Returns total number of annotations written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by image filename
    by_image: dict[str, list[dict]] = {}
    for ann in annotations:
        fname = ann["image_filename"]
        by_image.setdefault(fname, []).append(ann)

    total = 0
    for image_fname, anns in by_image.items():
        label_fname = Path(image_fname).stem + ".txt"
        label_path = output_dir / label_fname

        lines = []
        for ann in anns:
            yolo_box = bbox_to_yolo(
                ann["bbox"], ann["img_width"], ann["img_height"]
            )
            line = f"{ann['class_id']} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}"
            lines.append(line)
            total += 1

        label_path.write_text("\n".join(lines) + "\n")

    return total
