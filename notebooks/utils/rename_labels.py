"""
Remap class IDs and names in a YOLOv8 segmentation dataset so they match
the canonical labels:

    0: Hammer
    1: Pliers
    2: Screwdriver
    3: Wrench

The source dataset is:
    data/Tools segmentation.v1i.yolov8   (names: hammer, pliers, screw, wrench)

The mapping supports many-to-one: several source names can map to the same
target ID.  Only label (.txt) files and data.yaml are modified in-place.
"""

import re
from pathlib import Path

# ── Paths (hardcoded) ─────────────────────────────────────────────────
DATASET_DIR = Path(__file__).resolve().parent.parent / "data" / "Tools segmentation.v1i.yolov8"

# ── Target class list (canonical) ─────────────────────────────────────
TARGET_NAMES = ["Hammer", "Pliers", "Screwdriver", "Wrench"]

# ── Mapping: source_name  →  target_name ─────────────────────────────
# Many-to-one is supported: just point several source names to the same target.
SOURCE_TO_TARGET_NAME = {
    "hammer":  "Hammer",
    "pliers":  "Pliers",
    "screw":   "Screwdriver",
    "wrench":  "Wrench",
}


def build_id_map(source_names: list[str]) -> dict[int, int]:
    """Return {source_id: target_id} using SOURCE_TO_TARGET_NAME."""
    target_name_to_id = {n: i for i, n in enumerate(TARGET_NAMES)}
    id_map: dict[int, int] = {}

    for src_id, src_name in enumerate(source_names):
        tgt_name = SOURCE_TO_TARGET_NAME.get(src_name)
        if tgt_name is None:
            print(f"  [SKIP] source class {src_id} ('{src_name}') has no mapping — lines with this ID will be REMOVED")
            continue
        id_map[src_id] = target_name_to_id[tgt_name]

    return id_map


def read_source_names_from_yaml(yaml_path: Path) -> list[str]:
    """Parse the 'names' list from a YOLO data.yaml."""
    text = yaml_path.read_text()
    match = re.search(r"names:\s*\[(.+?)\]", text)
    if not match:
        raise ValueError(f"Could not parse 'names' from {yaml_path}")
    raw = match.group(1)
    return [n.strip().strip("'\"") for n in raw.split(",")]


def remap_label_file(path: Path, id_map: dict[int, int]) -> tuple[int, int]:
    """Remap class IDs in a single YOLO label file.

    Returns (total_lines, removed_lines).
    """
    lines = path.read_text().strip().splitlines()
    new_lines: list[str] = []
    removed = 0

    for line in lines:
        parts = line.split()
        if not parts:
            continue
        src_id = int(parts[0])
        if src_id not in id_map:
            removed += 1
            continue
        parts[0] = str(id_map[src_id])
        new_lines.append(" ".join(parts))

    path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
    return len(lines), removed


def update_data_yaml(yaml_path: Path) -> None:
    """Rewrite data.yaml with the target names and nc."""
    text = yaml_path.read_text()
    names_str = "[" + ", ".join(f"'{n}'" for n in TARGET_NAMES) + "]"
    text = re.sub(r"nc:\s*\d+", f"nc: {len(TARGET_NAMES)}", text)
    text = re.sub(r"names:\s*\[.+?\]", f"names: {names_str}", text)
    yaml_path.write_text(text)


def main() -> None:
    yaml_path = DATASET_DIR / "data.yaml"
    print(f"Dataset: {DATASET_DIR}\n")

    # 1. Read current source names
    source_names = read_source_names_from_yaml(yaml_path)
    print(f"Source names ({len(source_names)}): {source_names}")
    print(f"Target names ({len(TARGET_NAMES)}): {TARGET_NAMES}\n")

    # 2. Build ID map
    id_map = build_id_map(source_names)
    print("ID mapping (source_id → target_id):")
    for src_id, tgt_id in sorted(id_map.items()):
        print(f"  {src_id} ('{source_names[src_id]}')  →  {tgt_id} ('{TARGET_NAMES[tgt_id]}')")
    print()

    # 3. Remap label files
    total_files = 0
    total_removed = 0
    for split in ("train", "test", "valid"):
        labels_dir = DATASET_DIR / split / "labels"
        if not labels_dir.exists():
            continue
        files = sorted(labels_dir.glob("*.txt"))
        split_removed = 0
        for f in files:
            _, removed = remap_label_file(f, id_map)
            split_removed += removed
        total_files += len(files)
        total_removed += split_removed
        print(f"  [{split}] {len(files)} files processed, {split_removed} lines removed (unmapped classes)")

    print(f"\nTotal: {total_files} label files, {total_removed} lines removed")

    # 4. Update data.yaml
    update_data_yaml(yaml_path)
    print(f"\ndata.yaml updated → names: {TARGET_NAMES}, nc: {len(TARGET_NAMES)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
