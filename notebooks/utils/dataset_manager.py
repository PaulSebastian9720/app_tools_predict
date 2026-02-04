"""
Dataset management utilities for YOLO-formatted object detection datasets.

This module prepares balanced training subsets for:
- Initial bootstrap transfer learning
- Incremental fine-tuning stages

A fixed test set is created by copying the original validation set.
The original dataset is never modified.
"""

import random
import shutil
from pathlib import Path
from collections import Counter, defaultdict


class YOLODatasetManager:
    """
    Manages YOLO-format datasets for bootstrap and incremental training.
    """

    def __init__(self, source_dir: str, output_dir: str, seed: int = 42,
                 class_filter: dict = None):
        """
        Initialize dataset paths and random seed.

        Parameters
        ----------
        source_dir : str
            Path to the original YOLO dataset directory.
        output_dir : str
            Path where processed datasets will be created.
        seed : int
            Random seed for reproducibility.
        class_filter : dict, optional
            Mapping {original_class_id: new_class_id} to keep only
            selected classes and remap their IDs in the output labels.
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_images = self.source_dir / "train" / "images"
        self.train_labels = self.source_dir / "train" / "labels"
        self.valid_images = self.source_dir / "test" / "images"
        self.valid_labels = self.source_dir / "test" / "labels"
        self.class_filter = class_filter
        random.seed(seed)

    def _read_label_file(self, label_path: Path) -> list[int]:
        """
        Read class IDs from a YOLO label file.
        """
        classes = []
        with open(label_path, "r") as file:
            for line in file:
                classes.append(int(line.split()[0]))
        return classes

    def _filter_label(self, label_path: Path) -> list[str]:
        """
        Filter label lines to keep only selected classes and remap IDs.
        """
        filtered = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts and int(parts[0]) in self.class_filter:
                    parts[0] = str(self.class_filter[int(parts[0])])
                    filtered.append(" ".join(parts))
        return filtered

    def analyze_distribution(self):
        """
        Analyze class distribution in the training set.

        Returns
        -------
        dict
            Mapping from class ID to list of image IDs.
        """
        class_to_images = defaultdict(set)

        for label_file in self.train_labels.glob("*.txt"):
            image_id = label_file.stem
            classes = set(self._read_label_file(label_file))
            for class_id in classes:
                if self.class_filter is None or class_id in self.class_filter:
                    class_to_images[class_id].add(image_id)

        return class_to_images

    def _copy_samples(self, image_ids: list[str], target_dir: Path):
        """
        Copy selected images and labels into a YOLO directory structure.
        When class_filter is set, labels are filtered and remapped.
        """
        images_out = target_dir / "images"
        labels_out = target_dir / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for image_id in image_ids:
            src_label = self.train_labels / f"{image_id}.txt"
            if self.class_filter and src_label.exists():
                lines = self._filter_label(src_label)
                if not lines:
                    continue
                shutil.copy(self.train_images / f"{image_id}.jpg", images_out)
                with open(labels_out / f"{image_id}.txt", "w") as f:
                    f.write("\n".join(lines) + "\n")
            else:
                shutil.copy(self.train_images / f"{image_id}.jpg", images_out)
                shutil.copy(self.train_labels / f"{image_id}.txt", labels_out)

    def create_bootstrap(self, max_images: int = 75) -> set[str]:
        """
        Create a small balanced bootstrap dataset for initial transfer learning.

        Returns
        -------
        set[str]
            Image IDs selected for the bootstrap dataset.
        """
        class_to_images = self.analyze_distribution()

        min_class_count = min(len(v) for v in class_to_images.values())
        images_per_class = max(
            1, min(min_class_count, max_images // len(class_to_images))
        )

        selected_images = set()
        for images in class_to_images.values():
            selected_images.update(
                random.sample(list(images), images_per_class)
            )

        selected_images = set(list(selected_images)[:max_images])

        output_path = self.output_dir / "train" / "bootstrap"
        self._copy_samples(list(selected_images), output_path)

        self._bootstrap_ids = selected_images
        return selected_images

    def create_incremental_splits(self, num_splits: int):
        """
        Create incremental training datasets from images NOT in bootstrap.

        Parameters
        ----------
        num_splits : int
            Number of incremental datasets to generate.
        """
        class_to_images = self.analyze_distribution()
        all_images = set().union(*class_to_images.values())

        # Exclude bootstrap images
        bootstrap_ids = getattr(self, "_bootstrap_ids", set())
        remaining = list(all_images - bootstrap_ids)
        random.shuffle(remaining)

        splits = [remaining[i::num_splits] for i in range(num_splits)]

        for index, split in enumerate(splits, start=1):
            output_path = self.output_dir / "train" / f"dataset_{index}"
            self._copy_samples(split, output_path)

    def create_test_set(self):
        """
        Create a fixed test set by copying the original test/validation data.
        When class_filter is set, labels are filtered and remapped.
        """
        test_images = self.output_dir / "test" / "images"
        test_labels = self.output_dir / "test" / "labels"
        test_images.mkdir(parents=True, exist_ok=True)
        test_labels.mkdir(parents=True, exist_ok=True)

        for img in self.valid_images.glob("*"):
            label_path = self.valid_labels / f"{img.stem}.txt"
            if self.class_filter and label_path.exists():
                lines = self._filter_label(label_path)
                if not lines:
                    continue
                shutil.copy(img, test_images)
                with open(test_labels / f"{img.stem}.txt", "w") as f:
                    f.write("\n".join(lines) + "\n")
            else:
                shutil.copy(img, test_images)
                if label_path.exists():
                    shutil.copy(label_path, test_labels)


if __name__ == "__main__":
    SOURCE_DATASET = "data/Tools segmentation.v1i.yolov8"
    OUTPUT_DATASET = "data/pre_process_2"

    # Top 4 classes (Pliers + plier merged) (original_id -> new_id)
    # Hammer(3)->0, Pliers(6)->1, Screwdriver(7)->2, Wrench(9)->3, plier(10)->1
    #CLASS_FILTER = {3: 0, 6: 1, 7: 2, 9: 3, 10: 1}

    manager = YOLODatasetManager(SOURCE_DATASET, OUTPUT_DATASET,
                                 class_filter=None)
    #manager.create_bootstrap(max_images=200)
    manager.create_incremental_splits(num_splits=8)
    manager.create_test_set()
