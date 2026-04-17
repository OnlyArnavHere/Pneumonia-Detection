"""Data loading, statistics, and sample visualization utilities."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Tuple

os.environ["MPLCONFIGDIR"] = str((Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src import config


def _setup_logging() -> None:
    """Configure logging with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Set random seeds across libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class DataLoader:
    """Load chest X-ray data, compute class weights, and save visual diagnostics."""

    def __init__(self) -> None:
        """Initialize output folders and validate expected dataset paths."""
        self.train_dir = config.TRAIN_DIR
        self.val_dir = config.VAL_DIR
        self.test_dir = config.TEST_DIR
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        config.GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    def _check_dataset_exists(self) -> None:
        """Raise a clear error if dataset directories are missing."""
        expected_paths = [self.train_dir, self.val_dir, self.test_dir]
        missing = [str(p) for p in expected_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Dataset not found. Expected folders are missing:\n"
                + "\n".join(missing)
                + "\n\nDownload and extract with:\n"
                "kaggle datasets download -d paultimothymooney/chest-xray-pneumonia\n"
                "unzip chest-xray-pneumonia.zip -d data/"
            )

    def get_generators(
        self,
    ) -> Tuple[tf.keras.preprocessing.image.DirectoryIterator, ...]:
        """Create train/validation/test generators using requested augmentation settings."""
        self._check_dataset_exists()
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
        )
        eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

        train_gen = train_datagen.flow_from_directory(
            directory=str(self.train_dir),
            target_size=(config.IMG_SIZE, config.IMG_SIZE),
            batch_size=config.BATCH_SIZE,
            class_mode="binary",
            shuffle=True,
            seed=config.RANDOM_SEED,
        )
        val_gen = eval_datagen.flow_from_directory(
            directory=str(self.val_dir),
            target_size=(config.IMG_SIZE, config.IMG_SIZE),
            batch_size=config.BATCH_SIZE,
            class_mode="binary",
            shuffle=False,
            seed=config.RANDOM_SEED,
        )
        test_gen = eval_datagen.flow_from_directory(
            directory=str(self.test_dir),
            target_size=(config.IMG_SIZE, config.IMG_SIZE),
            batch_size=config.BATCH_SIZE,
            class_mode="binary",
            shuffle=False,
            seed=config.RANDOM_SEED,
        )
        return train_gen, val_gen, test_gen

    def get_class_weights(
        self, train_gen: tf.keras.preprocessing.image.DirectoryIterator
    ) -> dict:
        """Compute class weights from training labels to address class imbalance."""
        y = train_gen.classes
        classes = np.unique(y)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        class_weights = {int(cls): float(w) for cls, w in zip(classes, weights)}
        logging.info("Computed class weights: %s", class_weights)
        return class_weights

    def print_dataset_stats(self) -> None:
        """Print image counts by split and class."""
        self._check_dataset_exists()
        split_map = {"train": self.train_dir, "val": self.val_dir, "test": self.test_dir}
        logging.info("Dataset statistics:")
        for split_name, split_path in split_map.items():
            class_counts = {}
            total = 0
            for class_dir in sorted(split_path.iterdir()):
                if class_dir.is_dir():
                    count = len(
                        [p for p in class_dir.rglob("*") if p.suffix.lower() in {".jpeg", ".jpg", ".png"}]
                    )
                    class_counts[class_dir.name] = count
                    total += count
            logging.info("%s total: %d | per class: %s", split_name, total, class_counts)

    def visualize_samples(
        self, train_gen: tf.keras.preprocessing.image.DirectoryIterator | None = None
    ) -> None:
        """Save 3x3 sample image grid per class to results directory."""
        self._check_dataset_exists()
        class_dirs = sorted([p for p in self.train_dir.iterdir() if p.is_dir()])
        for class_dir in class_dirs:
            image_paths = [
                p
                for p in class_dir.rglob("*")
                if p.suffix.lower() in {".jpeg", ".jpg", ".png"}
            ]
            if not image_paths:
                logging.warning("No images found in %s", class_dir)
                continue
            sample_paths = image_paths[:9]
            fig, axes = plt.subplots(3, 3, figsize=(10, 10))
            fig.suptitle(f"Sample Images - {class_dir.name}", fontsize=14)
            for idx, ax in enumerate(axes.flat):
                if idx < len(sample_paths):
                    img = tf.keras.utils.load_img(sample_paths[idx], target_size=(config.IMG_SIZE, config.IMG_SIZE))
                    img_arr = tf.keras.utils.img_to_array(img).astype(np.uint8)
                    ax.imshow(img_arr.astype(np.uint8), cmap="gray")
                    ax.set_title(sample_paths[idx].name[:28], fontsize=8)
                ax.axis("off")
            output_path = config.RESULTS_DIR / f"samples_{class_dir.name.lower()}.png"
            fig.tight_layout()
            fig.savefig(output_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            logging.info("Saved sample grid: %s", output_path)


def main() -> None:
    """Run data inspection workflow."""
    _setup_logging()
    set_seed()
    sns.set_style("whitegrid")
    loader = DataLoader()
    loader.print_dataset_stats()
    train_gen, _, _ = loader.get_generators()
    loader.get_class_weights(train_gen)
    loader.visualize_samples(train_gen)
    logging.info("Data loading and visualization completed.")


if __name__ == "__main__":
    main()
