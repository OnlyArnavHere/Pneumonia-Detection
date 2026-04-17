"""End-to-end runner for pneumonia detection project phases."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".cache" / "matplotlib")

import tensorflow as tf

from src import config
from src.data_loader import DataLoader, set_seed
from src.evaluate import Evaluator
from src.gradcam import save_gradcam_samples
from src.train import run_training_pipeline


def setup_logging() -> None:
    """Configure timestamped logging for orchestration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_all() -> None:
    """Execute data checks, training, evaluation, and Grad-CAM generation."""
    setup_logging()
    set_seed()
    logging.info("Starting full project pipeline.")

    loader = DataLoader()
    loader.print_dataset_stats()
    train_gen, val_gen, test_gen = loader.get_generators()
    class_weights = loader.get_class_weights(train_gen)
    loader.visualize_samples(train_gen)
    logging.info("Data diagnostics complete.")

    run_training_pipeline()
    logging.info("Training complete.")

    evaluator = Evaluator()
    model_paths = {
        "custom_cnn": config.CUSTOM_MODEL_PATH,
        "resnet50_frozen": config.RESNET_FROZEN_MODEL_PATH,
        "resnet50_finetuned": config.RESNET_FINETUNE_MODEL_PATH,
    }
    results = {}
    best_name = None
    best_f1 = -1.0
    best_model = None
    for model_name, model_path in model_paths.items():
        model = tf.keras.models.load_model(str(model_path))
        metrics = evaluator.evaluate_model(model, test_gen, model_name=model_name)
        results[model_name] = metrics
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = model_name
            best_model = model
    evaluator.compare_models(results)
    logging.info("Evaluation complete. Best model: %s (F1=%.4f)", best_name, best_f1)

    if best_model is not None:
        save_gradcam_samples(best_model, test_gen, n=5)
        logging.info("Grad-CAM generation complete.")
    logging.info("Pipeline finished successfully.")


if __name__ == "__main__":
    try:
        run_all()
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        sys.exit(1)
