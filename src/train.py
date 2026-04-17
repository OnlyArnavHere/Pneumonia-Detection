"""Training pipeline for custom CNN and ResNet50 transfer learning models."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

os.environ["MPLCONFIGDIR"] = str((Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from src import config
from src.data_loader import DataLoader
from src.model import ModelBuilder


def setup_logging() -> None:
    """Configure logging output with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class Trainer:
    """Model trainer with callbacks and plot utilities."""

    def __init__(self) -> None:
        """Create expected output directories."""
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        model: tf.keras.Model,
        train_gen: tf.keras.preprocessing.image.DirectoryIterator,
        val_gen: tf.keras.preprocessing.image.DirectoryIterator,
        class_weights: dict,
        model_name: str,
        epochs: int,
        save_path: Path,
        initial_epoch: int = 0,
    ) -> tf.keras.callbacks.History:
        """Train model with standard callbacks and return history."""
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
            ModelCheckpoint(filepath=str(save_path), monitor="val_loss", save_best_only=True),
            TensorBoard(log_dir=str(config.LOGS_DIR / model_name)),
        ]
        logging.info("Starting training for %s", model_name)
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            initial_epoch=initial_epoch,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )
        logging.info("Finished training for %s", model_name)
        return history

    def plot_training_curves(self, history: tf.keras.callbacks.History, model_name: str) -> None:
        """Plot and save train/validation accuracy and loss curves."""
        hist = history.history
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(hist.get("accuracy", []), label="Train Accuracy")
        axes[0].plot(hist.get("val_accuracy", []), label="Val Accuracy")
        axes[0].set_title(f"Accuracy - {model_name}")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()

        axes[1].plot(hist.get("loss", []), label="Train Loss")
        axes[1].plot(hist.get("val_loss", []), label="Val Loss")
        axes[1].set_title(f"Loss - {model_name}")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()

        output_path = config.RESULTS_DIR / f"training_curves_{model_name}.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        logging.info("Saved training curves to %s", output_path)

    @staticmethod
    def save_history(history: tf.keras.callbacks.History, model_name: str) -> None:
        """Save raw training history JSON for reproducibility."""
        history_path = config.RESULTS_DIR / f"history_{model_name}.json"
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history.history, f, indent=2)
        logging.info("Saved training history to %s", history_path)


def run_training_pipeline() -> dict[str, tf.keras.Model]:
    """Train custom CNN and ResNet50 frozen/fine-tuned variants."""
    setup_logging()
    set_seed()

    loader = DataLoader()
    loader.print_dataset_stats()
    train_gen, val_gen, _ = loader.get_generators()
    class_weights = loader.get_class_weights(train_gen)
    loader.visualize_samples(train_gen)

    builder = ModelBuilder()
    trainer = Trainer()
    trained_models: dict[str, tf.keras.Model] = {}

    custom_model = builder.build_custom_cnn()
    custom_history = trainer.train(
        model=custom_model,
        train_gen=train_gen,
        val_gen=val_gen,
        class_weights=class_weights,
        model_name="custom_cnn",
        epochs=config.EPOCHS,
        save_path=config.CUSTOM_MODEL_PATH,
    )
    trainer.plot_training_curves(custom_history, "custom_cnn")
    trainer.save_history(custom_history, "custom_cnn")
    custom_model.load_weights(str(config.CUSTOM_MODEL_PATH))
    trained_models["custom_cnn"] = custom_model

    resnet_model = builder.build_transfer_model(base="resnet50")
    resnet_frozen_history = trainer.train(
        model=resnet_model,
        train_gen=train_gen,
        val_gen=val_gen,
        class_weights=class_weights,
        model_name="resnet50_frozen",
        epochs=config.EPOCHS,
        save_path=config.RESNET_FROZEN_MODEL_PATH,
    )
    trainer.plot_training_curves(resnet_frozen_history, "resnet50_frozen")
    trainer.save_history(resnet_frozen_history, "resnet50_frozen")

    resnet_model = builder.unfreeze_and_finetune(resnet_model, num_layers=20)
    finetune_epochs = 15
    resnet_finetune_history = trainer.train(
        model=resnet_model,
        train_gen=train_gen,
        val_gen=val_gen,
        class_weights=class_weights,
        model_name="resnet50_finetuned",
        epochs=config.EPOCHS + finetune_epochs,
        initial_epoch=config.EPOCHS,
        save_path=config.RESNET_FINETUNE_MODEL_PATH,
    )
    trainer.plot_training_curves(resnet_finetune_history, "resnet50_finetuned")
    trainer.save_history(resnet_finetune_history, "resnet50_finetuned")
    resnet_model.load_weights(str(config.RESNET_FINETUNE_MODEL_PATH))
    trained_models["resnet50_frozen"] = tf.keras.models.load_model(str(config.RESNET_FROZEN_MODEL_PATH))
    trained_models["resnet50_finetuned"] = resnet_model

    final_path = config.PROJECT_ROOT / config.MODEL_SAVE_PATH
    final_path.parent.mkdir(parents=True, exist_ok=True)
    resnet_model.save(str(final_path))
    logging.info("Saved best model alias to %s", final_path)

    return trained_models


if __name__ == "__main__":
    run_training_pipeline()
