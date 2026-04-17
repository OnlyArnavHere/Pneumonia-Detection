"""Evaluation utilities for trained pneumonia detection models."""

from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str((Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config
from src.data_loader import DataLoader


def setup_logging() -> None:
    """Configure logging output with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class Evaluator:
    """Evaluate models and generate comparable reports and visualizations."""

    def __init__(self) -> None:
        """Initialize evaluator and create result directory."""
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def evaluate_model(
        self,
        model: tf.keras.Model,
        test_gen: tf.keras.preprocessing.image.DirectoryIterator,
        model_name: str = "model",
    ) -> dict:
        """Evaluate a model on the test set and return scalar metrics."""
        test_gen.reset()
        probas = model.predict(test_gen, verbose=1).ravel()
        y_true = test_gen.classes
        y_pred = (probas >= 0.5).astype(int)

        metrics = {
            "model": model_name,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "auc_roc": float(roc_auc_score(y_true, probas)),
        }

        logging.info("[%s] Accuracy: %.4f", model_name, metrics["accuracy"])
        logging.info("[%s] Precision: %.4f", model_name, metrics["precision"])
        logging.info("[%s] Recall: %.4f", model_name, metrics["recall"])
        logging.info("[%s] F1-score: %.4f", model_name, metrics["f1"])
        logging.info("[%s] AUC-ROC: %.4f", model_name, metrics["auc_roc"])
        logging.info(
            "[%s] Classification report:\n%s",
            model_name,
            classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"], zero_division=0),
        )

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["NORMAL", "PNEUMONIA"],
            yticklabels=["NORMAL", "PNEUMONIA"],
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix - {model_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        cm_path = config.RESULTS_DIR / f"confusion_matrix_{model_name}.png"
        fig.tight_layout()
        fig.savefig(cm_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        logging.info("Saved confusion matrix: %s", cm_path)

        fpr, tpr, _ = roc_curve(y_true, probas)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_title(f"ROC Curve - {model_name}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        roc_path = config.RESULTS_DIR / f"roc_curve_{model_name}.png"
        fig.tight_layout()
        fig.savefig(roc_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        logging.info("Saved ROC curve: %s", roc_path)

        return metrics

    def compare_models(self, results_dict: dict[str, dict]) -> pd.DataFrame:
        """Create and save comparison table and F1/AUC bar chart across models."""
        df = pd.DataFrame(results_dict.values()).sort_values(by="f1", ascending=False).reset_index(drop=True)
        df.to_csv(config.MODEL_COMPARISON_CSV, index=False)
        logging.info("Saved model comparison CSV: %s", config.MODEL_COMPARISON_CSV)
        logging.info("Model comparison table:\n%s", df.to_string(index=False))

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_df = df.melt(id_vars="model", value_vars=["f1", "auc_roc"], var_name="metric", value_name="value")
        sns.barplot(data=plot_df, x="model", y="value", hue="metric", ax=ax)
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Model Comparison (F1 and AUC-ROC)")
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        plt.xticks(rotation=15)
        fig.tight_layout()
        fig.savefig(config.MODEL_COMPARISON_PLOT, dpi=180, bbox_inches="tight")
        plt.close(fig)
        logging.info("Saved model comparison plot: %s", config.MODEL_COMPARISON_PLOT)
        return df


def main() -> None:
    """Evaluate all available saved models and compare performance."""
    setup_logging()
    loader = DataLoader()
    _, _, test_gen = loader.get_generators()

    model_paths = {
        "custom_cnn": config.CUSTOM_MODEL_PATH,
        "resnet50_frozen": config.RESNET_FROZEN_MODEL_PATH,
        "resnet50_finetuned": config.RESNET_FINETUNE_MODEL_PATH,
    }
    evaluator = Evaluator()
    results: dict[str, dict] = {}
    for name, path in model_paths.items():
        if not Path(path).exists():
            logging.warning("Skipping %s, model not found at %s", name, path)
            continue
        model = tf.keras.models.load_model(str(path))
        results[name] = evaluator.evaluate_model(model, test_gen, model_name=name)
    if results:
        evaluator.compare_models(results)
    else:
        logging.warning("No models found to evaluate.")


if __name__ == "__main__":
    main()
