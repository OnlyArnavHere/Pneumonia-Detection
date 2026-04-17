"""Grad-CAM generation utilities for model explainability."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str((Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src import config
from src.data_loader import DataLoader

try:
    from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
    from tf_keras_vis.utils.scores import BinaryScore
except ImportError:  # pragma: no cover
    GradcamPlusPlus = None
    ReplaceToLinear = None
    BinaryScore = None


def setup_logging() -> None:
    """Configure logging output with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_and_preprocess(img_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load image from disk and produce normalized tensor input."""
    img = tf.keras.utils.load_img(img_path, target_size=(config.IMG_SIZE, config.IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    input_tensor = np.expand_dims(img_array / 255.0, axis=0)
    return img_array.astype(np.uint8), input_tensor


def _find_last_conv_layer_name(model: tf.keras.Model) -> str:
    """Find the last convolutional layer name from the model graph."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model for Grad-CAM.")


def generate_gradcam(
    model: tf.keras.Model,
    img_path: str | Path,
    layer_name: str = "conv5_block3_out",
) -> np.ndarray:
    """Generate a Grad-CAM++ overlay image for a single X-ray."""
    if GradcamPlusPlus is None:
        raise ImportError("tf-keras-vis is not installed. Please install dependencies first.")
    img_path = Path(img_path)
    original, input_tensor = _load_and_preprocess(img_path)

    if layer_name not in [l.name for l in model.layers]:
        layer_name = _find_last_conv_layer_name(model)

    gradcam = GradcamPlusPlus(
        model,
        model_modifier=ReplaceToLinear(),
        clone=True,
    )
    score = BinaryScore(1)
    cam = gradcam(score, input_tensor, penultimate_layer=layer_name)
    cam = np.uint8(255 * cam[0])
    cam = cv2.resize(cam, (original.shape[1], original.shape[0]))
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(original, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def save_gradcam_samples(model: tf.keras.Model, test_gen, n: int = 5) -> None:
    """Save Grad-CAM grids for correct and incorrect predictions."""
    if GradcamPlusPlus is None:
        raise ImportError("tf-keras-vis is not installed. Please install dependencies first.")
    config.GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    test_gen.reset()
    probas = model.predict(test_gen, verbose=1).ravel()
    y_true = test_gen.classes
    y_pred = (probas >= 0.5).astype(int)
    filepaths = [Path(p) for p in test_gen.filepaths]

    correct_indices = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t == p]
    incorrect_indices = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]

    random.seed(config.RANDOM_SEED)
    random.shuffle(correct_indices)
    random.shuffle(incorrect_indices)
    selected = [("correct", idx) for idx in correct_indices[:n]] + [
        ("incorrect", idx) for idx in incorrect_indices[:n]
    ]
    if not selected:
        logging.warning("No Grad-CAM samples selected.")
        return

    rows = int(np.ceil(len(selected) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, (status, idx) in enumerate(selected):
        ax = axes[i]
        img_path = filepaths[idx]
        overlay = generate_gradcam(model, img_path)
        true_label = "PNEUMONIA" if y_true[idx] == 1 else "NORMAL"
        pred_label = "PNEUMONIA" if y_pred[idx] == 1 else "NORMAL"
        confidence = probas[idx] if y_pred[idx] == 1 else 1.0 - probas[idx]
        ax.imshow(overlay)
        ax.set_title(
            f"{status.upper()} | True: {true_label} | Pred: {pred_label} | Conf: {confidence:.3f}",
            fontsize=9,
        )
        ax.axis("off")
        out_file = config.GRADCAM_DIR / f"{status}_{i+1}_{img_path.stem}.png"
        plt.imsave(out_file, overlay)

    for j in range(len(selected), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    grid_path = config.GRADCAM_DIR / "gradcam_grid.png"
    fig.savefig(grid_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved Grad-CAM samples to %s", config.GRADCAM_DIR)


def main() -> None:
    """Generate Grad-CAM samples for the best saved model."""
    setup_logging()
    if not config.RESNET_FINETUNE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best model not found at {config.RESNET_FINETUNE_MODEL_PATH}. Train models first."
        )
    model = tf.keras.models.load_model(str(config.RESNET_FINETUNE_MODEL_PATH))
    loader = DataLoader()
    _, _, test_gen = loader.get_generators()
    save_gradcam_samples(model, test_gen, n=5)


if __name__ == "__main__":
    main()
