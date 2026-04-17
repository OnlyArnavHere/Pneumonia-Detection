"""Gradio app for pneumonia detection with Grad-CAM visualization."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".cache" / "matplotlib")

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

from src import config
from src.gradcam import generate_gradcam

MODEL_PATH = config.PROJECT_ROOT / config.MODEL_SAVE_PATH


def load_model() -> tf.keras.Model:
    """Load the best trained model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best model not found at {MODEL_PATH}. Train the model before launching the app."
        )
    return tf.keras.models.load_model(str(MODEL_PATH))


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Preprocess input image identically to training pipeline."""
    img = img.convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def _format_prediction(prob_pneumonia: float) -> Dict[str, float]:
    """Format probabilities for Gradio label output."""
    return {
        "PNEUMONIA": float(prob_pneumonia),
        "NORMAL": float(1.0 - prob_pneumonia),
    }


def predict_and_explain(img: Image.Image) -> Tuple[Dict[str, float], np.ndarray]:
    """Run model inference and return class probabilities with Grad-CAM heatmap."""
    if img is None:
        return {"PNEUMONIA": 0.0, "NORMAL": 0.0}, np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
    input_tensor = preprocess_image(img)
    prob = float(MODEL.predict(input_tensor, verbose=0).ravel()[0])
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    img.save(tmp_path)
    heatmap = generate_gradcam(MODEL, tmp_path)
    tmp_path.unlink(missing_ok=True)
    return _format_prediction(prob), heatmap


def _get_examples() -> list[list[str]]:
    """Collect 2-3 example images from test set if available."""
    test_dir = config.TEST_DIR
    if not test_dir.exists():
        return []
    paths = []
    for class_name in ["NORMAL", "PNEUMONIA"]:
        class_dir = test_dir / class_name
        if class_dir.exists():
            for p in sorted(class_dir.glob("*"))[:2]:
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    paths.append([str(p)])
    return paths[:3]


MODEL = load_model()

DESCRIPTION = (
    "This model predicts whether an uploaded chest X-ray appears to show pneumonia patterns. "
    "It also generates a Grad-CAM heatmap to visualize image regions that influenced the prediction.\n\n"
    "Disclaimer: This tool is for educational purposes only and is not a medical diagnostic device."
)

interface = gr.Interface(
    fn=predict_and_explain,
    inputs=gr.Image(type="pil", label="Upload Chest X-Ray"),
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction Probabilities"),
        gr.Image(label="Grad-CAM Heatmap"),
    ],
    title="Pneumonia Detection from Chest X-Ray",
    description=DESCRIPTION,
    examples=_get_examples(),
    flagging_mode="never",
)


if __name__ == "__main__":
    interface.launch(share=config.APP_SHARE)
