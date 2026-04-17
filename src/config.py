"""Global configuration values for the pneumonia detection project."""

from pathlib import Path

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-5
DATA_DIR = "data/chest_xray"
MODEL_SAVE_PATH = "models/best_model.h5"
RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / DATA_DIR
TRAIN_DIR = DATA_PATH / "train"
VAL_DIR = DATA_PATH / "val"
TEST_DIR = DATA_PATH / "test"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
GRADCAM_DIR = RESULTS_DIR / "gradcam_samples"
LOGS_DIR = PROJECT_ROOT / "logs"

CUSTOM_MODEL_PATH = MODELS_DIR / "custom_cnn_best.h5"
RESNET_FROZEN_MODEL_PATH = MODELS_DIR / "resnet50_frozen_best.h5"
RESNET_FINETUNE_MODEL_PATH = MODELS_DIR / "resnet50_finetuned_best.h5"
MODEL_COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"
MODEL_COMPARISON_PLOT = RESULTS_DIR / "model_comparison.png"
APP_SHARE = False
