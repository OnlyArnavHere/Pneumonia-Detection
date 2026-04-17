# Pneumonia Detection from Chest X-Ray using Deep Learning

## Overview
This project builds a binary deep learning classifier to detect **PNEUMONIA** vs **NORMAL** from chest X-ray images.  
It includes:
- data loading and augmentation
- custom CNN baseline
- transfer learning with ResNet50 (frozen + fine-tuned)
- evaluation (accuracy, precision, recall, F1, AUC-ROC, confusion matrix)
- Grad-CAM explainability
- deployable Gradio app

## Demo
Hugging Face Spaces URL: `TBD`

## Results
The training/evaluation pipeline is fully implemented, but results depend on dataset authentication and local training completion.

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| custom_cnn | pending | pending | pending | pending | pending |
| resnet50_frozen | pending | pending | pending | pending | pending |
| resnet50_finetuned | pending | pending | pending | pending | pending |

After running:
```bash
python -m src.train
python -m src.evaluate
```
metrics will be written to `results/model_comparison.csv`.

## Dataset
- Name: Chest X-Ray Images (Pneumonia)
- Source: Kaggle, Paul Mooney
- Link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Classes: `NORMAL`, `PNEUMONIA`
- Pre-split folders: `train/`, `val/`, `test/`

Download command:
```bash
mkdir -p .kaggle
# place kaggle.json inside .kaggle/ (or set KAGGLE_USERNAME and KAGGLE_KEY)
export KAGGLE_CONFIG_DIR=.kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

## Architecture
1. Custom CNN:
- Conv2D(32) -> MaxPool
- Conv2D(64) -> MaxPool
- Conv2D(128) -> MaxPool
- Flatten -> Dense(256, relu) -> Dropout(0.5) -> Dense(1, sigmoid)

2. Transfer Learning (ResNet50 / VGG16 option):
- ImageNet pretrained backbone (`include_top=False`)
- GlobalAveragePooling2D -> Dense(256, relu) -> Dropout(0.5) -> Dense(1, sigmoid)
- Stage 1: freeze full backbone
- Stage 2: unfreeze last 20 layers and fine-tune with lower LR

## Project Structure
```text
pneumonia-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_cnn_baseline.ipynb
│   ├── 04_transfer_learning.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── gradcam.py
├── app/
│   └── app.py
├── results/
│   └── gradcam_samples/
└── models/
    └── .gitkeep
```

## Setup & Installation
```bash
git clone <your-repo-url>
cd pneumonia-detection
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/Mac
source .venv/bin/activate
pip install -r requirements.txt
```

## How to Run
1. Data diagnostics:
```bash
python -m src.data_loader
```

2. Train all models:
```bash
python -m src.train
```

3. Evaluate saved models:
```bash
python -m src.evaluate
```

4. Generate Grad-CAM samples:
```bash
python -m src.gradcam
```

5. Launch Gradio app:
```bash
python app/app.py
```

## Grad-CAM Visualizations
Generated artifacts are saved in:
- `results/gradcam_samples/`
- `results/gradcam_samples/gradcam_grid.png`

## Limitations & Future Work
- Dataset label quality and class imbalance can bias performance.
- Performance may drop on external hospitals/devices due to domain shift.
- This is **not** a medical diagnosis system; it is a research/educational model.
- Future work:
  - external validation datasets
  - calibration + uncertainty estimation
  - segmentation-guided models
  - better threshold tuning and cost-sensitive optimization

## References
- Rajpurkar et al., CheXNet (2017)
- He et al., Deep Residual Learning (2016)
- Dataset: Mooney (2018), Kaggle
