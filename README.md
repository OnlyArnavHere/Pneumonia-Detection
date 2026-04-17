# Pneumonia Detection from Chest X-Rays

Deep learning project for binary chest X-ray classification (`NORMAL` vs `PNEUMONIA`) using TensorFlow/Keras. The repository includes dataset loading, model training, evaluation, Grad-CAM explainability, notebooks for experimentation, and a Gradio app for local inference.

## Highlights

## Results
The training/evaluation pipeline is fully implemented, but results depend on dataset authentication and local training completion.
- Binary classifier for chest X-ray pneumonia detection
- Custom CNN baseline plus ResNet50 transfer learning
- Training histories and curve plots saved to `results/`
- Evaluation pipeline for confusion matrices, ROC curves, and model comparison
- Grad-CAM visualization for model explainability
- Gradio app for interactive local predictions

## Repository

- GitHub: `https://github.com/OnlyArnavHere/Pneumonia-Detection`

## Dataset

- Dataset: Chest X-Ray Images (Pneumonia)
- Source: Kaggle
- Link: `https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia`
- Expected local path: `data/chest_xray/`
- Classes: `NORMAL`, `PNEUMONIA`

Expected structure:

```text
data/chest_xray/
train/NORMAL
train/PNEUMONIA
val/NORMAL
val/PNEUMONIA
test/NORMAL
test/PNEUMONIA
```

## Models Included

### 1. Custom CNN

- Three convolution + max-pooling blocks
- Dense classifier head with dropout
- Trained as a baseline model

### 2. ResNet50 Transfer Learning

- ImageNet-pretrained ResNet50 backbone
- Frozen-backbone training stage
- Fine-tuning stage with the last layers unfrozen
- Final best-model alias saved as `models/best_model.h5` for app inference

## Project Structure

```text
pneumonia-detection/
|-- README.md
|-- RUN_GUIDE.md
|-- requirements.txt
|-- .gitignore
|-- app/
|   `-- app.py
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_preprocessing.ipynb
|   |-- 03_cnn_baseline.ipynb
|   |-- 04_transfer_learning.ipynb
|   `-- 05_evaluation.ipynb
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_loader.py
|   |-- evaluate.py
|   |-- gradcam.py
|   |-- model.py
|   `-- train.py
|-- models/
|   `-- .gitkeep
|-- results/
|   |-- .gitkeep
|   |-- history_custom_cnn.json
|   |-- history_resnet50_finetuned.json
|   |-- history_resnet50_frozen.json
|   |-- samples_normal.png
|   |-- samples_pneumonia.png
|   |-- training_curves_custom_cnn.png
|   |-- training_curves_resnet50_finetuned.png
|   `-- training_curves_resnet50_frozen.png
`-- run_all.py
```

## Setup

### Clone and install

```powershell
git clone https://github.com/OnlyArnavHere/Pneumonia-Detection.git
cd Pneumonia-Detection
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation, use the interpreter directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Download the Dataset

If the dataset is not already available locally, place your `kaggle.json` file inside `.kaggle/` and run:

```powershell
$env:KAGGLE_CONFIG_DIR=".kaggle"
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
Expand-Archive -Path .\chest-xray-pneumonia.zip -DestinationPath .\data -Force
```

After extraction, make sure the final folder is `data\chest_xray\...`.

## How to Run

### 1. Data diagnostics

```powershell
.venv\Scripts\python.exe -m src.data_loader
```

### 2. Train models

```powershell
.venv\Scripts\python.exe -m src.train
```

This creates:

- `models/custom_cnn_best.h5`
- `models/resnet50_frozen_best.h5`
- `models/resnet50_finetuned_best.h5`
- `models/best_model.h5`
- training history JSON files in `results/`
- training curve plots in `results/`

### 3. Evaluate saved models

```powershell
.venv\Scripts\python.exe -m src.evaluate
```

This generates:

- `results/model_comparison.csv`
- `results/model_comparison.png`
- per-model confusion matrices
- per-model ROC curves

### 4. Generate Grad-CAM outputs

```powershell
.venv\Scripts\python.exe -m src.gradcam
```

Outputs are written to `results/gradcam_samples/`.

### 5. Launch the Gradio app

```powershell
.venv\Scripts\python.exe app\app.py
```

Then open the local URL shown in the terminal, usually `http://127.0.0.1:7860`.

## App Behavior

The Gradio app:

- loads `models/best_model.h5`
- accepts a chest X-ray image upload
- returns class probabilities for `PNEUMONIA` and `NORMAL`
- generates a Grad-CAM heatmap for the prediction

If `models/best_model.h5` does not exist, train the models first with `python -m src.train`.

## Current Repository Artifacts

The repo already includes lightweight training artifacts such as:

- `results/history_custom_cnn.json`
- `results/history_resnet50_frozen.json`
- `results/history_resnet50_finetuned.json`
- `results/training_curves_*.png`
- sample visualization images in `results/`

Large assets and sensitive files are intentionally excluded from Git:

- dataset contents under `data/`
- model weight files under `models/*.h5`
- logs and cache directories
- `.kaggle/` credentials

## Notes on Results

Training history artifacts are included, but the full comparison outputs from `src.evaluate` are generated locally after saved model files are available. That means `results/model_comparison.csv` and related evaluation charts may not exist in a fresh clone until you run evaluation yourself.

## Limitations

- This is a research and educational project, not a medical diagnosis system.
- Performance depends on dataset quality, class balance, and train/test domain similarity.
- Real-world clinical generalization is not guaranteed.

## Future Improvements

- External validation on additional X-ray datasets
- Threshold calibration and uncertainty estimation
- Better experiment tracking
- Deployment to a hosted demo
- More robust explainability and failure-case analysis

## References

- He et al., Deep Residual Learning for Image Recognition
- Rajpurkar et al., CheXNet
- Kaggle Chest X-Ray Images (Pneumonia) dataset by Paul Mooney
