# Pneumonia Detection Project Run Guide

This guide gives end-to-end commands to run the project locally on Windows (PowerShell).

## 1. Prerequisites

1. Install Python 3.10 or 3.11.
2. Install Git (optional, only if cloning).
3. (Optional) Install Kaggle CLI if you need to download the dataset:
   - `pip install kaggle`

## 2. Open Project Folder

```powershell
cd C:\Users\ARNAV\Desktop\BM_DeeplearningModel\pneumonia-detection
```

## 3. Create Virtual Environment (first time only)

From inside `pneumonia-detection`, create the virtual env one level up (this repo already uses `..\ .venv` style paths):

```powershell
python -m venv ..\.venv
```

## 4. Install Dependencies

You can either activate the virtual environment or call its Python directly.

### Option A: Activate virtual env

```powershell
& ..\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option B: No activation (recommended if execution policy blocks scripts)

```powershell
& ..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 5. Dataset Setup

Expected dataset path in this project:

`data/chest_xray/`

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

If you already have this folder structure, skip to section 6.

If you need to download from Kaggle:

1. Put `kaggle.json` in `.kaggle\` (inside `pneumonia-detection`).
2. Run:

```powershell
$env:KAGGLE_CONFIG_DIR=".kaggle"
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
Expand-Archive -Path .\chest-xray-pneumonia.zip -DestinationPath .\data -Force
```

After unzip, ensure final path is `data\chest_xray\...` as shown above.

## 6. Run Data Diagnostics

```powershell
& ..\.venv\Scripts\python.exe -m src.data_loader
```

## 7. Train Models

This trains models and saves checkpoints in `models\`.

```powershell
& ..\.venv\Scripts\python.exe -m src.train
```

## 8. Evaluate Models

This writes evaluation artifacts to `results\` (including model comparison CSV/plot).

```powershell
& ..\.venv\Scripts\python.exe -m src.evaluate
```

## 9. Generate Grad-CAM Samples

```powershell
& ..\.venv\Scripts\python.exe -m src.gradcam
```

Outputs are saved in `results\gradcam_samples\`.

## 10. Launch the Gradio App

```powershell
& ..\.venv\Scripts\python.exe app\app.py
```

Then open the URL shown in terminal (usually `http://127.0.0.1:7860`).

## 11. Quick Run (if everything is already set up)

```powershell
cd C:\Users\ARNAV\Desktop\BM_DeeplearningModel\pneumonia-detection
& ..\.venv\Scripts\python.exe app\app.py
```

## 12. Key Output Files

1. Best app model: `models\best_model.h5`
2. Other trained models:
   - `models\custom_cnn_best.h5`
   - `models\resnet50_frozen_best.h5`
   - `models\resnet50_finetuned_best.h5`
3. Evaluation summary:
   - `results\model_comparison.csv`
   - `results\model_comparison.png`
4. Explainability outputs:
   - `results\gradcam_samples\`

## 13. Troubleshooting

### Error: `allow_flagging` unexpected keyword

Cause: Gradio API mismatch.  
Fix: Use `flagging_mode="never"` in `app/app.py` (already updated in this repo).

### Error: model file not found

If you see `Best model not found at ... models/best_model.h5`, run training first:

```powershell
& ..\.venv\Scripts\python.exe -m src.train
```

### Error: activation script blocked

Use direct Python path commands instead of activation:

```powershell
& ..\.venv\Scripts\python.exe <command>
```

### Port 7860 already in use

Stop the process using port 7860 or launch again after closing previous app sessions.

### TensorFlow oneDNN warning

This is informational and usually safe to ignore.

