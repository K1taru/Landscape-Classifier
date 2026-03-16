# Landscape Classifier

Scene recognition using ResNet50 transfer learning on the Intel Image Classification dataset. Classifies images into 6 natural and urban scene categories with **95.78% validation accuracy**.

---

## Classes

| Label | Description |
|---|---|
| `buildings` | Urban structures and architecture |
| `forest` | Dense tree coverage and woodland |
| `glacier` | Ice fields and glacier landscapes |
| `mountain` | Mountain peaks and rocky terrain |
| `sea` | Ocean, coastline, and open water |
| `street` | Urban roads and city streets |

---

## Results

| Metric | Value |
|---|---|
| Best Validation Accuracy | **95.78%** |
| Best Val Top-5 Accuracy | 100.00% |
| Best Epoch | 8 / 30 |
| Total Training Time | 17m 57s |
| Architecture | ResNet50 (IMAGENET1K_V2) |

---

## Project Structure

```
Landscape-Classifier/
├── src/
│   ├── data/
│   │   └── raw/
│   │       └── Datasets/          # Dataset root (6 class folders, not committed)
│   │           ├── buildings/
│   │           ├── forest/
│   │           ├── glacier/
│   │           ├── mountain/
│   │           ├── sea/
│   │           └── street/
│   ├── models/
│   │   └── results/               # Per-run training reports and visualizations
│   ├── notebooks/
│   │   └── landscape_classifier_model.ipynb   # Main training pipeline
│   ├── scripts/
│   │   ├── dataset_counter.py     # Dataset analysis utility
│   │   ├── gpu_utils.py           # GPU/CUDA detection utilities
│   │   ├── training_report.py     # Auto-generates .md training report
│   │   └── training_visualizer.py # Gradient descent visualization
│   └── utils/
│       └── kaggle_dataset_installer.py  # Download dataset from Kaggle
├── docs/
│   └── Activity.md
├── App-Inference/                 # Inference app (in development)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/K1taru/Landscape-Classifier.git
cd Landscape-Classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

**Option A — Kaggle (automated):**
```bash
python src/utils/kaggle_dataset_installer.py
```
Requires a Kaggle account and `kaggle.json` API token placed in `~/.kaggle/`.

**Option B — Manual:**
Download the [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) dataset from Kaggle, extract it, and place the 6 class folders inside `src/data/raw/Datasets/`.

Expected structure:
```
src/data/raw/Datasets/
├── buildings/   (~500 images)
├── forest/      (~500 images)
├── glacier/     (~500 images)
├── mountain/    (~500 images)
├── sea/         (~500 images)
└── street/      (~500 images)
```

---

## Training

Open and run the notebook:

```
src/notebooks/landscape_classifier_model.ipynb
```

Run all cells top to bottom. After training completes, the following are saved automatically to `src/models/results/<model_name>/`:

- `<model_name>_report.md` — full training report (specs, hyperparameters, epoch table)
- `<model_name>_gradient_descent.png` — gradient descent visualization

---

## Configuration

All parameters are in **Cell 5** (Global Config) and **Cell 21** (Training Config) of the notebook.

| Parameter | Default | Description |
|---|---|---|
| `TRAIN_SPLIT` | 0.70 | Training set ratio |
| `VAL_SPLIT` | 0.15 | Validation set ratio |
| `TEST_SPLIT` | 0.15 | Test set ratio |
| `BATCH_SIZE` | 32 | Batch size |
| `USE_AUGMENTATION` | `True` | Enable/disable data augmentation |
| `LEARNING_RATE` | 1e-4 | Initial learning rate (AdamW) |
| `MAX_EPOCHS` | 30 | Maximum training epochs |
| `EARLY_STOPPING_PATIENCE` | 10 | Epochs without improvement before stopping |
| `DROPOUT_RATE` | 0.4 | Dropout in classifier head |
| `LABEL_SMOOTHING` | 0.1 | Label smoothing factor |

### Augmentation toggles (all individually configurable):

```python
AUGMENTATION_OPTIONS = {
    "random_horizontal_flip": True,
    "random_rotation":        True,
    "random_affine":          True,
    "color_jitter":           True,
    "random_perspective":     True,
    "random_resized_crop":    True,
    "gaussian_blur":          True,
    "random_erasing":         True,
}
```

---

## Model Architecture

```
ResNet50 (pretrained: IMAGENET1K_V2)
└── fc: Sequential(
    ├── Dropout(p=0.4)
    └── Linear(in=2048, out=6)
    )
Total parameters: 23,520,326
```

---

## Training Pipeline Features

- Transfer learning from ImageNet (IMAGENET1K_V2 weights)
- Weighted random sampling for class imbalance
- Weighted CrossEntropyLoss with label smoothing
- CosineAnnealingWarmRestarts scheduler
- Early stopping with best model checkpoint
- Gradient clipping (max norm = 1.0)
- Per-epoch timing and total training time
- Worst-performing class tracked each epoch
- Top-1 and Top-5 accuracy tracking
- Auto-generated training report (.md)
- Gradient descent visualization (2D + 3D)

---

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA (recommended)
- See `requirements.txt` for full list
