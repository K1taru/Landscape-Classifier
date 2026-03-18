# Datasets2 Classifier - Training Report
**Generated:** 2026-03-18 01:10:16

---

## Model Specifications
| Parameter | Value |
|---|---|
| Architecture | RESNET50 |
| Pre-trained Weights | IMAGENET1K_V2 |
| Input Size | 224 x 224 |
| Number of Classes | 5 |
| Class Names | collapsed_building, fire, flooded_areas, normal, traffic_incident |
| Total Parameters | 23,518,277 |
| Classifier Head | Dropout(0.4) -> Linear(2048, 5) |

## Dataset Information
| Parameter | Value |
|---|---|
| Dataset | Datasets2 |
| Dataset Directory | `../data/raw/Datasets2` |
| Total Samples | 6,433 |
| Training Set | 4,503 (70%) |
| Validation Set | 964 (15%) |
| Test Set | 966 (15%) |
| Batch Size | 32 |
| Weighted Sampler | Disabled |
| Random Seed | 42 |

## Hyperparameters
| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 0.0001 |
| Weight Decay | 0.0001 |
| Label Smoothing | 0.1 |
| Dropout Rate | 0.4 |
| Gradient Clipping (max_norm) | 1.0 |
| LR Scheduler | ReduceLROnPlateau |
| Scheduler factor | 0.5 |
| Scheduler patience | 3 |
| Scheduler min_lr | 1e-07 |
| Max Epochs | 50 |
| Early Stopping Patience | 8 |
| Progressive Unfreezing | Disabled |

## Data Augmentation
**Status:** Enabled

| Augmentation | Enabled |
|---|---|
| Random Horizontal Flip | Yes |
| Random Rotation (20°) | Yes |
| Random Affine | Yes |
| Color Jitter | Yes |
| Random Perspective | Yes |
| Random Resized Crop | Yes |
| Gaussian Blur | Yes |
| Random Erasing | Yes |

## Hardware Information
| Component | Details |
|---|---|
| Device | cuda |
| GPU | NVIDIA GeForce RTX 2060 |
| GPU Memory | 5.6 GB |
| CUDA Version | 12.8 |
| cuDNN Version | 91002 |
| PyTorch Version | 2.10.0+cu128 |
| Python Version | 3.12.3 |
| OS | Linux-6.17.0-19-generic-x86_64-with-glibc2.39 |

## Training Time
| Metric | Value |
|---|---|
| Total Training Time | 55m 42.2s |
| Time to Best Epoch (Saved Model) | 43m 57.7s |
| Average Epoch Time | 1m 30.3s |
| Best Epoch | 29 / 50 |
| Epochs Run | 37 |
| Early Stopping | Yes (stopped at epoch 37) |

## Epoch-by-Epoch Training Status

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Top-5 Train | Top-5 Val | Worst Class | Worst Acc | LR | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.1393 | 0.8410 | 62.54% | 84.75% | 100.00% | 100.00% | normal | 79.91% | 1.00e-04 | 1m 30.6s |
| 2 | 0.8413 | 0.7962 | 88.25% | 93.15% | 100.00% | 100.00% | normal | 91.69% | 1.00e-04 | 1m 26.7s |
| 3 | 0.8109 | 0.7911 | 92.29% | 94.09% | 100.00% | 100.00% | normal | 93.05% | 1.00e-04 | 1m 27.1s |
| 4 | 0.7719 | 0.7810 | 94.29% | 95.95% | 100.00% | 100.00% | flooded_areas | 91.67% | 1.00e-04 | 1m 24.0s |
| 5 | 0.7490 | 0.7987 | 95.67% | 94.61% | 100.00% | 100.00% | flooded_areas | 91.67% | 1.00e-04 | 1m 24.0s |
| 6 | 0.7344 | 0.7450 | 96.60% | 97.41% | 100.00% | 100.00% | normal | 96.83% | 1.00e-04 | 1m 25.8s |
| 7 | 0.7415 | 0.7492 | 96.87% | 96.89% | 100.00% | 100.00% | traffic_incident | 96.15% | 1.00e-04 | 1m 23.5s |
| 8 | 0.7386 | 0.7718 | 97.40% | 97.30% | 100.00% | 100.00% | traffic_incident | 92.31% | 1.00e-04 | 1m 25.4s |
| 9 | 0.7155 | 0.8188 | 98.05% | 95.12% | 100.00% | 100.00% | traffic_incident | 84.62% | 1.00e-04 | 1m 32.0s |
| 10 | 0.7161 | 0.7745 | 98.18% | 96.47% | 100.00% | 100.00% | collapsed_building | 90.91% | 5.00e-05 | 1m 38.3s |
| 11 | 0.7084 | 0.7752 | 98.67% | 97.41% | 100.00% | 100.00% | traffic_incident | 87.18% | 5.00e-05 | 1m 32.3s |
| 12 | 0.7001 | 0.7546 | 99.02% | 97.30% | 100.00% | 100.00% | traffic_incident | 94.87% | 5.00e-05 | 1m 34.6s |
| 13 | 0.7030 | 0.7584 | 98.98% | 97.51% | 100.00% | 100.00% | traffic_incident | 94.87% | 5.00e-05 | 1m 30.2s |
| 14 | 0.6882 | 0.7586 | 99.36% | 97.72% | 100.00% | 100.00% | traffic_incident | 93.59% | 2.50e-05 | 1m 27.7s |
| 15 | 0.6893 | 0.7634 | 99.47% | 97.41% | 100.00% | 100.00% | traffic_incident | 93.59% | 2.50e-05 | 1m 30.8s |
| 16 | 0.6951 | 0.7588 | 99.62% | 97.61% | 100.00% | 100.00% | traffic_incident | 93.59% | 2.50e-05 | 1m 29.9s |
| 17 | 0.6899 | 0.7594 | 99.67% | 97.93% | 100.00% | 100.00% | traffic_incident | 93.59% | 2.50e-05 | 1m 34.4s |
| 18 | 0.6884 | 0.7587 | 99.64% | 97.82% | 100.00% | 100.00% | traffic_incident | 91.03% | 1.25e-05 | 1m 26.5s |
| 19 | 0.6948 | 0.7523 | 99.64% | 97.82% | 100.00% | 100.00% | flooded_areas | 95.83% | 1.25e-05 | 1m 32.4s |
| 20 | 0.6824 | 0.7534 | 99.78% | 98.03% | 100.00% | 100.00% | traffic_incident | 94.87% | 1.25e-05 | 1m 30.7s |
| 21 | 0.6815 | 0.7545 | 99.73% | 98.03% | 100.00% | 100.00% | traffic_incident | 93.59% | 1.25e-05 | 1m 27.1s |
| 22 | 0.6897 | 0.7589 | 99.71% | 98.13% | 100.00% | 100.00% | traffic_incident | 93.59% | 6.25e-06 | 1m 42.7s |
| 23 | 0.6818 | 0.7495 | 99.78% | 97.93% | 100.00% | 100.00% | traffic_incident | 94.87% | 6.25e-06 | 1m 48.1s |
| 24 | 0.6855 | 0.7535 | 99.73% | 98.03% | 100.00% | 100.00% | traffic_incident | 94.87% | 6.25e-06 | 1m 36.3s |
| 25 | 0.6856 | 0.7547 | 99.84% | 97.93% | 100.00% | 100.00% | traffic_incident | 93.59% | 6.25e-06 | 1m 31.4s |
| 26 | 0.6870 | 0.7544 | 99.82% | 98.03% | 100.00% | 100.00% | traffic_incident | 93.59% | 3.13e-06 | 1m 35.1s |
| 27 | 0.6861 | 0.7559 | 99.89% | 98.03% | 100.00% | 100.00% | traffic_incident | 93.59% | 3.13e-06 | 1m 38.9s |
| 28 | 0.6779 | 0.7538 | 99.91% | 98.03% | 100.00% | 100.00% | traffic_incident | 94.87% | 3.13e-06 | 1m 25.1s |
| **29** ⭐ | 0.6892 | 0.7519 | 99.71% | 98.24% | 100.00% | 100.00% | flooded_areas | 95.83% | 3.13e-06 | 1m 26.0s |
| 30 | 0.6910 | 0.7545 | 99.84% | 97.93% | 100.00% | 100.00% | traffic_incident | 94.87% | 1.56e-06 | 1m 30.8s |
| 31 | 0.6800 | 0.7552 | 99.89% | 97.93% | 100.00% | 100.00% | traffic_incident | 94.87% | 1.56e-06 | 1m 27.7s |
| 32 | 0.6837 | 0.7532 | 99.89% | 98.03% | 100.00% | 100.00% | traffic_incident | 93.59% | 1.56e-06 | 1m 27.4s |
| 33 | 0.6866 | 0.7550 | 99.82% | 98.13% | 100.00% | 100.00% | flooded_areas | 95.83% | 1.56e-06 | 1m 27.6s |
| 34 | 0.6852 | 0.7543 | 99.78% | 97.72% | 100.00% | 100.00% | traffic_incident | 93.59% | 7.81e-07 | 1m 27.5s |
| 35 | 0.6821 | 0.7525 | 99.84% | 98.03% | 100.00% | 100.00% | traffic_incident | 94.87% | 7.81e-07 | 1m 27.9s |
| 36 | 0.6835 | 0.7541 | 99.82% | 98.13% | 100.00% | 100.00% | traffic_incident | 94.87% | 7.81e-07 | 1m 27.9s |
| **37** (Last) | 0.6852 | 0.7553 | 99.82% | 97.93% | 100.00% | 100.00% | traffic_incident | 93.59% | 7.81e-07 | 1m 27.5s |

## Training Results Summary
| Metric | Value |
|---|---|
| Best Validation Accuracy | 98.24% |
| Best Epoch | 29 |
| Final Training Accuracy | 99.82% |
| Final Validation Accuracy | 97.93% |
| Final Training Loss | 0.6852 |
| Final Validation Loss | 0.7553 |
| Best Train Top-5 Accuracy | 100.00% |
| Best Val Top-5 Accuracy | 100.00% |


================================================================================
                         🎯 MODEL EVALUATION SUMMARY
================================================================================

📦 Model Architecture: RESNET50
📊 Dataset: Datasets2
🔢 Number of Classes: 5 (collapsed_building, fire, flooded_areas, normal, traffic_incident)
🧪 Test Set Size: 966 images

────────────────────────────────────────────────────────────────────────────────
📈 OVERALL PERFORMANCE METRICS:
────────────────────────────────────────────────────────────────────────────────
  Overall Accuracy:         98.24%
  Balanced Accuracy:        96.75%
  Macro Precision:          96.92%
  Macro Recall:             96.75%
  Macro F1-Score:           96.80%
  Weighted Precision:       98.26%
  Weighted Recall:          98.24%
  Weighted F1-Score:        98.24%

────────────────────────────────────────────────────────────────────────────────
✅ BEST PERFORMING CLASSES (Top-3):
────────────────────────────────────────────────────────────────────────────────
  1. fire: F1=100.00%, Accuracy=100.00%, Support=81
  2. normal: F1= 99.08%, Accuracy= 99.08%, Support=653
  3. flooded_areas: F1= 98.88%, Accuracy=100.00%, Support=88

────────────────────────────────────────────────────────────────────────────────
⚠️  WORST PERFORMING CLASSES (Bottom-3):
────────────────────────────────────────────────────────────────────────────────
  1. flooded_areas: F1= 98.88%, Accuracy=100.00%, Support=88
  2. collapsed_building: F1= 93.43%, Accuracy= 90.14%, Support=71
  3. traffic_incident: F1= 92.62%, Accuracy= 94.52%, Support=73

────────────────────────────────────────────────────────────────────────────────
🔄 CONFUSION INSIGHTS:
────────────────────────────────────────────────────────────────────────────────
  Most confused pair: normal → traffic_incident (5 misclassifications)
  Total misclassifications: 17
  Error rate: 1.76%

────────────────────────────────────────────────────────────────────────────────
🎲 CONFIDENCE STATISTICS:
────────────────────────────────────────────────────────────────────────────────
  Avg confidence (correct):    68.16%
  Avg confidence (incorrect):  65.28%
  Confidence gap:               2.88%

────────────────────────────────────────────────────────────────────────────────
💾 MODEL ARTIFACTS:
────────────────────────────────────────────────────────────────────────────────
  Saved model: Datasets2_resnet50_E29_VAL98.24.pth
  Training history: Stored in 'history' dictionary
  Metrics tracked: Top-1 Acc, Top-5 Acc, Loss, Learning Rate

────────────────────────────────────────────────────────────────────────────────
🚀 TRAINING CONFIGURATION:
────────────────────────────────────────────────────────────────────────────────
  Optimizer: AdamW (lr=0.0001, weight_decay=0.0001)
  Scheduler: ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-07)
  Batch size: 32
  Max epochs: 50
  Early stopping patience: 8
  Dropout rate: 0.4
  Gradient clipping: max_norm=1.0
  Train/Val/Test split: 70%/15%/15%
  Augmentation: ENABLED
  Weighted Sampler: DISABLED



## Saved Model
- **Filename:** `Datasets2_resnet50_E29_VAL98.24.pth`
- **Path:** `../models/Datasets2_resnet50_E29_VAL98.24.pth`

---
*Report generated automatically by `training_report.py`*