# Datasets2 Classifier - Training Report
**Generated:** 2026-03-20 15:07:30

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
| Batch Size | 8 |
| Weighted Sampler | Disabled |
| Random Seed | 42 |

## Hyperparameters
| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 2e-05 |
| Weight Decay | 0.0001 |
| Label Smoothing | 0.08 |
| Dropout Rate | 0.4 |
| Gradient Clipping (max_norm) | 1.0 |
| LR Scheduler | CosineAnnealingLR |
| Scheduler T_max | 50 |
| Scheduler eta_min | 5e-08 |
| Max Epochs | 40 |
| Early Stopping Patience | 6 |
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
| Total Training Time | 1h 15m 51.1s |
| Time to Best Epoch (Saved Model) | 1h 5m 15.5s |
| Average Epoch Time | 4m 12.8s |
| Best Epoch | 12 / 40 |
| Epochs Run | 18 |
| Early Stopping | Yes (stopped at epoch 18) |

## Epoch-by-Epoch Training Status

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Top-5 Train | Top-5 Val | Worst Class | Worst Acc | LR | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.4532 | 0.9168 | 52.74% | 82.99% | 100.00% | 100.00% | normal | 77.64% | 2.00e-05 | 9m 45.6s |
| 2 | 0.9880 | 0.8071 | 80.81% | 90.56% | 100.00% | 100.00% | normal | 88.22% | 1.99e-05 | 10m 45.5s |
| 3 | 0.9158 | 0.7895 | 86.94% | 94.81% | 100.00% | 100.00% | collapsed_building | 92.42% | 1.98e-05 | 9m 58.4s |
| 4 | 0.8594 | 0.7721 | 90.52% | 96.47% | 100.00% | 100.00% | collapsed_building | 93.94% | 1.97e-05 | 3m 0.0s |
| 5 | 0.8390 | 0.7523 | 92.21% | 96.47% | 100.00% | 100.00% | traffic_incident | 93.59% | 1.95e-05 | 9m 6.8s |
| 6 | 0.7939 | 0.7575 | 94.03% | 96.68% | 100.00% | 100.00% | collapsed_building | 92.42% | 1.93e-05 | 8m 2.0s |
| 7 | 0.7857 | 0.7419 | 94.56% | 96.89% | 100.00% | 100.00% | traffic_incident | 93.59% | 1.91e-05 | 2m 13.9s |
| 8 | 0.7688 | 0.7632 | 95.51% | 97.10% | 100.00% | 100.00% | traffic_incident | 91.03% | 1.88e-05 | 3m 39.8s |
| 9 | 0.7575 | 0.7427 | 95.91% | 97.10% | 100.00% | 100.00% | flooded_areas | 93.06% | 1.84e-05 | 3m 40.9s |
| 10 | 0.7436 | 0.7504 | 96.38% | 97.51% | 100.00% | 100.00% | collapsed_building | 92.42% | 1.81e-05 | 1m 38.9s |
| 11 | 0.7457 | 0.7460 | 96.34% | 97.61% | 100.00% | 100.00% | flooded_areas | 91.67% | 1.77e-05 | 1m 39.4s |
| **12** ⭐ | 0.7304 | 0.7382 | 97.02% | 97.82% | 100.00% | 100.00% | flooded_areas | 94.44% | 1.73e-05 | 1m 44.4s |
| 13 | 0.7306 | 0.7483 | 97.11% | 97.51% | 100.00% | 100.00% | traffic_incident | 91.03% | 1.69e-05 | 1m 45.6s |
| 14 | 0.7142 | 0.7324 | 97.56% | 97.30% | 100.00% | 100.00% | flooded_areas | 94.44% | 1.64e-05 | 1m 45.8s |
| 15 | 0.7158 | 0.7461 | 97.73% | 97.10% | 100.00% | 100.00% | traffic_incident | 91.03% | 1.59e-05 | 1m 46.0s |
| 16 | 0.7157 | 0.7418 | 97.82% | 97.20% | 100.00% | 100.00% | collapsed_building | 93.94% | 1.54e-05 | 1m 46.2s |
| 17 | 0.7037 | 0.7442 | 98.09% | 97.30% | 100.00% | 100.00% | collapsed_building | 92.42% | 1.48e-05 | 1m 45.6s |
| **18** (Last) | 0.7031 | 0.7502 | 98.33% | 97.41% | 100.00% | 100.00% | traffic_incident | 92.31% | 1.43e-05 | 1m 46.0s |

## Training Results Summary
| Metric | Value |
|---|---|
| Best Validation Accuracy | 97.82% |
| Best Epoch | 12 |
| Final Training Accuracy | 98.33% |
| Final Validation Accuracy | 97.41% |
| Final Training Loss | 0.7031 |
| Final Validation Loss | 0.7502 |
| Best Train Top-5 Accuracy | 100.00% |
| Best Val Top-5 Accuracy | 100.00% |

## Saved Model
- **Filename:** `Datasets2_resnet50_E12_VAL97.82.pth`
- **Path:** `../models/Datasets2_resnet50_E12_VAL97.82.pth`

---
*Report generated automatically by `training_report.py`*