# Landscape (Intel Image Classification) Classifier - Training Report
**Generated:** 2026-03-16 21:55:03

---

## Model Specifications
| Parameter | Value |
|---|---|
| Architecture | RESNET50 |
| Pre-trained Weights | IMAGENET1K_V2 |
| Input Size | 224 x 224 |
| Number of Classes | 6 |
| Class Names | buildings, forest, glacier, mountain, sea, street |
| Total Parameters | 23,520,326 |
| Classifier Head | Dropout(0.4) -> Linear(2048, 6) |

## Dataset Information
| Parameter | Value |
|---|---|
| Dataset | Landscape (Intel Image Classification) |
| Dataset Directory | `../data/raw/Datasets` |
| Total Samples | 3,000 |
| Training Set | 2,100 (70%) |
| Validation Set | 450 (15%) |
| Test Set | 450 (15%) |
| Batch Size | 32 |
| Weighted Sampler | Enabled |
| Random Seed | 42 |

## Hyperparameters
| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 3e-05 |
| Weight Decay | 0.0001 |
| Label Smoothing | 0.1 |
| Dropout Rate | 0.4 |
| Gradient Clipping (max_norm) | 1.0 |
| LR Scheduler | CosineAnnealingWarmRestarts |
| Scheduler T_0 | 15 |
| Scheduler T_mult | 2 |
| Scheduler eta_min | 1e-07 |
| Max Epochs | 40 |
| Early Stopping Patience | 8 |

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
| Total Training Time | 16m 15.3s |
| Time to Best Epoch (Saved Model) | 11m 25.5s |
| Average Epoch Time | 37.5s |
| Best Epoch | 18 / 40 |
| Epochs Run | 26 |
| Early Stopping | Yes (stopped at epoch 26) |

## Epoch-by-Epoch Training Status

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Top-5 Train | Top-5 Val | Worst Class | Worst Acc | LR | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.5939 | 1.2640 | 47.57% | 80.89% | 96.19% | 100.00% | mountain | 47.76% | 2.97e-05 | 47.6s |
| 2 | 1.0668 | 0.7477 | 78.90% | 92.00% | 99.52% | 100.00% | mountain | 77.61% | 2.87e-05 | 36.8s |
| 3 | 0.7411 | 0.6176 | 88.10% | 91.78% | 99.71% | 100.00% | mountain | 73.13% | 2.71e-05 | 40.1s |
| 4 | 0.6689 | 0.6003 | 89.19% | 91.78% | 99.81% | 100.00% | mountain | 77.61% | 2.51e-05 | 37.1s |
| 5 | 0.6390 | 0.5946 | 90.48% | 92.00% | 99.95% | 100.00% | mountain | 71.64% | 2.25e-05 | 35.9s |
| 6 | 0.6169 | 0.5747 | 92.38% | 93.56% | 99.95% | 100.00% | mountain | 79.10% | 1.97e-05 | 38.0s |
| 7 | 0.6184 | 0.5756 | 90.81% | 93.11% | 99.90% | 100.00% | mountain | 83.58% | 1.66e-05 | 35.0s |
| 8 | 0.5786 | 0.5672 | 93.33% | 93.56% | 100.00% | 100.00% | mountain | 80.60% | 1.35e-05 | 49.3s |
| 9 | 0.5762 | 0.5607 | 93.24% | 93.33% | 99.95% | 100.00% | mountain | 82.09% | 1.04e-05 | 40.3s |
| 10 | 0.5589 | 0.5615 | 94.29% | 94.00% | 99.95% | 100.00% | mountain | 82.09% | 7.58e-06 | 35.6s |
| 11 | 0.5785 | 0.5550 | 92.86% | 93.56% | 100.00% | 100.00% | mountain | 79.10% | 5.05e-06 | 35.3s |
| 12 | 0.5789 | 0.5582 | 93.33% | 93.56% | 100.00% | 100.00% | mountain | 77.61% | 2.96e-06 | 35.4s |
| 13 | 0.5605 | 0.5541 | 94.10% | 93.56% | 99.90% | 100.00% | mountain | 80.60% | 1.39e-06 | 35.1s |
| 14 | 0.5789 | 0.5554 | 93.00% | 93.33% | 99.95% | 100.00% | mountain | 82.09% | 4.27e-07 | 36.9s |
| 15 | 0.5649 | 0.5555 | 94.24% | 93.33% | 100.00% | 100.00% | mountain | 82.09% | 3.00e-05 | 36.1s |
| 16 | 0.5564 | 0.5581 | 94.05% | 92.89% | 100.00% | 100.00% | mountain | 77.61% | 2.99e-05 | 36.3s |
| 17 | 0.5432 | 0.5585 | 94.81% | 94.00% | 100.00% | 100.00% | mountain | 79.10% | 2.97e-05 | 36.6s |
| **18** ⭐ | 0.5615 | 0.5503 | 93.52% | 95.11% | 100.00% | 100.00% | mountain | 86.57% | 2.93e-05 | 38.0s |
| 19 | 0.5362 | 0.5514 | 95.14% | 94.00% | 99.95% | 100.00% | mountain | 85.07% | 2.87e-05 | 36.2s |
| 20 | 0.5279 | 0.5631 | 95.19% | 93.33% | 100.00% | 100.00% | mountain | 77.61% | 2.80e-05 | 36.8s |
| 21 | 0.5235 | 0.5571 | 95.48% | 94.44% | 100.00% | 100.00% | mountain | 80.60% | 2.71e-05 | 36.4s |
| 22 | 0.5123 | 0.5639 | 96.43% | 93.56% | 100.00% | 99.78% | mountain | 80.60% | 2.62e-05 | 36.0s |
| 23 | 0.5004 | 0.5479 | 96.86% | 94.00% | 100.00% | 100.00% | mountain | 83.58% | 2.51e-05 | 36.0s |
| 24 | 0.4946 | 0.5505 | 97.00% | 94.22% | 100.00% | 99.78% | mountain | 85.07% | 2.38e-05 | 35.9s |
| 25 | 0.4962 | 0.5662 | 97.10% | 93.33% | 100.00% | 99.78% | mountain | 82.09% | 2.25e-05 | 36.3s |
| **26** (Last) | 0.4797 | 0.5593 | 98.10% | 94.44% | 100.00% | 99.78% | mountain | 85.07% | 2.11e-05 | 36.3s |

## Training Results Summary
| Metric | Value |
|---|---|
| Best Validation Accuracy | 95.11% |
| Best Epoch | 18 |
| Final Training Accuracy | 98.10% |
| Final Validation Accuracy | 94.44% |
| Final Training Loss | 0.4797 |
| Final Validation Loss | 0.5593 |
| Best Train Top-5 Accuracy | 100.00% |
| Best Val Top-5 Accuracy | 100.00% |

## Saved Model
- **Filename:** `Landscape_resnet50_E18_VAL95.11.pth`
- **Path:** `../models/Landscape_resnet50_E18_VAL95.11.pth`

---
*Report generated automatically by `training_report.py`*