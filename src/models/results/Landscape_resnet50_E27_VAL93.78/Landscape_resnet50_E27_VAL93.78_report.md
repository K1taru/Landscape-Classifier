# Landscape (Intel Image Classification) Classifier - Training Report
**Generated:** 2026-03-16 07:32:15

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
| Batch Size | 24 |
| Weighted Sampler | Enabled |
| Random Seed | 42 |

## Hyperparameters
| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 1e-05 |
| Weight Decay | 0.0001 |
| Label Smoothing | 0.1 |
| Dropout Rate | 0.4 |
| Gradient Clipping (max_norm) | 1.0 |
| LR Scheduler | CosineAnnealingWarmRestarts |
| Scheduler T_0 | 12 |
| Scheduler T_mult | 2 |
| Scheduler eta_min | 1e-08 |
| Max Epochs | 30 |
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
| GPU Memory | 6.0 GB |
| CUDA Version | 12.4 |
| cuDNN Version | 90100 |
| PyTorch Version | 2.6.0+cu124 |
| Python Version | 3.12.12 |
| OS | Windows-11-10.0.26200-SP0 |

## Training Time
| Metric | Value |
|---|---|
| Total Training Time | 31m 7.0s |
| Time to Best Epoch (Saved Model) | 27m 57.9s |
| Average Epoch Time | 1m 2.2s |
| Best Epoch | 27 / 30 |
| Epochs Run | 30 |
| Early Stopping | No (ran all epochs) |

## Epoch-by-Epoch Training Status

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Top-5 Train | Top-5 Val | Worst Class | Worst Acc | LR | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.7229 | 1.6521 | 31.90% | 54.67% | 93.38% | 99.56% | mountain | 25.37% | 9.83e-06 | 1m 1.2s |
| 2 | 1.5589 | 1.4557 | 55.10% | 72.89% | 97.90% | 100.00% | mountain | 31.34% | 9.33e-06 | 1m 1.9s |
| 3 | 1.3218 | 1.1744 | 69.90% | 85.33% | 99.29% | 100.00% | mountain | 67.16% | 8.54e-06 | 1m 6.7s |
| 4 | 1.1232 | 0.9575 | 76.24% | 88.22% | 99.48% | 100.00% | mountain | 65.67% | 7.50e-06 | 1m 3.3s |
| 5 | 0.9929 | 0.8223 | 80.19% | 90.44% | 99.71% | 100.00% | mountain | 70.15% | 6.30e-06 | 1m 2.4s |
| 6 | 0.8702 | 0.7320 | 84.00% | 90.89% | 99.71% | 100.00% | mountain | 74.63% | 5.01e-06 | 1m 1.3s |
| 7 | 0.8241 | 0.6866 | 84.29% | 91.33% | 99.67% | 100.00% | mountain | 76.12% | 3.71e-06 | 1m 4.0s |
| 8 | 0.7728 | 0.6600 | 86.67% | 91.33% | 99.95% | 100.00% | mountain | 73.13% | 2.51e-06 | 1m 3.6s |
| 9 | 0.7633 | 0.6520 | 84.90% | 91.11% | 99.81% | 100.00% | mountain | 77.61% | 1.47e-06 | 1m 3.8s |
| 10 | 0.7350 | 0.6458 | 87.00% | 90.44% | 99.76% | 100.00% | mountain | 76.12% | 6.79e-07 | 1m 2.1s |
| 11 | 0.7481 | 0.6431 | 86.90% | 90.89% | 99.81% | 100.00% | mountain | 73.13% | 1.80e-07 | 1m 4.8s |
| 12 | 0.7524 | 0.6383 | 85.81% | 92.22% | 99.90% | 100.00% | mountain | 73.13% | 1.00e-05 | 1m 4.7s |
| 13 | 0.7220 | 0.6212 | 87.81% | 92.44% | 99.86% | 100.00% | mountain | 77.61% | 9.96e-06 | 1m 0.4s |
| 14 | 0.7137 | 0.6070 | 86.24% | 92.89% | 99.86% | 100.00% | mountain | 77.61% | 9.83e-06 | 1m 2.2s |
| 15 | 0.6751 | 0.6067 | 89.29% | 92.44% | 99.95% | 100.00% | mountain | 77.61% | 9.62e-06 | 1m 2.8s |
| 16 | 0.6657 | 0.5939 | 89.19% | 92.67% | 99.76% | 100.00% | mountain | 79.10% | 9.33e-06 | 1m 1.8s |
| 17 | 0.6416 | 0.5861 | 90.76% | 92.89% | 99.81% | 100.00% | mountain | 76.12% | 8.97e-06 | 1m 4.2s |
| 18 | 0.6502 | 0.5814 | 89.90% | 92.89% | 99.95% | 100.00% | mountain | 79.10% | 8.54e-06 | 59.1s |
| 19 | 0.6436 | 0.5797 | 89.90% | 93.11% | 99.76% | 99.78% | mountain | 77.61% | 8.05e-06 | 1m 2.9s |
| 20 | 0.6307 | 0.5851 | 91.19% | 92.67% | 99.95% | 100.00% | mountain | 77.61% | 7.50e-06 | 1m 0.8s |
| 21 | 0.6225 | 0.5719 | 90.95% | 92.67% | 99.95% | 100.00% | mountain | 79.10% | 6.92e-06 | 59.6s |
| 22 | 0.6170 | 0.5734 | 90.90% | 92.89% | 99.90% | 100.00% | mountain | 79.10% | 6.30e-06 | 59.6s |
| 23 | 0.6126 | 0.5736 | 90.90% | 93.56% | 99.86% | 100.00% | mountain | 79.10% | 5.66e-06 | 60.0s |
| 24 | 0.5928 | 0.5655 | 92.10% | 93.33% | 99.95% | 100.00% | mountain | 79.10% | 5.01e-06 | 1m 0.1s |
| 25 | 0.6040 | 0.5672 | 92.38% | 93.56% | 99.95% | 100.00% | mountain | 77.61% | 4.35e-06 | 1m 1.5s |
| 26 | 0.5904 | 0.5663 | 92.29% | 93.11% | 99.95% | 100.00% | mountain | 80.60% | 3.71e-06 | 1m 1.9s |
| **27** ⭐ | 0.6032 | 0.5644 | 91.52% | 93.78% | 99.95% | 100.00% | mountain | 80.60% | 3.09e-06 | 1m 1.2s |
| 28 | 0.5780 | 0.5598 | 93.76% | 93.33% | 99.95% | 100.00% | mountain | 80.60% | 2.51e-06 | 1m 0.1s |
| 29 | 0.5738 | 0.5660 | 93.33% | 93.56% | 99.95% | 100.00% | mountain | 80.60% | 1.96e-06 | 1m 4.3s |
| **30** (Last) | 0.5650 | 0.5678 | 94.10% | 93.56% | 99.95% | 100.00% | mountain | 80.60% | 1.47e-06 | 1m 3.4s |

## Training Results Summary
| Metric | Value |
|---|---|
| Best Validation Accuracy | 93.78% |
| Best Epoch | 27 |
| Final Training Accuracy | 94.10% |
| Final Validation Accuracy | 93.56% |
| Final Training Loss | 0.5650 |
| Final Validation Loss | 0.5678 |
| Best Train Top-5 Accuracy | 99.95% |
| Best Val Top-5 Accuracy | 100.00% |

## Saved Model
- **Filename:** `Landscape_resnet50_E27_VAL93.78.pth`
- **Path:** `../models\Landscape_resnet50_E27_VAL93.78.pth`

---
*Report generated automatically by `training_report.py`*