# Landscape (Intel Image Classification) Classifier - Training Report
**Generated:** 2026-03-16 05:10:57

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
| Learning Rate | 0.0001 |
| Weight Decay | 0.0001 |
| Label Smoothing | 0.1 |
| Dropout Rate | 0.4 |
| Gradient Clipping (max_norm) | 1.0 |
| LR Scheduler | CosineAnnealingWarmRestarts |
| Scheduler T_0 | 10 |
| Scheduler T_mult | 2 |
| Scheduler eta_min | 1e-07 |
| Max Epochs | 30 |
| Early Stopping Patience | 10 |

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
| Total Training Time | 17m 57.7s |
| Time to Best Epoch (Saved Model) | 7m 57.1s |
| Average Epoch Time | 59.9s |
| Best Epoch | 8 / 30 |
| Epochs Run | 18 |
| Early Stopping | Yes (stopped at epoch 18) |

## Epoch-by-Epoch Training Status

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Top-5 Train | Top-5 Val | Worst Class | Worst Acc | LR | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.1582 | 0.6429 | 69.43% | 90.22% | 98.24% | 100.00% | mountain | 68.66% | 9.76e-05 | 1m 0.5s |
| 2 | 0.6619 | 0.5869 | 89.62% | 92.89% | 99.62% | 100.00% | mountain | 76.12% | 9.05e-05 | 59.4s |
| 3 | 0.6114 | 0.5892 | 91.86% | 92.44% | 99.95% | 100.00% | mountain | 64.18% | 7.94e-05 | 59.3s |
| 4 | 0.5843 | 0.5708 | 92.48% | 92.67% | 99.90% | 100.00% | mountain | 79.10% | 6.55e-05 | 59.9s |
| 5 | 0.5566 | 0.5558 | 94.43% | 94.00% | 99.90% | 100.00% | mountain | 82.09% | 5.01e-05 | 59.6s |
| 6 | 0.5353 | 0.5561 | 94.81% | 94.22% | 99.95% | 100.00% | mountain | 77.61% | 3.46e-05 | 59.5s |
| 7 | 0.5440 | 0.5450 | 94.86% | 93.78% | 100.00% | 100.00% | glacier | 84.42% | 2.07e-05 | 59.3s |
| **8** ⭐ | 0.5049 | 0.5437 | 96.76% | 95.78% | 100.00% | 100.00% | mountain | 82.09% | 9.64e-06 | 59.6s |
| 9 | 0.5066 | 0.5410 | 96.48% | 95.11% | 100.00% | 100.00% | mountain | 82.09% | 2.54e-06 | 59.4s |
| 10 | 0.4942 | 0.5341 | 96.86% | 95.33% | 100.00% | 100.00% | mountain | 82.09% | 1.00e-04 | 1m 0.3s |
| 11 | 0.5157 | 0.5953 | 96.29% | 92.67% | 100.00% | 99.78% | mountain | 82.09% | 9.94e-05 | 58.9s |
| 12 | 0.5316 | 0.5881 | 95.14% | 93.33% | 99.95% | 99.78% | mountain | 80.60% | 9.76e-05 | 59.1s |
| 13 | 0.5051 | 0.5952 | 96.29% | 92.67% | 100.00% | 99.78% | mountain | 79.10% | 9.46e-05 | 59.2s |
| 14 | 0.4990 | 0.5839 | 96.86% | 93.33% | 100.00% | 99.78% | mountain | 85.07% | 9.05e-05 | 59.6s |
| 15 | 0.4920 | 0.6027 | 97.14% | 92.22% | 100.00% | 99.56% | mountain | 74.63% | 8.54e-05 | 1m 3.3s |
| 16 | 0.4828 | 0.5558 | 97.33% | 94.44% | 100.00% | 99.78% | mountain | 85.07% | 7.94e-05 | 1m 0.5s |
| 17 | 0.4627 | 0.5814 | 98.48% | 92.67% | 100.00% | 99.33% | mountain | 80.60% | 7.27e-05 | 1m 0.0s |
| **18** (Last) | 0.4735 | 0.5810 | 97.95% | 93.78% | 99.95% | 99.56% | mountain | 82.09% | 6.55e-05 | 59.9s |

## Training Results Summary
| Metric | Value |
|---|---|
| Best Validation Accuracy | 95.78% |
| Best Epoch | 8 |
| Final Training Accuracy | 97.95% |
| Final Validation Accuracy | 93.78% |
| Final Training Loss | 0.4735 |
| Final Validation Loss | 0.5810 |
| Best Train Top-5 Accuracy | 100.00% |
| Best Val Top-5 Accuracy | 100.00% |

## Saved Model
- **Filename:** `Landscape_resnet50_E8_VAL95.78.pth`
- **Path:** `../models\Landscape_resnet50_E8_VAL95.78.pth`

---
*Report generated automatically by `training_report.py`*