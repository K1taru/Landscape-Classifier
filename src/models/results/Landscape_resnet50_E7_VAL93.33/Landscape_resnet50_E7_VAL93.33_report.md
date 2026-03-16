# Landscape (Intel Image Classification) Classifier - Training Report
**Generated:** 2026-03-16 06:43:42

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
| Training Set | 2,250 (75%) |
| Validation Set | 390 (13%) |
| Test Set | 360 (12%) |
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
| Total Training Time | 18m 56.0s |
| Time to Best Epoch (Saved Model) | 7m 45.6s |
| Average Epoch Time | 1m 6.8s |
| Best Epoch | 7 / 30 |
| Epochs Run | 17 |
| Early Stopping | Yes (stopped at epoch 17) |

## Epoch-by-Epoch Training Status

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Top-5 Train | Top-5 Val | Worst Class | Worst Acc | LR | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.1189 | 0.6621 | 69.51% | 90.00% | 97.96% | 100.00% | glacier | 69.23% | 9.76e-05 | 1m 8.7s |
| 2 | 0.6760 | 0.6204 | 88.40% | 89.49% | 99.82% | 100.00% | glacier | 73.85% | 9.05e-05 | 1m 6.4s |
| 3 | 0.6034 | 0.5980 | 91.82% | 92.82% | 99.87% | 100.00% | mountain | 78.57% | 7.94e-05 | 1m 7.1s |
| 4 | 0.5835 | 0.5926 | 92.62% | 92.56% | 99.96% | 99.74% | mountain | 82.86% | 6.55e-05 | 1m 5.3s |
| 5 | 0.5623 | 0.5904 | 94.27% | 92.05% | 99.96% | 100.00% | mountain | 80.00% | 5.01e-05 | 1m 5.4s |
| 6 | 0.5347 | 0.5791 | 95.38% | 92.31% | 100.00% | 99.74% | mountain | 82.86% | 3.46e-05 | 1m 5.2s |
| **7** ⭐ | 0.5200 | 0.5698 | 96.09% | 93.33% | 100.00% | 100.00% | mountain | 82.86% | 2.07e-05 | 1m 7.6s |
| 8 | 0.5060 | 0.5701 | 96.49% | 93.33% | 100.00% | 100.00% | mountain | 84.29% | 9.64e-06 | 1m 8.5s |
| 9 | 0.5098 | 0.5818 | 96.09% | 92.82% | 100.00% | 100.00% | mountain | 81.43% | 2.54e-06 | 1m 6.6s |
| 10 | 0.5103 | 0.5786 | 96.31% | 92.31% | 100.00% | 100.00% | mountain | 81.43% | 1.00e-04 | 1m 5.8s |
| 11 | 0.5107 | 0.6165 | 96.36% | 90.77% | 100.00% | 99.74% | glacier | 73.85% | 9.94e-05 | 1m 5.6s |
| 12 | 0.5194 | 0.6187 | 96.00% | 92.56% | 99.91% | 100.00% | mountain | 77.14% | 9.76e-05 | 1m 6.6s |
| 13 | 0.5070 | 0.5966 | 96.58% | 92.05% | 100.00% | 99.74% | glacier | 75.38% | 9.46e-05 | 1m 7.5s |
| 14 | 0.5033 | 0.5765 | 96.49% | 92.82% | 100.00% | 100.00% | glacier | 84.62% | 9.05e-05 | 1m 8.2s |
| 15 | 0.4977 | 0.6155 | 96.62% | 91.03% | 100.00% | 99.74% | mountain | 81.43% | 8.54e-05 | 1m 7.3s |
| 16 | 0.4839 | 0.6029 | 97.64% | 93.08% | 100.00% | 99.49% | mountain | 81.43% | 7.94e-05 | 1m 9.3s |
| **17** (Last) | 0.4737 | 0.5928 | 97.91% | 93.08% | 99.96% | 99.74% | mountain | 80.00% | 7.27e-05 | 1m 4.7s |

## Training Results Summary
| Metric | Value |
|---|---|
| Best Validation Accuracy | 93.33% |
| Best Epoch | 7 |
| Final Training Accuracy | 97.91% |
| Final Validation Accuracy | 93.08% |
| Final Training Loss | 0.4737 |
| Final Validation Loss | 0.5928 |
| Best Train Top-5 Accuracy | 100.00% |
| Best Val Top-5 Accuracy | 100.00% |

## Saved Model
- **Filename:** `Landscape_resnet50_E7_VAL93.33.pth`
- **Path:** `../models\Landscape_resnet50_E7_VAL93.33.pth`

---
*Report generated automatically by `training_report.py`*