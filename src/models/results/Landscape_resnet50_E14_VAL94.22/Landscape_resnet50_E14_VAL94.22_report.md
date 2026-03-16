# Landscape (Intel Image Classification) Classifier - Training Report
**Generated:** 2026-03-16 06:58:56

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
| Learning Rate | 5e-05 |
| Weight Decay | 0.0001 |
| Label Smoothing | 0.1 |
| Dropout Rate | 0.4 |
| Gradient Clipping (max_norm) | 1.0 |
| LR Scheduler | CosineAnnealingWarmRestarts |
| Scheduler T_0 | 10 |
| Scheduler T_mult | 2 |
| Scheduler eta_min | 1e-07 |
| Max Epochs | 30 |
| Early Stopping Patience | 8 |

## Data Augmentation
**Status:** Disabled

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
| Total Training Time | 12m 14.9s |
| Time to Best Epoch (Saved Model) | 7m 44.8s |
| Average Epoch Time | 33.4s |
| Best Epoch | 14 / 30 |
| Epochs Run | 22 |
| Early Stopping | Yes (stopped at epoch 22) |

## Epoch-by-Epoch Training Status

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Top-5 Train | Top-5 Val | Worst Class | Worst Acc | LR | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.3220 | 0.8171 | 66.33% | 86.22% | 97.71% | 100.00% | mountain | 61.19% | 4.88e-05 | 35.4s |
| 2 | 0.6788 | 0.6086 | 91.29% | 92.67% | 99.95% | 100.00% | mountain | 76.12% | 4.52e-05 | 32.5s |
| 3 | 0.5442 | 0.6087 | 95.95% | 92.44% | 100.00% | 100.00% | mountain | 74.63% | 3.97e-05 | 33.0s |
| 4 | 0.5123 | 0.5770 | 96.81% | 93.11% | 99.95% | 100.00% | mountain | 80.60% | 3.28e-05 | 32.6s |
| 5 | 0.4812 | 0.6014 | 98.38% | 92.89% | 100.00% | 100.00% | mountain | 74.63% | 2.50e-05 | 31.9s |
| 6 | 0.4661 | 0.5901 | 99.33% | 92.67% | 100.00% | 100.00% | mountain | 74.63% | 1.73e-05 | 31.9s |
| 7 | 0.4589 | 0.5826 | 99.38% | 92.89% | 100.00% | 100.00% | mountain | 76.12% | 1.04e-05 | 34.3s |
| 8 | 0.4525 | 0.5804 | 99.67% | 92.89% | 100.00% | 100.00% | mountain | 79.10% | 4.87e-06 | 32.2s |
| 9 | 0.4528 | 0.5877 | 99.52% | 92.89% | 100.00% | 100.00% | mountain | 76.12% | 1.32e-06 | 32.6s |
| 10 | 0.4430 | 0.5919 | 99.86% | 92.00% | 100.00% | 99.78% | mountain | 76.12% | 5.00e-05 | 34.8s |
| 11 | 0.4473 | 0.6015 | 99.71% | 93.33% | 100.00% | 99.78% | mountain | 80.60% | 4.97e-05 | 34.8s |
| 12 | 0.4448 | 0.5965 | 99.90% | 92.89% | 100.00% | 99.33% | mountain | 77.61% | 4.88e-05 | 32.9s |
| 13 | 0.4436 | 0.6166 | 99.71% | 93.33% | 100.00% | 99.11% | mountain | 74.63% | 4.73e-05 | 33.8s |
| **14** ⭐ | 0.4397 | 0.6049 | 99.95% | 94.22% | 100.00% | 99.56% | mountain | 79.10% | 4.52e-05 | 32.0s |
| 15 | 0.4362 | 0.6073 | 99.90% | 92.89% | 100.00% | 99.11% | mountain | 77.61% | 4.27e-05 | 34.3s |
| 16 | 0.4350 | 0.5992 | 99.95% | 92.89% | 100.00% | 99.56% | mountain | 79.10% | 3.97e-05 | 33.2s |
| 17 | 0.4374 | 0.6095 | 99.81% | 92.67% | 100.00% | 99.33% | mountain | 76.12% | 3.64e-05 | 33.8s |
| 18 | 0.4314 | 0.6047 | 99.95% | 93.11% | 100.00% | 99.33% | mountain | 77.61% | 3.28e-05 | 33.2s |
| 19 | 0.4306 | 0.5995 | 100.00% | 92.44% | 100.00% | 99.56% | mountain | 76.12% | 2.90e-05 | 33.9s |
| 20 | 0.4302 | 0.5975 | 100.00% | 92.67% | 100.00% | 99.56% | mountain | 77.61% | 2.50e-05 | 34.3s |
| 21 | 0.4319 | 0.6093 | 99.95% | 92.00% | 100.00% | 99.56% | mountain | 76.12% | 2.11e-05 | 33.2s |
| **22** (Last) | 0.4300 | 0.5915 | 100.00% | 93.11% | 100.00% | 99.78% | mountain | 79.10% | 1.73e-05 | 33.6s |

## Training Results Summary
| Metric | Value |
|---|---|
| Best Validation Accuracy | 94.22% |
| Best Epoch | 14 |
| Final Training Accuracy | 100.00% |
| Final Validation Accuracy | 93.11% |
| Final Training Loss | 0.4300 |
| Final Validation Loss | 0.5915 |
| Best Train Top-5 Accuracy | 100.00% |
| Best Val Top-5 Accuracy | 100.00% |

## Saved Model
- **Filename:** `Landscape_resnet50_E14_VAL94.22.pth`
- **Path:** `../models\Landscape_resnet50_E14_VAL94.22.pth`

---
*Report generated automatically by `training_report.py`*