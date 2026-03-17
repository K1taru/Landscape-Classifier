"""
Training Report Generator
=========================
Generates a comprehensive Markdown (.md) training report containing
model specs, dataset info, hyperparameters, hardware, timing, and
full epoch-by-epoch status.
"""

import os
import platform
from datetime import datetime


def _format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hrs}h {mins}m {secs:.1f}s"


def generate_training_report(report_data, save_path):
    """
    Generate a comprehensive training report as a Markdown (.md) file.

    Parameters
    ----------
    report_data : dict
        Dictionary containing all training metadata, configuration,
        hardware info, history, and timing data.
    save_path : str
        Full path where the .md file will be saved.

    Returns
    -------
    str
        The full path of the saved report.
    """
    d = report_data
    h = d["history"]
    total_epochs_run = len(h["train_loss"])
    epoch_times = h.get("epoch_times", [0.0] * total_epochs_run)
    time_to_best = sum(epoch_times[: d["best_epoch"]])
    total_time = d.get("total_training_time", sum(epoch_times))
    avg_epoch_time = sum(epoch_times) / max(len(epoch_times), 1)

    lines = []

    def add(text=""):
        lines.append(text)

    # ── Header ────────────────────────────────────────────────────────
    add(f"# {d.get('dataset_name', 'Landscape')} Classifier - Training Report")
    add(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add()
    add("---")
    add()

    # ── Model Specifications ──────────────────────────────────────────
    add("## Model Specifications")
    add("| Parameter | Value |")
    add("|---|---|")
    add(f"| Architecture | {d['model_arch'].upper()} |")
    add(f"| Pre-trained Weights | {d.get('pretrained_weights', 'IMAGENET1K_V2')} |")
    add(f"| Input Size | {d['img_size'][0]} x {d['img_size'][1]} |")
    add(f"| Number of Classes | {d['num_classes']} |")
    add(f"| Class Names | {', '.join(d['class_names'])} |")
    add(f"| Total Parameters | {d['total_params']:,} |")
    add(
        f"| Classifier Head | Dropout({d['dropout_rate']}) -> "
        f"Linear({d.get('fc_in_features', 2048)}, {d['num_classes']}) |"
    )
    add()

    # ── Dataset Information ───────────────────────────────────────────
    add("## Dataset Information")
    add("| Parameter | Value |")
    add("|---|---|")
    add(f"| Dataset | {d.get('dataset_name', 'Landscape (Intel Image Classification)')} |")
    add(f"| Dataset Directory | `{d['dataset_dir']}` |")
    add(f"| Total Samples | {d['total_samples']:,} |")
    add(f"| Training Set | {d['train_size']:,} ({d['train_split']*100:.0f}%) |")
    add(f"| Validation Set | {d['val_size']:,} ({d['val_split']*100:.0f}%) |")
    add(f"| Test Set | {d['test_size']:,} ({d['test_split']*100:.0f}%) |")
    add(f"| Batch Size | {d['batch_size']} |")
    add(f"| Weighted Sampler | {'Enabled' if d['use_weighted_sampler'] else 'Disabled'} |")
    add(f"| Random Seed | {d['seed']} |")
    add()

    # ── Hyperparameters ───────────────────────────────────────────────
    add("## Hyperparameters")
    add("| Parameter | Value |")
    add("|---|---|")
    add(f"| Optimizer | {d.get('optimizer_name', 'AdamW')} |")
    add(f"| Learning Rate | {d['learning_rate']} |")
    add(f"| Weight Decay | {d['weight_decay']} |")
    add(f"| Label Smoothing | {d['label_smoothing']} |")
    add(f"| Dropout Rate | {d['dropout_rate']} |")
    add(f"| Gradient Clipping (max_norm) | {d['max_grad_norm']} |")
    add(f"| LR Scheduler | {d.get('scheduler_name', 'CosineAnnealingWarmRestarts')} |")
    scheduler_params = d.get('scheduler_params')
    if scheduler_params:
        for param_key, param_val in scheduler_params.items():
            add(f"| Scheduler {param_key} | {param_val} |")
    else:
        # Legacy fallback for reports generated before scheduler_params was added
        add(f"| Scheduler T_0 | {d.get('cosine_t_0', 'N/A')} |")
        add(f"| Scheduler T_mult | {d.get('cosine_t_mult', 'N/A')} |")
        add(f"| Scheduler eta_min | {d.get('cosine_eta_min', 'N/A')} |")
    add(f"| Max Epochs | {d['max_epochs']} |")
    add(f"| Early Stopping Patience | {d['early_stopping_patience']} |")
    prog_unfreeze = d.get('progressive_unfreeze', False)
    add(f"| Progressive Unfreezing | {'Enabled' if prog_unfreeze else 'Disabled'} |")
    if prog_unfreeze and d.get('unfreeze_schedule'):
        schedule_str = ", ".join(
            f"{layer} @ epoch {ep}"
            for layer, ep in sorted(d['unfreeze_schedule'].items(), key=lambda x: x[1])
        )
        add(f"| Unfreeze Schedule | {schedule_str} |")
    add()

    # ── Data Augmentation ─────────────────────────────────────────────
    add("## Data Augmentation")
    add(f"**Status:** {'Enabled' if d['use_augmentation'] else 'Disabled'}")
    add()
    if d["use_augmentation"]:
        add("| Augmentation | Enabled |")
        add("|---|---|")
        aug_display = {
            "random_horizontal_flip": "Random Horizontal Flip",
            "random_rotation": f"Random Rotation ({d.get('rotation_degrees', 20)}°)",
            "random_affine": "Random Affine",
            "color_jitter": "Color Jitter",
            "random_perspective": "Random Perspective",
            "random_resized_crop": "Random Resized Crop",
            "gaussian_blur": "Gaussian Blur",
            "random_erasing": "Random Erasing",
        }
        for key, label in aug_display.items():
            enabled = d["augmentation_options"].get(key, False)
            add(f"| {label} | {'Yes' if enabled else 'No'} |")
        add()

    # ── Hardware Information ──────────────────────────────────────────
    add("## Hardware Information")
    add("| Component | Details |")
    add("|---|---|")
    add(f"| Device | {d.get('device', 'N/A')} |")
    add(f"| GPU | {d.get('gpu_name', 'N/A')} |")
    add(f"| GPU Memory | {d.get('gpu_memory_gb', 'N/A')} GB |")
    add(f"| CUDA Version | {d.get('cuda_version', 'N/A')} |")
    add(f"| cuDNN Version | {d.get('cudnn_version', 'N/A')} |")
    add(f"| PyTorch Version | {d.get('torch_version', 'N/A')} |")
    add(f"| Python Version | {d.get('python_version', platform.python_version())} |")
    add(f"| OS | {d.get('os_info', platform.platform())} |")
    add()

    # ── Training Time ─────────────────────────────────────────────────
    add("## Training Time")
    add("| Metric | Value |")
    add("|---|---|")
    add(f"| Total Training Time | {_format_time(total_time)} |")
    add(f"| Time to Best Epoch (Saved Model) | {_format_time(time_to_best)} |")
    add(f"| Average Epoch Time | {_format_time(avg_epoch_time)} |")
    add(f"| Best Epoch | {d['best_epoch']} / {d['max_epochs']} |")
    add(f"| Epochs Run | {total_epochs_run} |")
    add(
        f"| Early Stopping | "
        f"{'Yes (stopped at epoch ' + str(total_epochs_run) + ')' if total_epochs_run < d['max_epochs'] else 'No (ran all epochs)'} |"
    )
    add()

    # ── Epoch-by-Epoch Table ──────────────────────────────────────────
    add("## Epoch-by-Epoch Training Status")
    add()
    add(
        "| Epoch | Train Loss | Val Loss | Train Acc | Val Acc "
        "| Top-5 Train | Top-5 Val | Worst Class | Worst Acc | LR | Time |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|")

    for i in range(total_epochs_run):
        epoch_num = i + 1
        is_best = epoch_num == d["best_epoch"]
        is_last = epoch_num == total_epochs_run

        label = str(epoch_num)
        if is_best:
            label = f"**{epoch_num}** ⭐"
        elif is_last and not is_best:
            label = f"**{epoch_num}** (Last)"

        worst_name = h["worst_class_name"][i] if i < len(h.get("worst_class_name", [])) else "-"
        worst_acc_val = h["worst_class_acc"][i] if i < len(h.get("worst_class_acc", [])) else 0
        epoch_t = epoch_times[i] if i < len(epoch_times) else 0

        add(
            f"| {label} | "
            f"{h['train_loss'][i]:.4f} | {h['val_loss'][i]:.4f} | "
            f"{h['train_acc'][i]*100:.2f}% | {h['val_acc'][i]*100:.2f}% | "
            f"{h['train_top5_acc'][i]*100:.2f}% | {h['val_top5_acc'][i]*100:.2f}% | "
            f"{worst_name} | {worst_acc_val*100:.2f}% | "
            f"{h['learning_rates'][i]:.2e} | {_format_time(epoch_t)} |"
        )

    add()

    # ── Training Results Summary ──────────────────────────────────────
    add("## Training Results Summary")
    add("| Metric | Value |")
    add("|---|---|")
    add(f"| Best Validation Accuracy | {d['best_val_acc']*100:.2f}% |")
    add(f"| Best Epoch | {d['best_epoch']} |")
    add(f"| Final Training Accuracy | {h['train_acc'][-1]*100:.2f}% |")
    add(f"| Final Validation Accuracy | {h['val_acc'][-1]*100:.2f}% |")
    add(f"| Final Training Loss | {h['train_loss'][-1]:.4f} |")
    add(f"| Final Validation Loss | {h['val_loss'][-1]:.4f} |")
    add(f"| Best Train Top-5 Accuracy | {max(h['train_top5_acc'])*100:.2f}% |")
    add(f"| Best Val Top-5 Accuracy | {max(h['val_top5_acc'])*100:.2f}% |")
    add()

    # ── Saved Model ───────────────────────────────────────────────────
    add("## Saved Model")
    add(f"- **Filename:** `{d.get('model_filename', 'N/A')}`")
    add(f"- **Path:** `{d.get('model_path', 'N/A')}`")
    add()
    add("---")
    add(f"*Report generated automatically by `training_report.py`*")

    # ── Write to file ─────────────────────────────────────────────────
    report_content = "\n".join(lines)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\u2705 Training report saved: {save_path}")
    return save_path
