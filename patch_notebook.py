"""Patch landscape_classifier.ipynb — refactor inline logic to module calls."""
import json
import re

NB = 'src/notebooks/landscape_classifier.ipynb'
with open(NB) as f:
    nb = json.load(f)


def src(*lines):
    """Build a list of source lines (each with trailing newline, except the last)."""
    result = [line + '\n' for line in lines]
    if result:
        result[-1] = result[-1].rstrip('\n')
    return result


# ── Cell 1: Imports ───────────────────────────────────────────────────────────
nb['cells'][1]['source'] = src(
    "import os",
    "import sys",
    "import random",
    "import numpy as np",
    "import torch",
    "import matplotlib.pyplot as plt",
    "from PIL import Image",
    "from sklearn.metrics import ConfusionMatrixDisplay",
    "",
    "# Add scripts directory to path for custom imports",
    "sys.path.insert(0, os.path.abspath('../scripts'))",
    "from gpu_utils import CheckGPU, CheckCUDA, CheckGPUBrief",
    "from dataset_counter import CountDataset",
    "from training_report import generate_training_report",
    "from training_visualizer import plot_gradient_descent",
    "from data_utils import (",
    "    build_transforms, load_and_split_dataset,",
    "    compute_class_weights, create_dataloaders,",
    ")",
    "from model_builder import (",
    "    build_model, create_criterion, create_optimizer,",
    "    create_scheduler, save_checkpoint, get_scheduler_params,",
    ")",
    "from trainer import train_model",
    "from evaluation import (",
    "    run_test_inference, compute_confusion_metrics,",
    "    compute_per_class_metrics, run_error_analysis,",
    ")",
    "",
    'print("\u2705 All libraries and custom modules imported successfully!")',
)
print("Cell  1 updated.")

# ── Cell 11: Transforms ───────────────────────────────────────────────────────
nb['cells'][11]['source'] = src(
    "train_transforms, val_test_transforms, applied_augmentations = build_transforms(",
    "    img_height=IMG_HEIGHT,",
    "    img_width=IMG_WIDTH,",
    "    normalize_mean=NORMALIZE_MEAN,",
    "    normalize_std=NORMALIZE_STD,",
    "    use_augmentation=USE_AUGMENTATION,",
    "    augmentation_options=AUGMENTATION_OPTIONS,",
    "    rotation_degrees=ROTATION_DEGREES,",
    "    affine_translate=AFFINE_TRANSLATE,",
    "    affine_scale=AFFINE_SCALE,",
    "    color_brightness=COLOR_BRIGHTNESS,",
    "    color_contrast=COLOR_CONTRAST,",
    "    color_saturation=COLOR_SATURATION,",
    "    color_hue=COLOR_HUE,",
    "    perspective_distortion=PERSPECTIVE_DISTORTION,",
    "    perspective_prob=PERSPECTIVE_PROB,",
    "    resized_crop_scale=RESIZED_CROP_SCALE,",
    "    gaussian_blur_kernel=GAUSSIAN_BLUR_KERNEL,",
    "    gaussian_blur_prob=GAUSSIAN_BLUR_PROB,",
    "    erasing_prob=ERASING_PROB,",
    "    erasing_scale=ERASING_SCALE,",
    ")",
)
print("Cell 11 updated.")

# ── Cell 13: Dataset Loading ──────────────────────────────────────────────────
nb['cells'][13]['source'] = src(
    "data = load_and_split_dataset(",
    "    dataset_dir=DATASET_DIR,",
    "    train_split=TRAIN_SPLIT,",
    "    val_split=VAL_SPLIT,",
    "    test_split=TEST_SPLIT,",
    "    train_transforms=train_transforms,",
    "    val_test_transforms=val_test_transforms,",
    "    seed=SEED,",
    ")",
    "",
    'full_dataset  = data["full_dataset"]',
    'train_subset  = data["train_subset"]',
    'val_subset    = data["val_subset"]',
    'test_subset   = data["test_subset"]',
    'train_dataset = data["train_dataset"]',
    'val_dataset   = data["val_dataset"]',
    'test_dataset  = data["test_dataset"]',
    'NUM_CLASSES   = data["num_classes"]',
    'class_names   = data["class_names"]',
    'train_size    = data["train_size"]',
    'val_size      = data["val_size"]',
    'test_size     = data["test_size"]',
)
print("Cell 13 updated.")

# ── Cell 15: Class Weights ────────────────────────────────────────────────────
nb['cells'][15]['source'] = src(
    "weights_data = compute_class_weights(",
    "    full_dataset=full_dataset,",
    "    train_subset=train_subset,",
    "    num_classes=NUM_CLASSES,",
    "    class_names=class_names,",
    "    use_weighted_sampler=USE_WEIGHTED_SAMPLER,",
    ")",
    "",
    'class_weights            = weights_data["class_weights"]',
    'sample_weights           = weights_data["sample_weights"]',
    'train_class_counts_array = weights_data["train_class_counts"]',
)
print("Cell 15 updated.")

# ── Cell 17: DataLoaders ──────────────────────────────────────────────────────
nb['cells'][17]['source'] = src(
    "train_loader, val_loader, test_loader = create_dataloaders(",
    "    train_dataset=train_dataset,",
    "    val_dataset=val_dataset,",
    "    test_dataset=test_dataset,",
    "    batch_size=BATCH_SIZE,",
    "    use_weighted_sampler=USE_WEIGHTED_SAMPLER,",
    "    sample_weights=sample_weights,",
    ")",
)
print("Cell 17 updated.")

# ── Cell 21: Remove history dict (keep everything else) ───────────────────────
c21 = ''.join(nb['cells'][21]['source'])
history_block = (
    "# Training history dictionary (global)\n"
    "history = {\n"
    '    "train_loss": [],\n'
    '    "val_loss": [],\n'
    '    "train_acc": [],\n'
    '    "val_acc": [],\n'
    '    "train_top5_acc": [],\n'
    '    "val_top5_acc": [],\n'
    '    "learning_rates": [],\n'
    '    "worst_class_acc": [],\n'
    '    "worst_class_name": [],\n'
    '    "epoch_times": []\n'
    "}\n"
    "\n"
)
if history_block in c21:
    c21 = c21.replace(history_block, "")
    nb['cells'][21]['source'] = list(c21)
    print("Cell 21 updated (history dict removed).")
else:
    print("Cell 21 WARNING: history block not found — may already be removed or format differs.")
    print("  Searching for partial match...")
    # Try regex
    match = re.search(r'# Training history dictionary.*?}\n\n', c21, re.DOTALL)
    if match:
        c21 = c21[:match.start()] + c21[match.end():]
        nb['cells'][21]['source'] = list(c21)
        print("  Cell 21 updated via regex.")
    else:
        print("  Cell 21: no match found, leaving unchanged.")

# ── Cell 23: Model Construction ───────────────────────────────────────────────
nb['cells'][23]['source'] = src(
    "CheckCUDA()",
    "model, device, in_features = build_model(",
    "    model_arch=MODEL_ARCH,",
    "    num_classes=NUM_CLASSES,",
    "    dropout_rate=DROPOUT_RATE,",
    "    use_progressive_unfreeze=USE_PROGRESSIVE_UNFREEZE,",
    ")",
)
print("Cell 23 updated.")

# ── Cell 25: Loss / Optimizer / Scheduler ─────────────────────────────────────
nb['cells'][25]['source'] = src(
    "criterion = create_criterion(class_weights, LABEL_SMOOTHING, device)",
    "",
    "optimizer = create_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)",
    "if USE_PROGRESSIVE_UNFREEZE:",
    '    print("   Note: Optimizer tracks head params only; backbone params added on unfreeze")',
    "",
    "scheduler = create_scheduler(",
    "    optimizer,",
    "    scheduler_name=LR_SCHEDULER,",
    "    cosine_t_0=COSINE_T_0,",
    "    cosine_t_mult=COSINE_T_MULT,",
    "    cosine_eta_min=COSINE_ETA_MIN,",
    "    step_size=STEP_SIZE,",
    "    step_gamma=STEP_GAMMA,",
    "    exp_gamma=EXP_GAMMA,",
    "    plateau_factor=PLATEAU_FACTOR,",
    "    plateau_patience=PLATEAU_PATIENCE,",
    "    plateau_min_lr=PLATEAU_MIN_LR,",
    "    cosine_t_max=COSINE_T_MAX,",
    "    cosine_anneal_eta_min=COSINE_ANNEAL_ETA_MIN,",
    ")",
)
print("Cell 25 updated.")

# ── Cell 27: Fix hardcoded scheduler name ─────────────────────────────────────
c27 = ''.join(nb['cells'][27]['source'])
old27 = "f\"{'  LR Scheduler:':<25} CosineAnnealingWarmRestarts\""
new27 = "f\"{'  LR Scheduler:':<25} {_scheduler_descriptions[LR_SCHEDULER]}\""
if old27 in c27:
    c27 = c27.replace(old27, new27)
    nb['cells'][27]['source'] = list(c27)
    print("Cell 27 updated.")
else:
    print("Cell 27 WARNING: pattern not found, skipping.")

# ── Cell 29: Replace training functions with stub + CheckGPUBrief ─────────────
nb['cells'][29]['source'] = src(
    "# Training functions (calculate_top_k_accuracy, _unfreeze_layer, train_model)",
    "# have been moved to  src/scripts/trainer.py",
    "CheckGPUBrief()",
)
print("Cell 29 updated.")

# ── Cell 31: Train + Save + Report ───────────────────────────────────────────
nb['cells'][31]['source'] = src(
    "# Train the model — history dict is created and returned by train_model()",
    "model, best_epoch, best_val_acc, total_training_time, history = train_model(",
    "    model=model,",
    "    criterion=criterion,",
    "    optimizer=optimizer,",
    "    scheduler=scheduler,",
    "    train_loader=train_loader,",
    "    val_loader=val_loader,",
    "    device=device,",
    "    class_names=class_names,",
    "    num_classes=NUM_CLASSES,",
    "    num_epochs=MAX_EPOCHS,",
    "    early_stopping_patience=EARLY_STOPPING_PATIENCE,",
    "    max_grad_norm=MAX_GRAD_NORM,",
    "    scheduler_name=LR_SCHEDULER,",
    "    weight_decay=WEIGHT_DECAY,",
    "    use_progressive_unfreeze=USE_PROGRESSIVE_UNFREEZE,",
    "    unfreeze_schedule=UNFREEZE_SCHEDULE,",
    ")",
    "",
    "# Save model checkpoint",
    "model_name, model_path = save_checkpoint(",
    "    model=model,",
    "    optimizer=optimizer,",
    "    best_epoch=best_epoch,",
    "    best_val_acc=best_val_acc,",
    "    model_arch=MODEL_ARCH,",
    "    num_classes=NUM_CLASSES,",
    "    class_names=class_names,",
    "    history=history,",
    "    use_progressive_unfreeze=USE_PROGRESSIVE_UNFREEZE,",
    "    unfreeze_schedule=UNFREEZE_SCHEDULE,",
    "    save_dir=MODEL_SAVE_DIR,",
    "    dataset_name=DATASET_NAME,",
    ")",
    "",
    "# Assemble training report data",
    "import platform",
    "scheduler_params = get_scheduler_params(",
    "    LR_SCHEDULER,",
    "    cosine_t_0=COSINE_T_0,       cosine_t_mult=COSINE_T_MULT,   cosine_eta_min=COSINE_ETA_MIN,",
    "    step_size=STEP_SIZE,          step_gamma=STEP_GAMMA,",
    "    exp_gamma=EXP_GAMMA,",
    "    plateau_factor=PLATEAU_FACTOR, plateau_patience=PLATEAU_PATIENCE, plateau_min_lr=PLATEAU_MIN_LR,",
    "    cosine_t_max=COSINE_T_MAX,    cosine_anneal_eta_min=COSINE_ANNEAL_ETA_MIN,",
    ")",
    "",
    "report_data = {",
    '    "dataset_name":            DATASET_NAME,',
    '    "model_arch":              MODEL_ARCH,',
    '    "pretrained_weights":      "IMAGENET1K_V2",',
    '    "img_size":                (IMG_HEIGHT, IMG_WIDTH),',
    '    "num_classes":             NUM_CLASSES,',
    '    "class_names":             class_names,',
    '    "total_params":            sum(p.numel() for p in model.parameters()),',
    '    "fc_in_features":          in_features,',
    '    "dropout_rate":            DROPOUT_RATE,',
    '    "dataset_dir":             DATASET_DIR,',
    '    "total_samples":           len(full_dataset),',
    '    "train_size":              train_size,',
    '    "val_size":                val_size,',
    '    "test_size":               test_size,',
    '    "train_split":             TRAIN_SPLIT,',
    '    "val_split":               VAL_SPLIT,',
    '    "test_split":              TEST_SPLIT,',
    '    "batch_size":              BATCH_SIZE,',
    '    "use_weighted_sampler":    USE_WEIGHTED_SAMPLER,',
    '    "use_augmentation":        USE_AUGMENTATION,',
    '    "augmentation_options":    AUGMENTATION_OPTIONS,',
    '    "rotation_degrees":        ROTATION_DEGREES,',
    '    "optimizer_name":          "AdamW",',
    '    "learning_rate":           LEARNING_RATE,',
    '    "weight_decay":            WEIGHT_DECAY,',
    '    "label_smoothing":         LABEL_SMOOTHING,',
    '    "max_grad_norm":           MAX_GRAD_NORM,',
    '    "scheduler_name":          LR_SCHEDULER,',
    '    "scheduler_params":        scheduler_params,',
    '    "cosine_t_0":              COSINE_T_0,',
    '    "cosine_t_mult":           COSINE_T_MULT,',
    '    "cosine_eta_min":          COSINE_ETA_MIN,',
    '    "max_epochs":              MAX_EPOCHS,',
    '    "early_stopping_patience": EARLY_STOPPING_PATIENCE,',
    '    "progressive_unfreeze":    USE_PROGRESSIVE_UNFREEZE,',
    '    "unfreeze_schedule":       UNFREEZE_SCHEDULE if USE_PROGRESSIVE_UNFREEZE else None,',
    '    "seed":                    SEED,',
    '    "device":                  str(device),',
    '    "gpu_name":                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",',
    '    "gpu_memory_gb":           f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}" if torch.cuda.is_available() else "N/A",',
    '    "cuda_version":            torch.version.cuda if torch.version.cuda else "N/A",',
    '    "cudnn_version":           str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A",',
    '    "torch_version":           torch.__version__,',
    '    "python_version":          platform.python_version(),',
    '    "os_info":                 platform.platform(),',
    '    "history":                 history,',
    '    "best_epoch":              best_epoch,',
    '    "best_val_acc":            best_val_acc,',
    '    "total_training_time":     total_training_time,',
    '    "model_filename":          model_name,',
    '    "model_path":              model_path,',
    "}",
    "",
    "# Save results directory, markdown report, and training visualisation",
    'model_base_name = model_name.replace(".pth", "")',
    'results_dir     = os.path.join(MODEL_SAVE_DIR, "results", model_base_name)',
    "os.makedirs(results_dir, exist_ok=True)",
    'print(f"Results directory: {results_dir}")',
    "",
    'report_path = os.path.join(results_dir, f"{model_base_name}_report.md")',
    "generate_training_report(report_data, report_path)",
    "",
    'vis_path = os.path.join(results_dir, f"{model_base_name}_gradient_descent.png")',
    "plot_gradient_descent(history, best_epoch, save_path=vis_path)",
    "",
    'print(f"Report and visualization saved in: {results_dir}")',
)
print("Cell 31 updated.")

# ── Cell 35: Inference + Confusion Metrics + Plots ───────────────────────────
nb['cells'][35]['source'] = src(
    "# Run inference on the test set",
    "infer      = run_test_inference(model, test_loader, device)",
    'all_preds  = infer["all_preds"]',
    'all_labels = infer["all_labels"]',
    'all_probs  = infer["all_probs"]',
    "",
    "# Compute confusion matrix, per-class accuracy, and confidence metrics",
    "cm_data = compute_confusion_metrics(all_preds, all_labels, all_probs, class_names)",
    "",
    'cm                    = cm_data["cm"]',
    'cm_normalized         = cm_data["cm_normalized"]',
    'accuracy              = cm_data["accuracy"]',
    'balanced_acc          = cm_data["balanced_acc"]',
    'per_class_acc         = cm_data["per_class_acc"]',
    'best_class_idx        = cm_data["best_class_idx"]',
    'worst_class_idx       = cm_data["worst_class_idx"]',
    'most_confused_idx     = cm_data["most_confused_idx"]',
    'most_confused_value   = cm_data["most_confused_value"]',
    'correct_confidences   = cm_data["correct_confidences"]',
    'incorrect_confidences = cm_data["incorrect_confidences"]',
    'avg_correct_conf      = cm_data["avg_correct_conf"]',
    'avg_incorrect_conf    = cm_data["avg_incorrect_conf"]',
    "",
    'print("\\n\U0001f4ca Classification Report Summary:")',
    'print(cm_data["classification_report_str"])',
    "",
    "# ── Confusion matrix plots (visualization stays in notebook) ──",
    "fig, axes = plt.subplots(1, 2, figsize=(18, 7))",
    "",
    "disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)",
    "disp1.plot(ax=axes[0], cmap='Blues', xticks_rotation=45, colorbar=True)",
    "axes[0].set_title('Confusion Matrix (Absolute Counts)', fontsize=14, fontweight='bold')",
    "",
    "disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)",
    "disp2.plot(ax=axes[1], cmap='Greens', xticks_rotation=45, colorbar=True, values_format='.2f')",
    "axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')",
    "",
    "plt.suptitle(f'Confusion Matrices \u2014 {MODEL_ARCH.upper()}', fontsize=16, fontweight='bold')",
    "plt.tight_layout()",
    "plt.show()",
)
print("Cell 35 updated.")

# ── Cell 37: Per-Class Metrics ────────────────────────────────────────────────
nb['cells'][37]['source'] = src(
    "pcm = compute_per_class_metrics(all_preds, all_labels, class_names, per_class_acc)",
    "",
    'precision           = pcm["precision"]',
    'recall              = pcm["recall"]',
    'f1                  = pcm["f1"]',
    'support             = pcm["support"]',
    'sorted_by_f1        = pcm["class_metrics"]',
    'macro_precision     = pcm["macro_precision"]',
    'macro_recall        = pcm["macro_recall"]',
    'macro_f1            = pcm["macro_f1"]',
    'weighted_precision  = pcm["weighted_precision"]',
    'weighted_recall     = pcm["weighted_recall"]',
    'weighted_f1         = pcm["weighted_f1"]',
    "",
    'print("\\n\U0001f4ca Detailed Per-Class Metrics:")',
    'print("=" * 90)',
    "print(f\"{'Class':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\")",
    'print("=" * 90)',
    'print("\U0001f3c6 CLASSES RANKED BY F1-SCORE:")',
    "for m in sorted_by_f1:",
    "    print(f\"{m['class']:<12} {m['accuracy']:>6.2f}%      {m['precision']:>6.2f}%      \"",
    "          f\"{m['recall']:>6.2f}%      {m['f1']:>6.2f}%      {m['support']:<10}\")",
    'print("=" * 90)',
    'print("\\n\U0001f4ca AGGREGATE METRICS:")',
    "print(f\"  Macro Avg:    Precision={macro_precision:.2f}%  Recall={macro_recall:.2f}%  F1={macro_f1:.2f}%\")",
    "print(f\"  Weighted Avg: Precision={weighted_precision:.2f}%  Recall={weighted_recall:.2f}%  F1={weighted_f1:.2f}%\")",
)
print("Cell 37 updated.")

# ── Cell 41: Error Analysis ───────────────────────────────────────────────────
nb['cells'][41]['source'] = src(
    "ea = run_error_analysis(",
    "    all_preds, all_labels, all_probs, class_names, support, incorrect_confidences",
    ")",
    "",
    'num_errors = ea["num_errors"]',
    "",
    'print("\U0001f50d Analyzing misclassification patterns...\\n")',
    "print(f\"\u274c Total misclassifications: {num_errors} out of {len(all_labels)} ({ea['error_rate']:.2f}%)\")",
    "",
    'print("\\n\U0001f534 Classes with Most Errors:")',
    'print("=" * 60)',
    "print(f\"{'Class':<15} {'Errors':<12} {'Error Rate':<15}\")",
    'print("=" * 60)',
    "for cls_idx, error_count in ea[\"sorted_error_classes\"]:",
    "    total    = int(support[cls_idx])",
    "    err_rate = (error_count / total) * 100 if total > 0 else 0",
    "    print(f\"{class_names[cls_idx]:<15} {error_count:<12} {err_rate:>6.2f}%\")",
    'print("=" * 60)',
    "",
    'print("\\n\U0001f4ca Confidence Analysis for Misclassifications:")',
    "q = ea[\"conf_quartiles\"]",
    "print(f\"  25th percentile:          {q[0] * 100:.2f}%\")",
    "print(f\"  50th percentile (median): {q[1] * 100:.2f}%\")",
    "print(f\"  75th percentile:          {q[2] * 100:.2f}%\")",
    "print(f\"  High-confidence errors (>80%): {ea['high_confidence_errors']} \"",
    "      f\"({ea['high_confidence_errors'] / max(num_errors, 1) * 100:.1f}%)\")",
    "",
    'print("\\n\U0001f504 Most Common Misclassification Patterns:")',
    'print("=" * 70)',
    "print(f\"{'True Class':<15} {'Predicted As':<15} {'Count':<12} {'% of Class':<15}\")",
    'print("=" * 70)',
    "for p in ea[\"top_confusion_patterns\"]:",
    "    print(f\"{p['true_class']:<15} {p['predicted_as']:<15} {p['count']:<12} {p['pct_of_class']:>6.2f}%\")",
    'print("=" * 70)',
    "",
    'print("\\n\u2705 Error analysis complete!")',
)
print("Cell 41 updated.")

# ── Cell 43: Fix hardcoded scheduler in final summary ─────────────────────────
c43 = ''.join(nb['cells'][43]['source'])
# Use regex to catch any variant of the hardcoded line
c43_new = re.sub(
    r'print\(f"  Scheduler: CosineAnnealingWarmRestarts[^"]*"\)',
    'print(f"  Scheduler: {_scheduler_descriptions[LR_SCHEDULER]}")',
    c43,
)
if c43_new != c43:
    nb['cells'][43]['source'] = list(c43_new)
    print("Cell 43 updated.")
else:
    print("Cell 43: pattern not found — scheduler line may already be correct or absent.")

# ── Save ──────────────────────────────────────────────────────────────────────
with open(NB, 'w') as f:
    json.dump(nb, f, indent=1)

print("\n✅ Notebook patched and saved.")
