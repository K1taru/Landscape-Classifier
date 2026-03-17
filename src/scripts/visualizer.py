"""
Visualizer
==========
Matplotlib plotting functions for the Landscape Classifier notebook.
Includes sample image display, training curves, confusion matrices,
and comprehensive performance analysis charts.
"""

import os
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay


# ── Sample Images ────────────────────────────────────────────────────────────

def plot_sample_images(dataset_dir, class_names, num_classes, samples_per_class=2):
    """
    Display random sample images from each class in a grid.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset root (ImageFolder structure).
    class_names : list[str]
        Sorted list of class names.
    num_classes : int
        Number of classes.
    samples_per_class : int
        Number of images to show per class (columns).
    """
    print("\U0001f5bc\ufe0f  Displaying sample landscape images from dataset...")

    fig, axes = plt.subplots(num_classes, samples_per_class,
                             figsize=(8, 3 * num_classes))

    # Handle single-class edge case (axes won't be 2D)
    if num_classes == 1:
        axes = np.array([axes])

    for row, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        selected = random.sample(image_files,
                                 min(samples_per_class, len(image_files)))

        for col, img_file in enumerate(selected):
            img_path = os.path.join(class_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            w, h = img.size

            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            axes[row, col].set_title(f'{class_name}\n{w}x{h}px',
                                     fontsize=10, fontweight='bold')

    plt.suptitle('Sample Landscape Images from Dataset',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print("\u2705 Sample visualization complete!")


# ── Training Curves ──────────────────────────────────────────────────────────

def plot_training_curves(history, model_arch):
    """
    Plot a 2x2 grid of training curves: Top-1 Accuracy, Loss,
    Top-5 Accuracy, and Learning Rate schedule.

    Parameters
    ----------
    history : dict
        Training history with keys: train_acc, val_acc, train_loss,
        val_loss, train_top5_acc, val_top5_acc, learning_rates.
    model_arch : str
        Model architecture name (for plot title).
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Top-1 Accuracy
    axes[0, 0].plot(history["train_acc"], label="Train Accuracy",
                    marker='o', linewidth=2)
    axes[0, 0].plot(history["val_acc"], label="Val Accuracy",
                    marker='s', linewidth=2)
    axes[0, 0].set_title("\U0001f4c8 Top-1 Accuracy",
                         fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])

    # 2. Loss
    axes[0, 1].plot(history["train_loss"], label="Train Loss",
                    marker='o', linewidth=2, color='orange')
    axes[0, 1].plot(history["val_loss"], label="Val Loss",
                    marker='s', linewidth=2, color='red')
    axes[0, 1].set_title("\U0001f4c9 Loss", fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Top-5 Accuracy
    axes[1, 0].plot(history["train_top5_acc"], label="Train Top-5",
                    marker='o', linewidth=2, color='green')
    axes[1, 0].plot(history["val_top5_acc"], label="Val Top-5",
                    marker='s', linewidth=2, color='darkgreen')
    axes[1, 0].set_title("\U0001f3af Top-5 Accuracy",
                         fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Top-5 Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])

    # 4. Learning Rate
    axes[1, 1].plot(history["learning_rates"],
                    marker='o', linewidth=2, color='purple')
    axes[1, 1].set_title("\U0001f4ca Learning Rate Schedule",
                         fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Training Curves - {model_arch.upper()}",
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()
    print("\u2705 Training curves displayed successfully!")


# ── Confusion Matrices ───────────────────────────────────────────────────────

def plot_confusion_matrices(cm, cm_normalized, class_names, model_arch):
    """
    Plot side-by-side absolute and normalized confusion matrices.

    Parameters
    ----------
    cm : np.ndarray
        Absolute confusion matrix.
    cm_normalized : np.ndarray
        Row-normalized confusion matrix.
    class_names : list[str]
        Class names for axis labels.
    model_arch : str
        Model architecture name (for plot title).
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=class_names)
    disp1.plot(ax=axes[0], cmap='Blues', xticks_rotation=45, colorbar=True)
    axes[0].set_title("Confusion Matrix (Absolute Counts)",
                      fontsize=14, fontweight='bold')

    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_normalized,
                                   display_labels=class_names)
    disp2.plot(ax=axes[1], cmap='Greens', xticks_rotation=45,
               colorbar=True, values_format='.2f')
    axes[1].set_title("Confusion Matrix (Normalized)",
                      fontsize=14, fontweight='bold')

    plt.suptitle(f"Confusion Matrices - {model_arch.upper()}",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ── Performance Analysis (6-panel) ──────────────────────────────────────────

def plot_performance_analysis(
    class_names, num_classes, per_class_acc, accuracy,
    precision, recall, f1, support,
    correct_confidences, incorrect_confidences, cm, model_arch,
):
    """
    Plot a 3x2 grid of performance analysis charts:
    1. Per-class accuracy bars
    2. Precision / Recall / F1 grouped bars
    3. Test set class support distribution
    4. Prediction confidence histogram (correct vs incorrect)
    5. Top most confused class pairs (horizontal bar)
    6. Accuracy vs support scatter with correlation

    Parameters
    ----------
    class_names : list[str]
    num_classes : int
    per_class_acc : np.ndarray
    accuracy : float
    precision, recall, f1, support : np.ndarray
    correct_confidences, incorrect_confidences : list[float]
    cm : np.ndarray
        Absolute confusion matrix.
    model_arch : str
    """
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))

    # 1. Per-Class Accuracy Bar Chart
    axes[0, 0].bar(class_names, per_class_acc * 100,
                   color='steelblue', alpha=0.7)
    axes[0, 0].axhline(y=accuracy * 100, color='red', linestyle='--',
                       linewidth=2, label=f'Overall Avg: {accuracy*100:.2f}%')
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Per-Class Accuracy',
                         fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Precision, Recall, F1-Score Comparison
    x_pos = np.arange(num_classes)
    width = 0.25
    axes[0, 1].bar(x_pos - width, precision * 100, width,
                   label='Precision', alpha=0.8, color='green')
    axes[0, 1].bar(x_pos, recall * 100, width,
                   label='Recall', alpha=0.8, color='orange')
    axes[0, 1].bar(x_pos + width, f1 * 100, width,
                   label='F1-Score', alpha=0.8, color='purple')
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Score (%)')
    axes[0, 1].set_title('Precision, Recall, F1-Score per Class',
                         fontsize=13, fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(class_names, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. Class Support Distribution
    axes[1, 0].bar(class_names, support, color='coral', alpha=0.7)
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].set_title('Test Set Class Distribution',
                         fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 4. Confidence Distribution (Correct vs Incorrect)
    axes[1, 1].hist(correct_confidences, bins=30, alpha=0.7,
                    label='Correct Predictions', color='green',
                    edgecolor='black')
    axes[1, 1].hist(incorrect_confidences, bins=30, alpha=0.7,
                    label='Incorrect Predictions', color='red',
                    edgecolor='black')
    axes[1, 1].set_xlabel('Confidence Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Confidence Distribution',
                         fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # 5. Top Most Confused Class Pairs
    cm_off_diagonal = cm.copy().astype(float)
    np.fill_diagonal(cm_off_diagonal, 0)
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm_off_diagonal[i, j] > 0:
                confused_pairs.append(
                    (class_names[i], class_names[j],
                     int(cm_off_diagonal[i, j]))
                )

    confused_pairs_sorted = sorted(confused_pairs,
                                   key=lambda x: x[2], reverse=True)[:10]
    pair_labels = [f"{p[0]}\u2192{p[1]}" for p in confused_pairs_sorted]
    pair_counts = [p[2] for p in confused_pairs_sorted]

    axes[2, 0].barh(pair_labels, pair_counts,
                    color='indianred', alpha=0.8)
    axes[2, 0].set_xlabel('Number of Misclassifications')
    axes[2, 0].set_ylabel('Class Pair (True\u2192Predicted)')
    axes[2, 0].set_title('Top Most Confused Class Pairs',
                         fontsize=13, fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3, axis='x')
    axes[2, 0].invert_yaxis()

    # 6. Accuracy vs Support Scatter Plot
    axes[2, 1].scatter(support, per_class_acc * 100, s=150, alpha=0.7,
                       c=range(num_classes), cmap='viridis',
                       edgecolors='black')
    for i, cn in enumerate(class_names):
        axes[2, 1].annotate(cn, (support[i], per_class_acc[i] * 100),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=9)
    axes[2, 1].set_xlabel('Test Set Support (samples)')
    axes[2, 1].set_ylabel('Accuracy (%)')
    axes[2, 1].set_title('Accuracy vs Class Support',
                         fontsize=13, fontweight='bold')
    axes[2, 1].grid(True, alpha=0.3)

    correlation = np.corrcoef(support, per_class_acc)[0, 1]
    axes[2, 1].text(
        0.05, 0.95, f'Correlation: {correlation:.3f}',
        transform=axes[2, 1].transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    plt.suptitle(
        f'Comprehensive Performance Analysis - {model_arch.upper()}',
        fontsize=16, fontweight='bold',
    )
    plt.tight_layout()
    plt.show()
    print("\u2705 Performance visualizations generated successfully!")
