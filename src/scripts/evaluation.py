"""
Evaluation
==========
Test-set inference, confusion matrix metrics, per-class metrics,
and error analysis for the Landscape Classifier pipeline.
"""

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ── Test Inference ────────────────────────────────────────────────────────────

def run_test_inference(model, test_loader, device):
    """
    Run inference on the test set and collect predictions, true labels,
    and per-sample softmax probabilities.

    Returns
    -------
    dict with keys:
        all_preds  np.ndarray (N,)
        all_labels np.ndarray (N,)
        all_probs  np.ndarray (N, num_classes)
    """
    print("\U0001f50d Collecting model predictions on test set...")
    all_preds  = []
    all_labels = []
    all_probs  = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs   = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    print("\u2705 Predictions collected successfully.")
    return {
        "all_preds":  np.array(all_preds),
        "all_labels": np.array(all_labels),
        "all_probs":  np.array(all_probs),
    }


# ── Confusion Matrix Metrics ──────────────────────────────────────────────────

def compute_confusion_metrics(all_preds, all_labels, all_probs, class_names):
    """
    Compute confusion matrix, accuracy, balanced accuracy, per-class accuracy,
    most-confused class pair, confidence statistics, and classification report.

    Returns
    -------
    dict with keys:
        cm, cm_normalized,
        accuracy, balanced_acc,
        per_class_acc,
        best_class_idx, worst_class_idx,
        most_confused_idx, most_confused_value,
        correct_confidences, incorrect_confidences,
        avg_correct_conf, avg_incorrect_conf,
        classification_report_str
    """
    cm       = confusion_matrix(all_labels, all_preds)
    accuracy = np.trace(cm) / np.sum(cm)

    print(f"\n{'=' * 60}")
    print(f"\U0001f3af Overall Test Accuracy: {accuracy * 100:.2f}%")
    print(f"{'=' * 60}")

    per_class_acc    = cm.diagonal() / cm.sum(axis=1)
    best_class_idx   = int(np.argmax(per_class_acc))
    worst_class_idx  = int(np.argmin(per_class_acc))

    print(f"\U0001f3c6 Best class:  {class_names[best_class_idx]}  "
          f"({per_class_acc[best_class_idx] * 100:.2f}%)")
    print(f"\u26a0\ufe0f  Worst class: {class_names[worst_class_idx]} "
          f"({per_class_acc[worst_class_idx] * 100:.2f}%)")

    # Most confused off-diagonal pair
    cm_copy = cm.copy().astype(float)
    np.fill_diagonal(cm_copy, 0)
    most_confused_idx   = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)
    most_confused_value = int(cm_copy[most_confused_idx])
    print(f"\U0001f504 Most confused: "
          f"{class_names[most_confused_idx[0]]} \u2194 {class_names[most_confused_idx[1]]} "
          f"({most_confused_value} misclassifications)")

    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    print(f"\u2696\ufe0f  Balanced Accuracy: {balanced_acc * 100:.2f}%")

    correct_mask          = all_preds == all_labels
    correct_confidences   = [all_probs[i][all_preds[i]] for i in range(len(all_preds)) if     correct_mask[i]]
    incorrect_confidences = [all_probs[i][all_preds[i]] for i in range(len(all_preds)) if not correct_mask[i]]
    avg_correct_conf      = float(np.mean(correct_confidences))   if correct_confidences   else 0.0
    avg_incorrect_conf    = float(np.mean(incorrect_confidences)) if incorrect_confidences else 0.0

    print(f"\u2705 Avg confidence (correct):   {avg_correct_conf   * 100:.2f}%")
    print(f"\u274c Avg confidence (incorrect): {avg_incorrect_conf * 100:.2f}%")
    print(f"{'=' * 60}\n")

    cm_normalized  = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    report_str     = classification_report(all_labels, all_preds, target_names=class_names, digits=3)

    return {
        "cm":                       cm,
        "cm_normalized":            cm_normalized,
        "accuracy":                 accuracy,
        "balanced_acc":             balanced_acc,
        "per_class_acc":            per_class_acc,
        "best_class_idx":           best_class_idx,
        "worst_class_idx":          worst_class_idx,
        "most_confused_idx":        most_confused_idx,
        "most_confused_value":      most_confused_value,
        "correct_confidences":      correct_confidences,
        "incorrect_confidences":    incorrect_confidences,
        "avg_correct_conf":         avg_correct_conf,
        "avg_incorrect_conf":       avg_incorrect_conf,
        "classification_report_str": report_str,
    }


# ── Per-Class Metrics ─────────────────────────────────────────────────────────

def compute_per_class_metrics(all_preds, all_labels, class_names, per_class_acc):
    """
    Compute per-class precision, recall, F1, and support; sort by F1 descending.

    Returns
    -------
    dict with keys:
        precision, recall, f1, support  (np.ndarray each)
        class_metrics (list[dict], each with class/accuracy/precision/recall/f1/support)
        macro_precision, macro_recall, macro_f1        (float, %)
        weighted_precision, weighted_recall, weighted_f1 (float, %)
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None,
    )

    class_metrics = [
        {
            "class":     class_names[i],
            "accuracy":  per_class_acc[i] * 100,
            "precision": precision[i] * 100,
            "recall":    recall[i]    * 100,
            "f1":        f1[i]        * 100,
            "support":   int(support[i]),
        }
        for i in range(len(class_names))
    ]
    sorted_by_f1 = sorted(class_metrics, key=lambda x: x["f1"], reverse=True)

    macro_precision    = float(np.mean(precision))                     * 100
    macro_recall       = float(np.mean(recall))                        * 100
    macro_f1           = float(np.mean(f1))                            * 100
    weighted_precision = float(np.average(precision, weights=support)) * 100
    weighted_recall    = float(np.average(recall,    weights=support)) * 100
    weighted_f1        = float(np.average(f1,        weights=support)) * 100

    return {
        "precision":          precision,
        "recall":             recall,
        "f1":                 f1,
        "support":            support,
        "class_metrics":      sorted_by_f1,
        "macro_precision":    macro_precision,
        "macro_recall":       macro_recall,
        "macro_f1":           macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall":    weighted_recall,
        "weighted_f1":        weighted_f1,
    }


# ── Error Analysis ────────────────────────────────────────────────────────────

def run_error_analysis(all_preds, all_labels, all_probs, class_names, support, incorrect_confidences):
    """
    Analyse misclassification patterns, confidence distribution for errors,
    and identify the most common true→predicted confusion pairs.

    Returns
    -------
    dict with keys:
        num_errors              int
        error_rate              float (%)
        sorted_error_classes    list[(class_idx, error_count)] desc
        conf_quartiles          np.ndarray [25th, 50th, 75th]
        high_confidence_errors  int  (confidence > 80%)
        top_confusion_patterns  list[dict] (top 5 true→predicted pairs)
            each dict: true_class, predicted_as, count, pct_of_class
    """
    misclassified_indices = np.where(all_preds != all_labels)[0]
    num_errors = len(misclassified_indices)
    error_rate = num_errors / len(all_labels) * 100

    # Build error_by_class: true_class_idx -> [predicted_class_idx, ...]
    error_by_class = {}
    for idx in misclassified_indices:
        true_cls = int(all_labels[idx])
        pred_cls = int(all_preds[idx])
        error_by_class.setdefault(true_cls, []).append(pred_cls)

    class_error_counts  = {cls: len(errs) for cls, errs in error_by_class.items()}
    sorted_error_classes = sorted(class_error_counts.items(), key=lambda x: x[1], reverse=True)

    # Confidence quartiles for incorrect predictions
    conf_quartiles = (
        np.percentile(incorrect_confidences, [25, 50, 75])
        if incorrect_confidences else np.zeros(3)
    )
    high_confidence_errors = sum(1 for c in incorrect_confidences if c > 0.8)

    # Top confusion patterns: most common predicted class for each true class
    top_confusion_patterns = []
    for true_cls_idx, pred_list in sorted(
        error_by_class.items(), key=lambda x: len(x[1]), reverse=True
    )[:5]:
        pred_counts = {}
        for pred in pred_list:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        most_common_pred, count = max(pred_counts.items(), key=lambda x: x[1])
        pct = (count / int(support[true_cls_idx])) * 100 if int(support[true_cls_idx]) > 0 else 0.0
        top_confusion_patterns.append({
            "true_class":   class_names[true_cls_idx],
            "predicted_as": class_names[most_common_pred],
            "count":        count,
            "pct_of_class": pct,
        })

    return {
        "num_errors":             num_errors,
        "error_rate":             error_rate,
        "sorted_error_classes":   sorted_error_classes,
        "conf_quartiles":         conf_quartiles,
        "high_confidence_errors": high_confidence_errors,
        "top_confusion_patterns": top_confusion_patterns,
    }
