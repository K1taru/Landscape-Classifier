"""
Trainer
=======
Training loop, top-k accuracy, and progressive layer unfreezing
for the Landscape Classifier pipeline.
"""

import copy
import time

import numpy as np
import torch
from tqdm import tqdm


# ── Helpers ───────────────────────────────────────────────────────────────────

def calculate_top_k_accuracy(outputs, labels, k=5):
    """Return the number of correct top-k predictions in a batch."""
    k = min(k, outputs.size(1))
    _, topk_preds = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))
    return correct.sum().item()


def _unfreeze_layer(model, layer_name, optimizer, scheduler, weight_decay):
    """
    Unfreeze a named layer group and add its parameters to the optimizer.

    Uses 1/10th of the current head LR for backbone layers to avoid
    disrupting pretrained features.
    ReduceLROnPlateau has no base_lrs attribute — the sync is skipped for it.
    """
    layer = getattr(model, layer_name, None)
    if layer is None:
        print(f"\u26a0\ufe0f  Layer '{layer_name}' not found on model, skipping")
        return

    newly_unfrozen = [p for p in layer.parameters() if not p.requires_grad]
    for p in newly_unfrozen:
        p.requires_grad = True

    if newly_unfrozen:
        current_lr  = optimizer.param_groups[0]["lr"]
        backbone_lr = current_lr * 0.1
        optimizer.add_param_group({
            "params": newly_unfrozen,
            "lr": backbone_lr,
            "weight_decay": weight_decay,
        })
        # ReduceLROnPlateau does not use base_lrs — guard before appending
        if hasattr(scheduler, "base_lrs"):
            scheduler.base_lrs.append(backbone_lr)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        count      = sum(p.numel() for p in newly_unfrozen)
        print(f"\U0001f513 Unfroze {layer_name}: {count:,} params "
              f"(lr={backbone_lr:.2e}) | Total trainable: {trainable:,}")


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    device,
    class_names,
    num_classes,
    num_epochs,
    early_stopping_patience,
    max_grad_norm,
    scheduler_name,
    weight_decay,
    use_progressive_unfreeze=False,
    unfreeze_schedule=None,
):
    """
    Full training loop with early stopping, gradient clipping, progressive
    unfreezing, top-k accuracy tracking, and configurable LR scheduling.

    Parameters
    ----------
    scheduler_name : str
        Name of the active scheduler. "ReduceLROnPlateau" requires
        scheduler.step(val_loss); all others use scheduler.step().

    Returns
    -------
    (model, best_epoch, best_val_acc, total_training_time, history)
        history is a dict with keys: train_loss, val_loss, train_acc, val_acc,
        train_top5_acc, val_top5_acc, learning_rates, worst_class_acc,
        worst_class_name, epoch_times.
    """
    if unfreeze_schedule is None:
        unfreeze_schedule = {}

    history = {
        "train_loss":      [],
        "val_loss":        [],
        "train_acc":       [],
        "val_acc":         [],
        "train_top5_acc":  [],
        "val_top5_acc":    [],
        "learning_rates":  [],
        "worst_class_acc": [],
        "worst_class_name": [],
        "epoch_times":     [],
    }

    best_val_acc   = 0.0
    best_epoch     = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    manually_stopped = False
    unfrozen_layers  = set()
    worst_name, worst_acc = class_names[0], 0.0  # initialise for KeyboardInterrupt edge case

    print("\n" + "=" * 80)
    print("\U0001f3af STARTING TRAINING")
    print("=" * 80)
    if use_progressive_unfreeze:
        print("\u2744\ufe0f  Progressive unfreezing is ACTIVE")
        print("   Backbone is frozen. Layers will unfreeze per schedule.")
    print("\U0001f6d1 Press Ctrl+C at any time to stop (best weights will be saved)")

    training_start_time = time.time()

    # Guard: re-running without a kernel restart may leave the optimizer with
    # extra param groups while the scheduler only knows about the original one(s).
    # ReduceLROnPlateau has no base_lrs — skip for it.
    if hasattr(scheduler, "base_lrs"):
        while len(scheduler.base_lrs) < len(optimizer.param_groups):
            scheduler.base_lrs.append(scheduler.base_lrs[0] * 0.1)
        if len(scheduler.base_lrs) > len(optimizer.param_groups):
            scheduler.base_lrs = scheduler.base_lrs[:len(optimizer.param_groups)]

    try:
        for epoch in range(num_epochs):
            print(f"\n{'=' * 80}")
            print(f"\U0001f4c6 Epoch {epoch + 1}/{num_epochs}")
            print(f"{'=' * 80}")

            # Progressive unfreezing check (1-indexed epoch)
            if use_progressive_unfreeze:
                for layer_name, unfreeze_at in unfreeze_schedule.items():
                    if (epoch + 1) >= unfreeze_at and layer_name not in unfrozen_layers:
                        _unfreeze_layer(model, layer_name, optimizer, scheduler, weight_decay)
                        unfrozen_layers.add(layer_name)

            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            if len(optimizer.param_groups) > 1:
                for i, pg in enumerate(optimizer.param_groups[1:], 1):
                    print(f"Learning Rate (backbone group {i}): {pg['lr']:.2e}")

            epoch_start = time.time()
            epoch_metrics = {}
            val_preds_epoch  = []
            val_labels_epoch = []

            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                    dataloader = train_loader
                else:
                    model.eval()
                    dataloader = val_loader

                running_loss          = 0.0
                running_corrects      = 0
                running_top5_corrects = 0
                total_samples         = 0

                phase_desc = "\U0001f525 TRAIN" if phase == "train" else "\u2705 VAL  "
                loop = tqdm(dataloader, desc=phase_desc, leave=False, ncols=100)

                for inputs, labels in loop:
                    inputs     = inputs.to(device)
                    labels     = labels.to(device)
                    batch_size = inputs.size(0)
                    total_samples += batch_size

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            optimizer.step()

                    running_loss          += loss.item() * batch_size
                    running_corrects      += torch.sum(preds == labels.data).item()
                    running_top5_corrects += calculate_top_k_accuracy(outputs, labels, k=5)

                    if phase == "val":
                        val_preds_epoch.extend(preds.cpu().numpy())
                        val_labels_epoch.extend(labels.cpu().numpy())

                    loop.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "acc":  f"{running_corrects / total_samples:.4f}",
                    })

                epoch_loss     = running_loss          / total_samples
                epoch_acc      = running_corrects       / total_samples
                epoch_top5_acc = running_top5_corrects  / total_samples

                epoch_metrics[f"{phase}_loss"]     = epoch_loss
                epoch_metrics[f"{phase}_acc"]      = epoch_acc
                epoch_metrics[f"{phase}_top5_acc"] = epoch_top5_acc

                print(f"{phase.upper():>5} | Loss: {epoch_loss:.4f} | "
                      f"Acc: {epoch_acc * 100:>6.2f}% | Top-5: {epoch_top5_acc * 100:>6.2f}%")

                if phase == "val":
                    _vp = np.array(val_preds_epoch)
                    _vl = np.array(val_labels_epoch)
                    per_cls = np.array([
                        (_vp[_vl == c] == c).sum() / max((_vl == c).sum(), 1)
                        for c in range(num_classes)
                    ])
                    worst_idx  = int(np.argmin(per_cls))
                    worst_name = class_names[worst_idx]
                    worst_acc  = per_cls[worst_idx]
                    print(f"  VAL  | Worst Class: {worst_name} ({worst_acc * 100:.2f}%)")

            # Step scheduler — ReduceLROnPlateau needs the monitored metric
            if scheduler_name == "ReduceLROnPlateau":
                scheduler.step(epoch_metrics["val_loss"])
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(epoch_metrics["train_loss"])
            history["val_loss"].append(epoch_metrics["val_loss"])
            history["train_acc"].append(epoch_metrics["train_acc"])
            history["val_acc"].append(epoch_metrics["val_acc"])
            history["train_top5_acc"].append(epoch_metrics["train_top5_acc"])
            history["val_top5_acc"].append(epoch_metrics["val_top5_acc"])
            history["learning_rates"].append(current_lr)
            history["worst_class_acc"].append(worst_acc)
            history["worst_class_name"].append(worst_name)
            epoch_time = time.time() - epoch_start
            history["epoch_times"].append(epoch_time)
            print(f"  TIME | Epoch Duration: {epoch_time:.1f}s")

            # Overfitting detection
            loss_gap = epoch_metrics["train_loss"] - epoch_metrics["val_loss"]
            acc_gap  = epoch_metrics["train_acc"]  - epoch_metrics["val_acc"]
            if loss_gap < -0.1 or acc_gap > 0.1:
                print(f"\u26a0\ufe0f  Overfitting: train-val gap = {acc_gap * 100:.2f}% acc, {loss_gap:.4f} loss")

            # Best model tracking
            if epoch_metrics["val_acc"] > best_val_acc:
                best_val_acc   = epoch_metrics["val_acc"]
                best_epoch     = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                print(f"\u2728 New best val accuracy: {best_val_acc * 100:.2f}%")
            else:
                epochs_without_improvement += 1
                print(f"\U0001f4ca No improvement for {epochs_without_improvement} epoch(s)")

            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n\U0001f6d1 Early stopping triggered after {epoch + 1} epochs")
                print(f"   Best val accuracy: {best_val_acc * 100:.2f}% at epoch {best_epoch}")
                break

    except KeyboardInterrupt:
        manually_stopped = True
        print(f"\n\n{'=' * 80}")
        print("\u270b MANUAL EARLY STOP (Ctrl+C)")
        print(f"{'=' * 80}")
        print(f"   Training interrupted — best model from epoch {best_epoch} will be restored")

    total_training_time = time.time() - training_start_time
    time_to_best        = sum(history["epoch_times"][:best_epoch])
    stop_reason = (
        "manual stop (Ctrl+C)" if manually_stopped
        else "early stopping" if epochs_without_improvement >= early_stopping_patience
        else "completed all epochs"
    )

    print("\n" + "=" * 80)
    print("\u2705 TRAINING COMPLETE")
    print(f"   Stop reason:       {stop_reason}")
    print(f"   Best Epoch:        {best_epoch}")
    print(f"   Best Val Accuracy: {best_val_acc * 100:.2f}%")
    print(f"   Total Time:        {total_training_time:.1f}s")
    print(f"   Time to Best:      {time_to_best:.1f}s")
    if use_progressive_unfreeze:
        print(f"   Layers unfrozen:   {sorted(unfrozen_layers) if unfrozen_layers else 'head only'}")
    print("=" * 80 + "\n")

    model.load_state_dict(best_model_wts)
    return model, best_epoch, best_val_acc, total_training_time, history
