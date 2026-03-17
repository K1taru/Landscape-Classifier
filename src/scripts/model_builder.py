"""
Model Builder
=============
Model construction, loss/optimizer/scheduler setup, and checkpoint saving
for the Landscape Classifier pipeline.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


# ── Model Construction ────────────────────────────────────────────────────────

def build_model(model_arch, num_classes, dropout_rate, use_progressive_unfreeze=False):
    """
    Load a pretrained ResNet50, replace the classifier head, and optionally
    freeze the backbone for progressive unfreezing.

    Returns
    -------
    (model, device, in_features)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_arch.lower() == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(
            f"Unsupported model_arch: '{model_arch}'. Only 'resnet50' is currently supported."
        )

    print(f"\n\u2705 Pre-trained {model_arch.upper()} loaded (weights: IMAGENET1K_V2)")

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, num_classes),
    )
    print(f"\u2705 Classifier replaced: {in_features} \u2192 Dropout({dropout_rate}) \u2192 {num_classes} classes")

    if use_progressive_unfreeze:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"\u2744\ufe0f  Progressive unfreezing enabled:")
        print(f"   Trainable parameters (head):     {trainable:,}")
        print(f"   Frozen parameters (backbone):    {frozen:,}")
        print(f"   Backbone layers will unfreeze according to schedule")
    else:
        print(f"\U0001f504 All layers trainable (progressive unfreezing disabled)")

    model = model.to(device)
    print(f"\u2705 Model moved to device: {device.type.upper()}")
    print(f"\U0001f4ca Total model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, device, in_features


# ── Loss / Optimizer / Scheduler ─────────────────────────────────────────────

def create_criterion(class_weights, label_smoothing, device):
    """
    Create a CrossEntropyLoss with per-class weights and label smoothing.

    Returns
    -------
    nn.CrossEntropyLoss
    """
    weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=label_smoothing)
    print(f"\u2705 Loss: CrossEntropyLoss (label_smoothing={label_smoothing}, class weights applied)")
    return criterion


def create_optimizer(model, learning_rate, weight_decay):
    """
    Create an AdamW optimizer over all parameters that currently require gradients.

    Returns
    -------
    optim.AdamW
    """
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    print(f"\u2705 Optimizer: AdamW (lr={learning_rate}, weight_decay={weight_decay})")
    return optimizer


def create_scheduler(
    optimizer,
    scheduler_name,
    cosine_t_0=15,
    cosine_t_mult=2,
    cosine_eta_min=1e-7,
    step_size=10,
    step_gamma=0.1,
    exp_gamma=0.95,
    plateau_factor=0.5,
    plateau_patience=5,
    plateau_min_lr=1e-7,
    cosine_t_max=50,
    cosine_anneal_eta_min=1e-7,
):
    """
    Create a learning rate scheduler by name.

    Supported schedulers
    --------------------
    "CosineAnnealingWarmRestarts"
        Cosine decay with periodic warm restarts — recommended for fine-tuning
        pretrained models. Escapes local minima via LR spikes at each restart.

    "StepLR"
        Drops LR by step_gamma every step_size epochs — simple and predictable.

    "ExponentialLR"
        Multiplies LR by exp_gamma every epoch — smooth continuous decay.

    "ReduceLROnPlateau"
        Reduces LR when val_loss stops improving — adaptive, no fixed schedule.
        Note: scheduler.step(val_loss) must be called; this is handled automatically
        in train_model() when scheduler_name == "ReduceLROnPlateau".

    "CosineAnnealingLR"
        Single cosine decay from LR to eta_min over T_max epochs, no restarts.

    Returns
    -------
    A PyTorch LR scheduler instance.
    """
    if scheduler_name == "CosineAnnealingWarmRestarts":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cosine_t_0, T_mult=cosine_t_mult, eta_min=cosine_eta_min,
        )
    elif scheduler_name == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=step_gamma,
        )
    elif scheduler_name == "ExponentialLR":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_gamma)
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=plateau_factor,
            patience=plateau_patience, min_lr=plateau_min_lr,
        )
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_t_max, eta_min=cosine_anneal_eta_min,
        )
    else:
        raise ValueError(
            f"Unknown scheduler_name: '{scheduler_name}'. "
            f"Choose from: CosineAnnealingWarmRestarts, StepLR, ExponentialLR, "
            f"ReduceLROnPlateau, CosineAnnealingLR"
        )

    print(f"\u2705 LR Scheduler: {scheduler_name}")
    return scheduler


def get_scheduler_params(
    scheduler_name,
    cosine_t_0=None,
    cosine_t_mult=None,
    cosine_eta_min=None,
    step_size=None,
    step_gamma=None,
    exp_gamma=None,
    plateau_factor=None,
    plateau_patience=None,
    plateau_min_lr=None,
    cosine_t_max=None,
    cosine_anneal_eta_min=None,
):
    """
    Return a flat dict of the active scheduler's parameters (for training reports).
    """
    if scheduler_name == "CosineAnnealingWarmRestarts":
        return {"T_0": cosine_t_0, "T_mult": cosine_t_mult, "eta_min": cosine_eta_min}
    elif scheduler_name == "StepLR":
        return {"step_size": step_size, "gamma": step_gamma}
    elif scheduler_name == "ExponentialLR":
        return {"gamma": exp_gamma}
    elif scheduler_name == "ReduceLROnPlateau":
        return {"factor": plateau_factor, "patience": plateau_patience, "min_lr": plateau_min_lr}
    elif scheduler_name == "CosineAnnealingLR":
        return {"T_max": cosine_t_max, "eta_min": cosine_anneal_eta_min}
    return {}


# ── Checkpoint Saving ─────────────────────────────────────────────────────────

def save_checkpoint(
    model,
    optimizer,
    best_epoch,
    best_val_acc,
    model_arch,
    num_classes,
    class_names,
    history,
    use_progressive_unfreeze,
    unfreeze_schedule,
    save_dir,
    dataset_name,
):
    """
    Save a model checkpoint to disk with metadata.

    Returns
    -------
    (model_name, model_path)
    """
    os.makedirs(save_dir, exist_ok=True)
    model_name = f"{dataset_name}_{model_arch}_E{best_epoch}_VAL{best_val_acc * 100:.2f}.pth"
    model_path = os.path.join(save_dir, model_name)

    torch.save({
        "epoch":                best_epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc":              best_val_acc,
        "model_arch":           model_arch,
        "num_classes":          num_classes,
        "class_names":          class_names,
        "history":              history,
        "progressive_unfreeze": use_progressive_unfreeze,
        "unfreeze_schedule":    unfreeze_schedule if use_progressive_unfreeze else None,
    }, model_path)

    print(f"\n\U0001f4be Model saved: {model_name}")
    print(f"   Path:              {model_path}")
    print(f"   Best Epoch:        {best_epoch}")
    print(f"   Val Accuracy:      {best_val_acc * 100:.2f}%")
    return model_name, model_path
