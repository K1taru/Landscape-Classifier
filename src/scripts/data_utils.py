"""
Data Utilities
==============
Transforms, dataset loading/splitting, class weight computation,
and DataLoader creation for the Landscape Classifier pipeline.
"""

import torch
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
import numpy as np
from collections import Counter


# ── Dataset Wrapper ───────────────────────────────────────────────────────────

class TransformSubset(Dataset):
    """Wraps a torch Subset with a specific transform pipeline."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)


# ── Transforms ────────────────────────────────────────────────────────────────

def build_transforms(
    img_height,
    img_width,
    normalize_mean,
    normalize_std,
    use_augmentation=True,
    augmentation_options=None,
    rotation_degrees=20,
    affine_translate=(0.1, 0.1),
    affine_scale=(0.85, 1.15),
    color_brightness=0.3,
    color_contrast=0.3,
    color_saturation=0.3,
    color_hue=0.1,
    perspective_distortion=0.2,
    perspective_prob=0.5,
    resized_crop_scale=(0.8, 1.0),
    gaussian_blur_kernel=(3, 3),
    gaussian_blur_prob=0.3,
    erasing_prob=0.1,
    erasing_scale=(0.02, 0.1),
):
    """
    Build train and val/test transform pipelines.

    Returns
    -------
    train_transforms : transforms.Compose
    val_test_transforms : transforms.Compose
    applied_augmentations : list[str]
        Human-readable descriptions of all active augmentations.
    """
    if augmentation_options is None:
        augmentation_options = {}

    val_test_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ])

    if not use_augmentation:
        train_transforms = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ])
        print("\u26a0\ufe0f  Training augmentation DISABLED")
        print(f"\n\u2705 Transforms defined | Size: {img_height}x{img_width} | "
              f"Norm mean={normalize_mean}, std={normalize_std}")
        return train_transforms, val_test_transforms, []

    aug_list = []
    applied  = []

    if augmentation_options.get("random_resized_crop", False):
        aug_list.append(transforms.RandomResizedCrop((img_height, img_width), scale=resized_crop_scale))
        applied.append(f"Random Resized Crop: scale={resized_crop_scale}")
    else:
        aug_list.append(transforms.Resize((img_height, img_width)))

    if augmentation_options.get("random_horizontal_flip", False):
        aug_list.append(transforms.RandomHorizontalFlip(p=0.5))
        applied.append("Random Horizontal Flip: p=0.5")

    if augmentation_options.get("random_rotation", False):
        aug_list.append(transforms.RandomRotation(rotation_degrees))
        applied.append(f"Random Rotation: \u00b1{rotation_degrees}\u00b0")

    if augmentation_options.get("random_affine", False):
        aug_list.append(transforms.RandomAffine(
            degrees=0, translate=affine_translate, scale=affine_scale,
        ))
        applied.append(f"Random Affine: translate={affine_translate}, scale={affine_scale}")

    if augmentation_options.get("color_jitter", False):
        aug_list.append(transforms.ColorJitter(
            brightness=color_brightness, contrast=color_contrast,
            saturation=color_saturation, hue=color_hue,
        ))
        applied.append(
            f"Color Jitter: brightness={color_brightness}, contrast={color_contrast}, "
            f"saturation={color_saturation}, hue={color_hue}"
        )

    if augmentation_options.get("random_perspective", False):
        aug_list.append(transforms.RandomPerspective(
            distortion_scale=perspective_distortion, p=perspective_prob,
        ))
        applied.append(f"Random Perspective: distortion={perspective_distortion}, p={perspective_prob}")

    if augmentation_options.get("gaussian_blur", False):
        aug_list.append(transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=gaussian_blur_kernel)], p=gaussian_blur_prob,
        ))
        applied.append(f"Gaussian Blur: kernel={gaussian_blur_kernel}, p={gaussian_blur_prob}")

    # ToTensor + Normalize always before RandomErasing (which operates on tensors)
    aug_list.append(transforms.ToTensor())
    aug_list.append(transforms.Normalize(mean=normalize_mean, std=normalize_std))

    if augmentation_options.get("random_erasing", False):
        aug_list.append(transforms.RandomErasing(p=erasing_prob, scale=erasing_scale))
        applied.append(f"Random Erasing: p={erasing_prob}, scale={erasing_scale}")

    train_transforms = transforms.Compose(aug_list)

    print("\u2705 Training augmentation ENABLED")
    print(f"   Active augmentations ({len(applied)}):")
    for a in applied:
        print(f"   - {a}")
    print(f"\n\u2705 Transforms defined | Size: {img_height}x{img_width} | "
          f"Norm mean={normalize_mean}, std={normalize_std}")

    return train_transforms, val_test_transforms, applied


# ── Dataset Loading & Splitting ───────────────────────────────────────────────

def load_and_split_dataset(
    dataset_dir,
    train_split,
    val_split,
    test_split,
    train_transforms,
    val_test_transforms,
    seed=42,
):
    """
    Load an ImageFolder dataset and split into train / val / test subsets.

    Returns
    -------
    dict with keys:
        full_dataset, train_subset, val_subset, test_subset,
        train_dataset, val_dataset, test_dataset,
        num_classes, class_names,
        train_size, val_size, test_size
    """
    print("\U0001f4c2 Loading landscape dataset...")
    full_dataset = datasets.ImageFolder(root=dataset_dir)

    print(f"\n\U0001f4ca Class to index mapping:")
    for cls_name, cls_idx in sorted(full_dataset.class_to_idx.items(), key=lambda x: x[1]):
        print(f"   {cls_idx}: {cls_name}")

    num_classes  = len(full_dataset.classes)
    class_names  = sorted(full_dataset.classes)
    total_samples = len(full_dataset)

    train_size = int(train_split * total_samples)
    val_size   = int(val_split   * total_samples)
    test_size  = total_samples - train_size - val_size

    print(f"\n\u2705 Dataset loaded: {total_samples:,} total samples, {num_classes} classes")

    train_subset, val_subset, test_subset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    print(f"\n\u2705 Data split completed:")
    print(f"   \U0001f539 Train:      {train_size:,} samples ({train_split*100:.0f}%)")
    print(f"   \U0001f539 Validation: {val_size:,} samples ({val_split*100:.0f}%)")
    print(f"   \U0001f539 Test:       {test_size:,} samples ({test_split*100:.0f}%)")

    train_dataset = TransformSubset(train_subset, train_transforms)
    val_dataset   = TransformSubset(val_subset,   val_test_transforms)
    test_dataset  = TransformSubset(test_subset,  val_test_transforms)

    print(f"\n\u2705 Transforms assigned (Train: augmented | Val/Test: no augmentation)")

    return {
        "full_dataset":  full_dataset,
        "train_subset":  train_subset,
        "val_subset":    val_subset,
        "test_subset":   test_subset,
        "train_dataset": train_dataset,
        "val_dataset":   val_dataset,
        "test_dataset":  test_dataset,
        "num_classes":   num_classes,
        "class_names":   class_names,
        "train_size":    train_size,
        "val_size":      val_size,
        "test_size":     test_size,
    }


# ── Class Weights ─────────────────────────────────────────────────────────────

def compute_class_weights(
    full_dataset,
    train_subset,
    num_classes,
    class_names,
    use_weighted_sampler=False,
):
    """
    Compute inverse-frequency class weights and optional per-sample weights.

    Returns
    -------
    dict with keys:
        class_weights       np.ndarray [num_classes]
        sample_weights      list[float] or None
        train_class_counts  np.ndarray [num_classes]
    """
    all_targets  = [full_dataset.targets[i] for i in train_subset.indices]
    counts       = Counter(all_targets)
    counts_array = np.array([counts[i] for i in range(num_classes)])

    print(f"\U0001f4ca Training set class distribution:")
    for i, cn in enumerate(class_names):
        print(f"   {cn}: {counts_array[i]:,} samples")
    print(f"   Most populated:  {class_names[np.argmax(counts_array)]} ({counts_array.max():,})")
    print(f"   Least populated: {class_names[np.argmin(counts_array)]} ({counts_array.min():,})")
    print(f"   Imbalance ratio: {counts_array.max() / counts_array.min():.2f}x")

    # Inverse-frequency weights, normalised so they average to 1.0
    class_weights  = 1.0 / counts_array.astype(np.float64)
    class_weights  = class_weights / class_weights.sum() * num_classes

    print(f"\n\U0001f4d0 Class weights (inverse frequency, avg=1.0):")
    for i, cn in enumerate(class_names):
        print(f"   {cn}: {class_weights[i]:.4f}")
    print(f"   Min/Max: {class_weights.min():.4f} / {class_weights.max():.4f}  "
          f"(ratio {class_weights.max() / class_weights.min():.2f}x)")

    sample_weights = None
    if use_weighted_sampler:
        sample_weights = [class_weights[label] for label in all_targets]
        print(f"\n\u2705 Per-sample weights ready for WeightedRandomSampler ({len(sample_weights):,} samples)")
    else:
        print("\n\u26a0\ufe0f  Weighted sampler DISABLED")

    return {
        "class_weights":      class_weights,
        "sample_weights":     sample_weights,
        "train_class_counts": counts_array,
    }


# ── DataLoaders ───────────────────────────────────────────────────────────────

def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size,
    use_weighted_sampler=False,
    sample_weights=None,
):
    """
    Create DataLoader instances for train, validation, and test sets.

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    pin = torch.cuda.is_available()

    if use_weighted_sampler and sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=0, pin_memory=pin,
        )
        print(f"\u2705 Train DataLoader: {len(train_dataset):,} samples with WeightedRandomSampler")
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=pin,
        )
        print(f"\u2705 Train DataLoader: {len(train_dataset):,} samples with shuffle=True")

    val_loader  = DataLoader(val_dataset,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    print(f"\u2705 Validation DataLoader: {len(val_dataset):,} samples")
    print(f"\u2705 Test DataLoader:       {len(test_dataset):,} samples")
    print(f"\n\U0001f4e6 Batch config: size={batch_size} | "
          f"train={len(train_loader)} | val={len(val_loader)} | test={len(test_loader)} batches")

    return train_loader, val_loader, test_loader
