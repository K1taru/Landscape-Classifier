"""Gradio inference app for Landscape-Classifier.

Features:
- Select any checkpoint from App-Inference/models/*.pth
- Upload image files or paste images from clipboard
- View top-k class probabilities
- Mobile-friendly responsive UI
- Interactive model browser with metadata
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration Constants
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "App-Inference" / "models"

# Image preprocessing
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

# Model configuration
MODEL_DROPOUT = 0.4
DEFAULT_TOP_K = 3
MAX_TOP_K = 10
INFO_BOX_LINES = 5

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Checkpoint Discovery
# ============================================================================

def discover_checkpoints() -> List[Path]:
    """Return available .pth checkpoint files sorted by name and date."""
    if not MODELS_DIR.exists():
        return []
    return sorted(MODELS_DIR.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)


def get_checkpoint_info(path: Path) -> Dict[str, str]:
    """Get human-readable information about a checkpoint."""
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        return {
            "name": path.name,
            "size": f"{size_mb:.1f} MB",
            "short_name": path.stem,
        }
    except (FileNotFoundError, OSError):
        return {"name": path.name, "size": "unknown", "short_name": path.stem}


def build_inference_model(num_classes: int, dropout_rate: float = MODEL_DROPOUT) -> nn.Module:
    """Create the model architecture used during training."""
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, num_classes),
    )
    return model


@lru_cache(maxsize=4)
def load_model_cached(checkpoint_path: str, mtime: float) -> Tuple[nn.Module, List[str], str]:
    """Load and cache a checkpoint by path and file mtime.

    Parameters
    ----------
    checkpoint_path : str
        Full path to the checkpoint file.
    mtime : float
        File modification time (used as cache invalidation key).

    Returns
    -------
    (model, class_names, model_arch)
    """
    _ = mtime  # cache key includes mtime so updated files reload automatically

    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    except (FileNotFoundError, OSError) as e:
        raise gr.Error(f"Failed to load checkpoint: {e}")

    class_names = checkpoint.get("class_names")
    num_classes = checkpoint.get("num_classes")

    if class_names is None and num_classes is None:
        raise gr.Error(
            "Checkpoint missing both 'class_names' and 'num_classes'. "
            "Cannot construct classifier head."
        )

    if class_names is None:
        class_names = [f"class_{i}" for i in range(int(num_classes))]

    num_classes = len(class_names)

    model = build_inference_model(num_classes=num_classes)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except KeyError as e:
        raise gr.Error(f"Invalid checkpoint format: missing 'model_state_dict': {e}")
    except RuntimeError as e:
        raise gr.Error(f"Failed to load model weights: {e}")

    model.to(DEVICE)
    model.eval()

    model_arch = str(checkpoint.get("model_arch", "resnet50"))
    return model, class_names, model_arch


@lru_cache(maxsize=1)
def get_transform() -> transforms.Compose:
    """Return deterministic preprocessing pipeline for inference.

    This is cached to avoid recreating the same transformation object
    on every inference request.
    """
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _validate_and_load_checkpoint(checkpoint_name: str) -> Tuple[Path, float]:
    """Validate checkpoint name and get file mtime.

    Combines check + stat into single I/O operation (fixes TOCTOU anti-pattern).
    """
    checkpoint_path = MODELS_DIR / checkpoint_name
    try:
        mtime = checkpoint_path.stat().st_mtime
        return checkpoint_path, mtime
    except FileNotFoundError:
        raise gr.Error(f"Checkpoint not found: {checkpoint_name}")
    except OSError as e:
        raise gr.Error(f"Cannot access checkpoint: {e}")


def predict_image(
    image: Image.Image | None, checkpoint_name: str, top_k: int
) -> Tuple[Dict[str, float], str]:
    """Run inference and return top-k class probabilities."""
    if image is None:
        raise gr.Error("Please upload or paste an image first.")

    checkpoint_path, mtime = _validate_and_load_checkpoint(checkpoint_name)

    try:
        model, class_names, model_arch = load_model_cached(str(checkpoint_path), mtime)
    except gr.Error:
        raise
    except Exception as exc:
        logger.exception("Unexpected error loading model")
        raise gr.Error(f"Failed to load model: {type(exc).__name__}")

    transform = get_transform()

    # Avoid redundant RGB conversion if image is already RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    k = max(1, min(int(top_k), len(class_names)))
    top_probs, top_indices = torch.topk(probs, k=k)

    results: Dict[str, float] = {}
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        results[class_names[idx]] = float(prob)

    summary = (
        f"✓ Model: {checkpoint_name}\n"
        f"Architecture: {model_arch}\n"
        f"Device: {DEVICE.type.upper()}\n"
        f"Predictions: {k}/{len(class_names)} classes"
    )

    return results, summary


def model_info(checkpoint_name: str) -> str:
    """Display metadata for the selected checkpoint."""
    if not checkpoint_name:
        return "No checkpoint selected."

    try:
        checkpoint_path, mtime = _validate_and_load_checkpoint(checkpoint_name)
        _, class_names, model_arch = load_model_cached(str(checkpoint_path), mtime)
    except gr.Error as e:
        return str(e)
    except Exception as exc:
        logger.exception("Unexpected error loading checkpoint metadata")
        return f"Failed to load checkpoint metadata: {type(exc).__name__}"

    model_info_text = (
        f"📋 **Checkpoint:** {checkpoint_name}\n"
        f"🏗️ **Architecture:** {model_arch}\n"
        f"🏷️ **Classes ({len(class_names)}):**\n"
        f"{', '.join(class_names)}"
    )
    return model_info_text


def build_app() -> gr.Blocks:
    """Build the Gradio app with responsive mobile-friendly UI."""
    checkpoints = discover_checkpoints()
    checkpoint_names = [p.name for p in checkpoints]

    with gr.Blocks(title="Landscape Classifier Inference") as demo:
        # Header
        gr.Markdown(
            """
            # 🌄 Landscape Classifier Inference

            Upload an image to classify landscapes: normal, fire, flooded areas, collapsed buildings, or traffic incidents.
            """
        )

        if not checkpoint_names:
            gr.Error(
                "❌ No checkpoints found in App-Inference/models/. "
                "Add a .pth file or run training first."
            )
            return demo

        # Main content
        with gr.Row(equal_height=False):
            # Left column: Input (responsive)
            with gr.Column(scale=1, min_width=250):
                gr.Markdown("### 📸 Input Image")
                image_input = gr.Image(
                    type="pil",
                    sources=["upload", "clipboard"],
                    label="Upload or paste image",
                    interactive=True,
                )

            # Right column: Results (responsive)
            with gr.Column(scale=1, min_width=250):
                gr.Markdown("### 🎯 Predictions")
                label_output = gr.Label(
                    label="Classification Results",
                    show_label=False,
                )

        # Model selection section
        with gr.Group(elem_classes="info-panel"):
            gr.Markdown("### ⚙️ Model Settings")

            with gr.Row():
                checkpoint_dropdown = gr.Dropdown(
                    choices=checkpoint_names,
                    value=checkpoint_names[0] if checkpoint_names else None,
                    label="Select Model",
                    interactive=True,
                    scale=2,
                )
                refresh_btn = gr.Button("🔄 Refresh", scale=1)

            with gr.Row():
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=MAX_TOP_K,
                    value=DEFAULT_TOP_K,
                    step=1,
                    label="Top-K Predictions",
                    interactive=True,
                    scale=1,
                )

        # Model information panel
        info_box = gr.Markdown(value="No model selected")

        # Action buttons
        with gr.Row():
            run_btn = gr.Button("▶️ Run Inference", variant="primary", scale=2)
            clear_btn = gr.Button("🗑️ Clear", scale=1)

        # Model browser section (collapsible on mobile)
        with gr.Accordion("📚 Available Models", open=False):
            model_browser = gr.Markdown()

        # Define button callbacks
        def refresh_checkpoints() -> tuple[list, str]:
            """Refresh checkpoint list and update UI."""
            checkpoints = discover_checkpoints()
            names = [p.name for p in checkpoints]
            if not names:
                return [], "No models found"
            browser_md = _format_model_browser(checkpoints)
            return names, browser_md

        def update_browser(selected_model: str) -> str:
            """Update model browser display."""
            checkpoints = discover_checkpoints()
            return _format_model_browser(checkpoints, selected_model)

        def clear_inputs():
            """Clear all inputs."""
            return None, 0, ""

        # Event handlers
        run_btn.click(
            fn=predict_image,
            inputs=[image_input, checkpoint_dropdown, top_k_slider],
            outputs=[label_output, info_box],
        )

        refresh_btn.click(
            fn=refresh_checkpoints,
            outputs=[checkpoint_dropdown, model_browser],
        )

        checkpoint_dropdown.change(
            fn=update_browser,
            inputs=[checkpoint_dropdown],
            outputs=[model_browser],
        )

        checkpoint_dropdown.change(
            fn=model_info,
            inputs=[checkpoint_dropdown],
            outputs=[info_box],
        )

        clear_btn.click(
            fn=clear_inputs,
            outputs=[image_input, top_k_slider, label_output],
        )

        demo.load(
            fn=update_browser,
            inputs=[checkpoint_dropdown],
            outputs=[model_browser],
        )

        demo.load(
            fn=model_info,
            inputs=[checkpoint_dropdown],
            outputs=[info_box],
        )

    return demo


def _format_model_browser(
    checkpoints: List[Path], selected: str | None = None
) -> str:
    """Format model list for display."""
    if not checkpoints:
        return "No models available"

    lines = ["| Model | Size | Status |", "|-------|------|--------|"]
    for path in checkpoints:
        try:
            size_mb = path.stat().st_size / (1024 * 1024)
            size_str = f"{size_mb:.1f} MB"
        except (FileNotFoundError, OSError):
            size_str = "unknown"

        status = "✅ Selected" if selected and path.name == selected else "⭕ Available"
        lines.append(f"| {path.name} | {size_str} | {status} |")

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Show startup info
    print("\n" + "=" * 60)
    print("🚀 Starting Landscape Classifier Inference App")
    print("=" * 60)
    print(f"📁 Models directory: {MODELS_DIR}")
    print(f"🎯 Device: {DEVICE}")

    # Check for models
    checkpoints = discover_checkpoints()
    print(f"📊 Found {len(checkpoints)} model(s)")
    for cp in checkpoints:
        size_mb = cp.stat().st_size / (1024 * 1024)
        print(f"   - {cp.name} ({size_mb:.1f} MB)")

    if not checkpoints:
        print("\n⚠️  WARNING: No models found in App-Inference/models/")

    print("=" * 60 + "\n")

    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
        css="""
        @media (max-width: 768px) {
            .model-list { max-height: 200px !important; }
            .input-row { flex-direction: column; }
            .info-panel { padding: 1rem 0.5rem; }
        }
        """,
    )
