"""
Training Visualizer
===================
Generates gradient descent visualizations showing training progress
across epochs with marked best-model and last-epoch points.

Layout:
  Left  (2D) — Epoch vs Loss with LR overlay (left → right = epoch 1 → N)
  Right (3D) — 3D trajectory: X=Epoch, Y=Train Loss, Z=Val Loss
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection


def plot_gradient_descent(history, best_epoch, save_path=None):
    """
    Visualise training progress as a 2-panel gradient descent figure.

    Left panel  — 2D loss descent over epochs (left to right = time)
    Right panel — 3D trajectory through (Epoch, Train Loss, Val Loss) space

    Parameters
    ----------
    history : dict
        Keys used: train_loss, val_loss, learning_rates, epoch_times.
    best_epoch : int
        1-indexed epoch of the saved best model.
    save_path : str or None
        Path to save the figure as .png.

    Returns
    -------
    matplotlib.figure.Figure
    """
    train_loss = np.array(history["train_loss"])
    val_loss   = np.array(history["val_loss"])
    lrs        = np.array(history["learning_rates"])
    epochs     = np.arange(1, len(train_loss) + 1)
    total_epochs = len(train_loss)
    best_idx   = best_epoch - 1  # 0-indexed

    # Colour palette: warm (start) → cool (end) — left to right on all plots
    seg_colors = plt.cm.plasma(np.linspace(0.05, 0.95, total_epochs))

    fig = plt.figure(figsize=(22, 9))
    gs  = fig.add_gridspec(1, 5, wspace=0.38)
    ax2d = fig.add_subplot(gs[0, :2])
    ax3d = fig.add_subplot(gs[0, 2:], projection="3d")

    # ══════════════════════════════════════════════════════════════════
    # LEFT PANEL — 2D Loss Descent (Epoch on X, left → right)
    # ══════════════════════════════════════════════════════════════════
    ax2d.plot(
        epochs, train_loss, "o-",
        color="#2196F3", linewidth=2, markersize=5,
        label="Train Loss", alpha=0.85, zorder=3,
    )
    ax2d.plot(
        epochs, val_loss, "s-",
        color="#FF5722", linewidth=2, markersize=5,
        label="Val Loss", alpha=0.85, zorder=3,
    )

    # Generalization gap
    ax2d.fill_between(
        epochs, train_loss, val_loss,
        alpha=0.07, color="gray", label="Generalization Gap",
    )

    # Best epoch (saved model)
    ax2d.scatter(
        [best_epoch], [val_loss[best_idx]],
        marker="*", s=380, color="gold", edgecolors="black",
        linewidth=1.5, zorder=6,
        label=f"Best Model — Epoch {best_epoch}",
    )
    ax2d.axvline(x=best_epoch, color="gold", linestyle="--", alpha=0.4, linewidth=1.2)
    ax2d.annotate(
        f"E{best_epoch} \u2b50",
        xy=(best_epoch, val_loss[best_idx]),
        xytext=(best_epoch + 0.4, val_loss[best_idx]),
        fontsize=9, fontweight="bold", color="darkgoldenrod",
    )

    # Last epoch marker (if different from best)
    if best_idx != total_epochs - 1:
        ax2d.scatter(
            [total_epochs], [val_loss[-1]],
            marker="X", s=190, color="#E53935", edgecolors="black",
            linewidth=1.5, zorder=6,
            label=f"Last Epoch — {total_epochs}",
        )
        ax2d.annotate(
            f"E{total_epochs}",
            xy=(total_epochs, val_loss[-1]),
            xytext=(total_epochs - 1.2, val_loss[-1] + 0.01),
            fontsize=9, fontweight="bold", color="#E53935",
        )

    # Learning rate on secondary axis
    ax2d_lr = ax2d.twinx()
    ax2d_lr.plot(epochs, lrs, "--", color="purple", alpha=0.30, linewidth=1)
    ax2d_lr.set_ylabel("Learning Rate", color="purple", fontsize=10)
    ax2d_lr.tick_params(axis="y", labelcolor="purple", labelsize=8)
    ax2d_lr.set_yscale("log")

    ax2d.set_xlabel("Epoch  (left → right = start → end)", fontsize=11)
    ax2d.set_ylabel("Loss", fontsize=11)
    ax2d.set_title("Loss Descent Over Epochs", fontsize=13, fontweight="bold")
    ax2d.legend(loc="upper right", fontsize=8.5)
    ax2d.grid(True, alpha=0.3)
    ax2d.set_xlim(0.5, total_epochs + 0.5)

    # ══════════════════════════════════════════════════════════════════
    # RIGHT PANEL — 3D Gradient Descent Trajectory
    #   X = Epoch  (left to right = time progression)
    #   Y = Train Loss
    #   Z = Val Loss
    # ══════════════════════════════════════════════════════════════════

    # ── Floor / wall reference planes ─────────────────────────────────
    z_floor  = val_loss.min()   - (val_loss.max()   - val_loss.min())  * 0.18
    y_back   = train_loss.max() + (train_loss.max() - train_loss.min()) * 0.15

    # Shadow: project path onto the Z-floor (epoch × train_loss plane)
    ax3d.plot(
        epochs, train_loss,
        zs=z_floor, zdir="z",
        color="darkgray", alpha=0.25, linewidth=1.2, linestyle="-",
    )
    # Shadow: project path onto the Y-back wall (epoch × val_loss plane)
    ax3d.plot(
        epochs, val_loss,
        zs=y_back, zdir="y",
        color="darkgray", alpha=0.25, linewidth=1.2, linestyle="-",
    )

    # ── Coloured path segments (plasma: dark-purple → yellow, left → right) ──
    for i in range(total_epochs - 1):
        ax3d.plot(
            [epochs[i],     epochs[i + 1]],
            [train_loss[i], train_loss[i + 1]],
            [val_loss[i],   val_loss[i + 1]],
            color=seg_colors[i], linewidth=2.5, alpha=0.9,
        )

    # ── Scatter points coloured by epoch ──────────────────────────────
    sc = ax3d.scatter(
        epochs, train_loss, val_loss,
        c=epochs, cmap="plasma",
        vmin=1, vmax=total_epochs,
        s=45, edgecolors="black", linewidth=0.4,
        depthshade=True, zorder=4,
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(
        cmap="plasma",
        norm=plt.Normalize(vmin=1, vmax=total_epochs),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax3d, label="Epoch", shrink=0.45, pad=0.12)

    # ── Start marker (Epoch 1) ─────────────────────────────────────────
    ax3d.scatter(
        [epochs[0]], [train_loss[0]], [val_loss[0]],
        marker="o", s=220, color="#4CAF50",
        edgecolors="black", linewidth=2, zorder=7,
        label="Start (Epoch 1)",
    )
    ax3d.text(
        epochs[0], train_loss[0], val_loss[0] + (val_loss.max() - val_loss.min()) * 0.05,
        "E1", fontsize=9, fontweight="bold", color="#2E7D32",
    )

    # ── Best epoch marker (saved model) ───────────────────────────────
    ax3d.scatter(
        [epochs[best_idx]], [train_loss[best_idx]], [val_loss[best_idx]],
        marker="*", s=500, color="gold",
        edgecolors="black", linewidth=1.5, zorder=8,
        label=f"Best Model (Epoch {best_epoch})",
    )
    ax3d.text(
        epochs[best_idx],
        train_loss[best_idx],
        val_loss[best_idx] + (val_loss.max() - val_loss.min()) * 0.06,
        f"E{best_epoch} \u2b50",
        fontsize=9, fontweight="bold", color="darkgoldenrod",
    )

    # ── Last epoch marker ──────────────────────────────────────────────
    if best_idx != total_epochs - 1:
        ax3d.scatter(
            [epochs[-1]], [train_loss[-1]], [val_loss[-1]],
            marker="X", s=240, color="#E53935",
            edgecolors="black", linewidth=2, zorder=8,
            label=f"Last Epoch ({total_epochs})",
        )
        ax3d.text(
            epochs[-1],
            train_loss[-1],
            val_loss[-1] + (val_loss.max() - val_loss.min()) * 0.05,
            f"E{total_epochs}",
            fontsize=9, fontweight="bold", color="#E53935",
        )

    ax3d.set_xlabel("Epoch", fontsize=10, labelpad=8)
    ax3d.set_ylabel("Train Loss", fontsize=10, labelpad=8)
    ax3d.set_zlabel("Val Loss", fontsize=10, labelpad=8)
    ax3d.set_title(
        "3D Gradient Descent Trajectory\n"
        r"Epoch $\rightarrow$ left to right",
        fontsize=13, fontweight="bold",
    )
    ax3d.legend(loc="upper right", fontsize=8, framealpha=0.7)

    # View angle: epoch axis goes clearly left to right
    ax3d.view_init(elev=22, azim=-55)

    # ══════════════════════════════════════════════════════════════════
    fig.suptitle(
        "Gradient Descent Visualization",
        fontsize=17, fontweight="bold", y=1.01,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\u2705 Gradient descent visualization saved: {save_path}")

    plt.show()
    return fig
