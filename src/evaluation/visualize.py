"""Plotting: uncertainty overlays, error/uncertainty side-by-side, boxplot,
reliability diagram, risk-coverage curve."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _pick_slice(label_map):
    """Slice with the most tumor voxels along axial axis (last dim)."""
    if label_map.ndim != 3:
        return label_map.shape[-1] // 2
    counts = (label_map > 0).sum(axis=(0, 1))
    return int(np.argmax(counts)) if counts.max() > 0 else label_map.shape[-1] // 2


def overlay_uncertainty(image_t1ce, pred_label, unc_map, save_path, title=""):
    """3-pane axial/coronal/sagittal overlay of uncertainty over the prediction."""
    z = _pick_slice(pred_label)
    y = pred_label.shape[1] // 2
    x = pred_label.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    views = [
        (image_t1ce[:, :, z], pred_label[:, :, z], unc_map[:, :, z], "Axial"),
        (image_t1ce[:, y, :], pred_label[:, y, :], unc_map[:, y, :], "Coronal"),
        (image_t1ce[x, :, :], pred_label[x, :, :], unc_map[x, :, :], "Sagittal"),
    ]
    for ax, (img, lab, unc, name) in zip(axes, views):
        ax.imshow(img.T, cmap="gray", origin="lower")
        ax.imshow(np.ma.masked_where(lab.T == 0, lab.T), cmap="tab10", alpha=0.4, origin="lower")
        ax.imshow(unc.T, cmap="hot", alpha=0.45, origin="lower")
        ax.set_title(f"{name}")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def error_vs_uncertainty(error_map, unc_map, save_path, title=""):
    z = _pick_slice(error_map.astype(np.uint8))
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(error_map[:, :, z].T, cmap="Reds", origin="lower")
    axes[0].set_title("Error (pred ≠ GT)"); axes[0].axis("off")
    axes[1].imshow(unc_map[:, :, z].T, cmap="hot", origin="lower")
    axes[1].set_title("Uncertainty"); axes[1].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def dice_boxplot(per_case_df, regions=("ET", "TC", "WT"), save_path="dice_boxplot.png"):
    data = [per_case_df[f"dice_{r}"].dropna().values for r in regions]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, labels=list(regions), showmeans=True)
    ax.set_ylabel("Dice")
    ax.set_title("Dice distribution by region")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def dice_boxplot_multi(method_to_df, regions=("ET", "TC", "WT"),
                       save_path="dice_boxplot_methods.png",
                       title="Per-case Dice by method"):
    """Side-by-side boxplots: one group per region, one box per method.

    method_to_df: dict mapping method label -> per_case DataFrame (already
                  filtered to one mode, e.g. only `tta_post` rows).
    """
    methods = list(method_to_df.keys())
    n_methods = len(methods)
    if n_methods == 0:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * n_methods * len(regions)), 4.5))

    # Position groups by region; within each region, one box per method
    width = 0.8 / max(n_methods, 1)
    positions = []
    boxes = []
    labels = []
    cmap = plt.cm.tab10
    for ri, region in enumerate(regions):
        for mi, method in enumerate(methods):
            df = method_to_df[method]
            col = f"dice_{region}"
            vals = df[col].dropna().values if col in df.columns else np.array([])
            pos = ri + (mi - (n_methods - 1) / 2) * width
            positions.append(pos)
            boxes.append(vals)
            labels.append(f"{method}\n{region}" if ri == 0 else region)

    bp = ax.boxplot(boxes, positions=positions, widths=width * 0.85,
                    showmeans=True, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        method_idx = i % n_methods
        patch.set_facecolor(cmap(method_idx % 10))
        patch.set_alpha(0.6)

    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(list(regions))
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    # Legend by method
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=cmap(i % 10), alpha=0.6, label=m)
                      for i, m in enumerate(methods)]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def failure_overlay(image_t1ce, pred_label, gt_label, save_path,
                    title="", region_colors=None):
    """Side-by-side: T1CE | GT overlay | Pred overlay | Error map.
    Used for the bottom-K Dice cases to flag failure modes.
    """
    if region_colors is None:
        # NCR=red, ED=green, ET=blue
        region_colors = {1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1)}

    z = _pick_slice(gt_label) if gt_label.any() else _pick_slice(pred_label)
    img = image_t1ce[z]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    def _color(label_slice):
        rgb = np.stack([img] * 3, axis=-1)
        for lbl, col in region_colors.items():
            mask = (label_slice == lbl)
            for c in range(3):
                rgb[..., c] = np.where(mask, 0.5 * rgb[..., c] + 0.5 * col[c], rgb[..., c])
        return rgb

    err = (pred_label[z] != gt_label[z]).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(img, cmap="gray");           axes[0].set_title("T1CE")
    axes[1].imshow(_color(gt_label[z]));        axes[1].set_title("GT")
    axes[2].imshow(_color(pred_label[z]));      axes[2].set_title("Pred")
    axes[3].imshow(img, cmap="gray")
    axes[3].imshow(err, alpha=0.6, cmap="Reds"); axes[3].set_title("Error")
    for a in axes:
        a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def reliability_diagram(bin_conf, bin_acc, bin_count, ece, save_path, title=""):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    width = 1.0 / max(len(bin_conf), 1)
    centers = np.arange(len(bin_conf)) * width + width / 2
    ax.bar(centers, bin_acc, width=width * 0.95, alpha=0.6, edgecolor="black", label="acc")
    ax.bar(centers, bin_conf - bin_acc, width=width * 0.95, bottom=bin_acc,
           color="red", alpha=0.3, edgecolor="red", label="gap")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}  ECE={ece:.4f}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def risk_coverage_plot(coverages, risks, aurc, save_path, title="Risk-Coverage"):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(coverages, risks, lw=2)
    ax.set_xlabel("Coverage"); ax.set_ylabel("Risk (1 − Dice)")
    ax.set_title(f"{title}  AURC={aurc:.4f}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
