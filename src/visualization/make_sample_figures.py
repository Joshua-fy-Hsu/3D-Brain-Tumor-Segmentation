"""Generate sample visualization figures for the project report.

Picks a few patients from the BraTS 2021 preprocessed dataset, finds an axial
slice that contains all three tumor sub-regions (NCR, ED, ET), and saves:

  1) Per-patient figure: 4 MRI modalities + segmentation mask side-by-side.
  2) Per-class figure: zoomed slices that emphasize each individual class.

Output: docs/report_figures/
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = r"D:/University/Projects/Brain_Tumor_Segmentation/data/BraTS2021_Optimized"
OUT  = r"D:/University/Projects/Brain_Tumor_Segmentation/docs/report_figures"
os.makedirs(OUT, exist_ok=True)

# Color map for the 4 classes:
#   0 = Background (transparent), 1 = NCR (red), 2 = ED (green), 3 = ET (blue)
SEG_CMAP = ListedColormap([(0, 0, 0, 0),
                           (0.90, 0.10, 0.10, 0.85),
                           (0.10, 0.80, 0.20, 0.85),
                           (0.10, 0.30, 0.95, 0.85)])
SEG_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], SEG_CMAP.N)
CLASS_NAMES = {1: "NCR", 2: "ED", 3: "ET"}


def best_axial_slice(mask):
    """Return axial slice index that contains the most balanced mix of NCR/ED/ET."""
    z_dim = mask.shape[-1]
    best_z, best_score = z_dim // 2, -1
    for z in range(z_dim):
        s = mask[..., z]
        c1 = (s == 1).sum()
        c2 = (s == 2).sum()
        c3 = (s == 3).sum()
        if c1 == 0 or c2 == 0 or c3 == 0:
            continue
        score = min(c1, c2, c3) + 0.1 * (c1 + c2 + c3)
        if score > best_score:
            best_score, best_z = score, z
    return best_z


def slice_with_class(mask, cls):
    """Return axial slice with the largest area of `cls`."""
    counts = (mask == cls).reshape(-1, mask.shape[-1]).sum(axis=0)
    return int(np.argmax(counts))


def normalize_for_display(img):
    """Percentile-normalize a 2D MRI slice for nicer display."""
    lo, hi = np.percentile(img[img > 0], [1, 99]) if (img > 0).any() else (0, 1)
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    return img


def render_patient_panel(pid, save_path):
    """4 modalities + segmentation overlay for one patient."""
    img = np.load(os.path.join(ROOT, pid, "image.npy"), mmap_mode="r")
    seg = np.load(os.path.join(ROOT, pid, "mask.npy"),  mmap_mode="r")
    z = best_axial_slice(seg)

    flair = normalize_for_display(img[3, :, :, z])
    t1    = normalize_for_display(img[0, :, :, z])
    t1ce  = normalize_for_display(img[1, :, :, z])
    t2    = normalize_for_display(img[2, :, :, z])
    s     = seg[:, :, z]

    # Layout: large Segmentation overlay on the left (spanning both rows),
    # and the 4 MRI modalities arranged as a 2x2 grid on the right.
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], hspace=0.12, wspace=0.05)

    ax_seg   = fig.add_subplot(gs[:, 0])
    ax_t1    = fig.add_subplot(gs[0, 1])
    ax_t1ce  = fig.add_subplot(gs[0, 2])
    ax_t2    = fig.add_subplot(gs[1, 1])
    ax_flair = fig.add_subplot(gs[1, 2])

    for ax, im, t in [(ax_t1, t1, "T1"),
                      (ax_t1ce, t1ce, "T1ce"),
                      (ax_t2, t2, "T2"),
                      (ax_flair, flair, "FLAIR")]:
        ax.imshow(np.rot90(im), cmap="gray")
        ax.set_title(t, fontsize=12)
        ax.axis("off")

    ax_seg.imshow(np.rot90(flair), cmap="gray")
    ax_seg.imshow(np.rot90(s), cmap=SEG_CMAP, norm=SEG_NORM, interpolation="nearest")
    ax_seg.set_title("Segmentation overlay", fontsize=14)
    ax_seg.axis("off")

    legend_elems = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=(0.90, 0.10, 0.10), markersize=12, label='NCR (Necrotic Core)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=(0.10, 0.80, 0.20), markersize=12, label='ED (Edema)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=(0.10, 0.30, 0.95), markersize=12, label='ET (Enhancing Tumor)'),
    ]
    fig.suptitle(f"Patient {pid} — axial slice z={z}", fontsize=13)
    fig.subplots_adjust(top=0.93, bottom=0.07)
    fig.legend(handles=legend_elems, loc='lower center', ncol=3, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.0))
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}")


def render_class_examples(patients, cls, save_path):
    """For one tumor class, show 3 example slices (FLAIR + class mask only)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, pid in zip(axes, patients):
        img = np.load(os.path.join(ROOT, pid, "image.npy"), mmap_mode="r")
        seg = np.load(os.path.join(ROOT, pid, "mask.npy"),  mmap_mode="r")
        z = slice_with_class(seg, cls)
        flair = normalize_for_display(img[3, :, :, z])
        s = (seg[:, :, z] == cls).astype(np.uint8)
        ax.imshow(np.rot90(flair), cmap="gray")
        # use the matching color from SEG_CMAP for this class
        rgba = SEG_CMAP(cls)
        overlay = np.zeros((*s.shape, 4))
        overlay[s == 1] = rgba
        ax.imshow(np.rot90(overlay), interpolation="nearest")
        ax.set_title(f"{pid}  (z={z})", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Class {cls}: {CLASS_NAMES[cls]} — highlighted on FLAIR", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}")


def main():
    patients = sorted([d for d in os.listdir(ROOT) if d.startswith("BraTS2021_")])

    # ------------------------------------------------------------------
    # 1. Pick a few patients that contain all 3 tumor classes for the
    #    full-modality panel.
    # ------------------------------------------------------------------
    chosen = []
    for pid in patients:
        seg = np.load(os.path.join(ROOT, pid, "mask.npy"), mmap_mode="r")
        u = np.unique(seg)
        if all(c in u for c in (1, 2, 3)):
            chosen.append(pid)
        if len(chosen) >= 3:
            break

    for pid in chosen:
        render_patient_panel(pid, os.path.join(OUT, f"panel_{pid}.png"))

    # ------------------------------------------------------------------
    # 2. Per-class examples: 3 patients per class.
    # ------------------------------------------------------------------
    examples = {1: [], 2: [], 3: []}
    for pid in patients:
        seg = np.load(os.path.join(ROOT, pid, "mask.npy"), mmap_mode="r")
        for c in (1, 2, 3):
            if (seg == c).sum() > 2000 and len(examples[c]) < 3 and pid not in examples[c]:
                examples[c].append(pid)
        if all(len(examples[c]) == 3 for c in (1, 2, 3)):
            break

    for cls in (1, 2, 3):
        render_class_examples(examples[cls], cls,
                              os.path.join(OUT, f"class_{cls}_{CLASS_NAMES[cls]}.png"))


if __name__ == "__main__":
    main()
