"""Slide 4 — the four MRI modalities + segmentation, one representative case.

Picks a patient that has all three sub-regions present and a mid-range tumor
size (representative, not cherry-picked), takes the axial slice with the
largest tumor area, and renders T1 | T1CE | T2 | FLAIR | + segmentation.

Region colors match docs/report_figures/nested_regions.png:
  NCR (label 1) amber · ED (label 2) blue · ET (label 3) red.

Output: docs/report_figures/modality_panel.png  (+ .pdf)
"""
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)

MOD_NAMES = ["T1", "T1CE", "T2", "FLAIR"]
# label -> RGBA overlay; 0 = transparent background
SEG_COLORS = [(0, 0, 0, 0),          # 0 background
              (0.13, 0.70, 0.30, 1), # 1 NCR  (green)
              (0.15, 0.40, 0.85, 1), # 2 ED   (blue)
              (0.86, 0.16, 0.16, 1)] # 3 ET   (red)
SEG_CMAP = ListedColormap(SEG_COLORS)
SEG_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], SEG_CMAP.N)


def pick_case(max_scan=60):
    """Return (folder, slice_idx) for a representative case."""
    cands = sorted(glob.glob(str(ROOT / "data" / "BraTS2021_*")))[:max_scan]
    best = None
    for d in cands:
        try:
            m = np.load(Path(d) / "mask.npy", mmap_mode="r")
        except Exception:
            continue
        m = np.asarray(m)
        labs = np.unique(m)
        if not {1, 2, 3}.issubset(set(labs.tolist())):
            continue                       # need all three sub-regions
        tumor_frac = (m > 0).mean()
        # representative band: not the tiniest, not a whole-hemisphere tumor
        if not (0.012 <= tumor_frac <= 0.045):
            continue
        # axial slice (last axis) with the most tumor voxels
        per_slice = (m > 0).sum(axis=(0, 1))
        z = int(per_slice.argmax())
        et_on_slice = int((m[:, :, z] == 3).sum())
        if et_on_slice < 80:               # ET must be clearly visible
            continue
        score = et_on_slice
        if best is None or score > best[0]:
            best = (score, d, z, tumor_frac)
    if best is None:                       # relax: just take first valid
        d = cands[0]
        m = np.asarray(np.load(Path(d) / "mask.npy"))
        z = int((m > 0).sum(axis=(0, 1)).argmax())
        return d, z
    return best[1], best[2]


def norm_disp(img2d):
    """Percentile-clip a (normalised) modality slice to [0,1] for display."""
    fg = img2d[img2d != 0]
    if fg.size == 0:
        return np.zeros_like(img2d)
    lo, hi = np.percentile(fg, (1, 99))
    return np.clip((img2d - lo) / (hi - lo + 1e-6), 0, 1)


def main():
    folder, z = pick_case()
    pid = Path(folder).name
    img = np.asarray(np.load(Path(folder) / "image.npy")).astype(np.float32)
    mask = np.asarray(np.load(Path(folder) / "mask.npy"))

    # rotate so the brain is upright; show the chosen axial slice
    def slc(a):
        return np.rot90(a[:, :, z])

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
    for i, name in enumerate(MOD_NAMES):
        axes[i].imshow(norm_disp(slc(img[i])), cmap="gray")
        axes[i].set_title(name, fontsize=17, fontweight="bold", pad=8)
        axes[i].axis("off")

    # 5th panel: T1CE (channel 1) with the segmentation overlay
    axes[4].imshow(norm_disp(slc(img[1])), cmap="gray")
    axes[4].imshow(slc(mask), cmap=SEG_CMAP, norm=SEG_NORM,
                   interpolation="nearest", alpha=0.55)
    axes[4].set_title("Ground-truth segmentation", fontsize=17,
                      fontweight="bold", pad=8)
    axes[4].axis("off")

    # legend strip under the overlay panel
    handles = [plt.Line2D([0], [0], marker="s", linestyle="",
                           markersize=13, markerfacecolor=c, markeredgecolor="none",
                           label=l)
               for c, l in [(SEG_COLORS[3], "ET — enhancing tumor"),
                            (SEG_COLORS[1], "NCR — necrotic core"),
                            (SEG_COLORS[2], "ED — edema")]]
    fig.suptitle(f"Same axial slice, four modalities — case {pid}",
                 fontsize=13, color="#555555", y=0.99)
    # reserve margins so the legend sits in clear space below the panels
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.16,
                        wspace=0.05)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=13, bbox_to_anchor=(0.5, 0.0))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"modality_panel.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT / 'modality_panel.png'}  (case {pid}, slice {z})")


if __name__ == "__main__":
    main()
