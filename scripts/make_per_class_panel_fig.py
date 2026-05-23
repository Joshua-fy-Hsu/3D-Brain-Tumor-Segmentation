"""Dataset section — per-class example panel (rubric: 2-3 examples per class).

3 rows (NCR, ED, ET) x 3 columns (three different cases each). For every region
we scan a candidate pool, pick the three cases with the largest amount of that
region, and show the slice with the most of it, overlaid on FLAIR.

Colors: NCR green, ED blue, ET red (consistent with the other report figures).
Output: docs/figures/per_class_panel.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

REGIONS = [("NCR", 1, (0.13, 0.70, 0.30), "necrotic core"),
           ("ED", 2, (0.15, 0.40, 0.85), "edema"),
           ("ET", 3, (0.86, 0.16, 0.16), "enhancing tumor")]

# candidate pool: a spread of cases (fast — load ~30 masks)
cands = [f"BraTS2021_{i:05d}" for i in range(0, 60, 2)]
cands = [c for c in cands if (DATA / c / "mask.npy").exists()][:30]

def norm_disp(img2d):
    fg = img2d[img2d != 0]
    if fg.size == 0:
        return np.zeros_like(img2d)
    lo, hi = np.percentile(fg, (1, 99))
    return np.clip((img2d - lo) / (hi - lo + 1e-6), 0, 1)

def brain_bbox(flair2d, pad=4):
    """Square bounding box around the non-zero brain region (removes dark border)."""
    ys, xs = np.where(flair2d != 0)
    if ys.size == 0:
        return slice(None), slice(None)
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = flair2d.shape
    side = max(r1 - r0, c1 - c0) // 2 + pad
    cr, cc = (r0 + r1) // 2, (c0 + c1) // 2
    r0 = max(cr - side, 0); r1 = min(cr + side, h)
    c0 = max(cc - side, 0); c1 = min(cc + side, w)
    return slice(r0, r1), slice(c0, c1)

# region voxel count + best slice per case
info = {}  # case -> (mask, {label: (count, best_slice)})
for c in cands:
    m = np.asarray(np.load(DATA / c / "mask.npy"))
    per = {}
    for _, lab, _, _ in REGIONS:
        cnt = int((m == lab).sum())
        bs = int((m == lab).sum(axis=(0, 1)).argmax()) if cnt else 0
        per[lab] = (cnt, bs)
    info[c] = per

fig, axes = plt.subplots(3, 3, figsize=(8.4, 8.6))
for row, (abbr, lab, color, full) in enumerate(REGIONS):
    top = sorted(cands, key=lambda c: info[c][lab][0], reverse=True)[:3]
    for col, case in enumerate(top):
        ax = axes[row, col]
        img = np.asarray(np.load(DATA / case / "image.npy")).astype(np.float32)
        m = np.asarray(np.load(DATA / case / "mask.npy"))
        sl = info[case][lab][1]
        flair = np.rot90(img[3][:, :, sl])
        mm = np.rot90(m[:, :, sl])
        rs, cs = brain_bbox(flair)              # crop away the dark border
        flair, mm = flair[rs, cs], mm[rs, cs]
        ax.imshow(norm_disp(flair), cmap="gray")
        ov = np.zeros((*mm.shape, 4))
        ov[mm == lab] = to_rgba(color, 0.78)
        ax.imshow(ov, interpolation="nearest")
        ax.text(0.5, 0.99, case.replace("BraTS2021_", "BraTS "), transform=ax.transAxes,
                fontsize=8, color="white", ha="center", va="top")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[row, 0].set_ylabel(f"{abbr} · {full}", fontsize=11, fontweight="bold", color=color)

fig.suptitle("Per-class examples (region overlaid on FLAIR): NCR green, ED blue, ET red",
             fontsize=11, y=0.995)
fig.subplots_adjust(left=0.045, right=0.997, top=0.965, bottom=0.003, wspace=0.015, hspace=0.015)
fig.savefig(OUT / "per_class_panel.png", dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.02)
print("wrote", OUT / "per_class_panel.png")
