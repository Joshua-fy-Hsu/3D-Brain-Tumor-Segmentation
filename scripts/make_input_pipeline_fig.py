"""Slide 8 — raw scans -> 5-channel network input -> 128^3 patch.

Built from the real case BraTS2021_00018 (same as Slides 4/5) so the
5-channel tensor and patch sampling are shown on actual data, not a cartoon.

Stage 1: the 5 input channels (T1, T1CE, T2, FLAIR, foreground mask)
Stage 2: 128^3 patch sampling — 50% tumor-centered / 50% random

Output: docs/report_figures/input_pipeline.png
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "report_figures"
CASE, SLICE, PATCH = "BraTS2021_00018", 91, 128

CH_NAMES = ["T1", "T1CE", "T2", "FLAIR", "Foreground\nmask"]


def norm_disp(a):
    fg = a[a != 0]
    if fg.size == 0:
        return a
    lo, hi = np.percentile(fg, (1, 99))
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


img = np.asarray(np.load(ROOT / "data" / CASE / "image.npy")).astype(np.float32)
mask = np.asarray(np.load(ROOT / "data" / CASE / "mask.npy"))
sl = [np.rot90(img[c][:, :, SLICE]) for c in range(5)]
mseg = np.rot90(mask[:, :, SLICE])
H, W = mseg.shape

# square brain bounding box (from foreground channel) so every thumbnail
# is the same size and the row aligns with the patch panel
fy, fx = np.where(sl[4] > 0)
p = 6
by0, by1 = max(fy.min() - p, 0), min(fy.max() + p, H)
bx0, bx1 = max(fx.min() - p, 0), min(fx.max() + p, W)
side = max(by1 - by0, bx1 - bx0)
cyc, cxc = (by0 + by1) // 2, (bx0 + bx1) // 2
by0 = max(cyc - side // 2, 0); by1 = min(by0 + side, H); by0 = by1 - side
bx0 = max(cxc - side // 2, 0); bx1 = min(bx0 + side, W); bx0 = bx1 - side

fig = plt.figure(figsize=(15, 5.0))
gs = GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 1, 2.4], wspace=0.10)

# ---- stage 1: the 5 channels (cropped to brain) ---------------------
for c in range(5):
    ax = fig.add_subplot(gs[0, c])
    crop = (norm_disp(sl[c]) if c < 4 else sl[c])[by0:by1, bx0:bx1]
    ax.imshow(crop, cmap="gray" if c < 4 else "cividis")
    ax.set_title(CH_NAMES[c], fontsize=13, fontweight="bold", pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#2666d9"); s.set_linewidth(1.8)

# ---- stage 2: 128^3 patch sampling ----------------------------------
axP = fig.add_subplot(gs[0, 5])
axP.imshow(norm_disp(sl[3]), cmap="gray")          # FLAIR backdrop
axP.imshow(np.where(mseg > 0, 1.0, np.nan), cmap="autumn",
           alpha=0.45, interpolation="nearest")
axP.set_xticks([]); axP.set_yticks([])
axP.set_title(f"{PATCH}³ patch sampling", fontsize=13,
              fontweight="bold", pad=5)
axP.set_xlim(-3, W + 3)
axP.set_ylim(H + 3, -3)                              # origin upper + margin

# tumor-centered patch (red), label inside top-left of the box
ys, xs = np.where(mseg > 0)
tcy, tcx = int(ys.mean()), int(xs.mean())
half = PATCH // 2
ty = np.clip(tcy - half, 0, H - PATCH)
tx = np.clip(tcx - half, 0, W - PATCH)
axP.add_patch(Rectangle((tx, ty), PATCH, PATCH, fill=False,
              edgecolor="#dc2929", linewidth=2.6))
axP.text(tx + 5, ty + 11, "tumor-centered  50%", ha="left", va="center",
         fontsize=11, fontweight="bold", color="#dc2929",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                   alpha=0.75))

# random patch (gray dashed), label inside bottom-left of the box
rng = np.random.default_rng(7)
ry = rng.integers(0, H - PATCH)
rx = rng.integers(0, W - PATCH)
axP.add_patch(Rectangle((rx, ry), PATCH, PATCH, fill=False,
              edgecolor="#f3f4f6", linewidth=2.2, linestyle=(0, (4, 3))))
axP.text(rx + 5, ry + PATCH - 11, "random  50%", ha="left", va="center",
         fontsize=11, fontweight="bold", color="#4b5563",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                   alpha=0.75))

fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.03)
fig.savefig(OUT / "input_pipeline.png", dpi=200, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT / 'input_pipeline.png'}  ({CASE}, slice {SLICE})")
