"""Slide 5 — clinical nested-region figure on a REAL brain MRI.

Left: a real T1CE slice (case BraTS2021_00018, same as Slides 4/5) cropped
around the tumor, with the ground-truth segmentation overlaid so the nested
biology is visible in situ (necrotic center → enhancing rim → edema).
Right: the clinical action each nested aggregate drives.

Colors match the other Section-B figures: NCR green, ET red, ED blue.
Output: docs/report_figures/clinical_regions.png
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgba

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "report_figures"

CASE, SLICE = "BraTS2021_00018", 91
NCR, ET, ED = "#22b34d", "#dc2929", "#2666d9"

SEG_CMAP = ListedColormap([(0, 0, 0, 0), to_rgba(NCR), to_rgba(ED),
                           to_rgba(ET)])
SEG_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], SEG_CMAP.N)


def norm_disp(img2d):
    fg = img2d[img2d != 0]
    lo, hi = np.percentile(fg, (1, 99))
    return np.clip((img2d - lo) / (hi - lo + 1e-6), 0, 1)


folder = ROOT / "data" / CASE
img = np.asarray(np.load(folder / "image.npy")).astype(np.float32)
mask = np.asarray(np.load(folder / "mask.npy"))

t1ce = np.rot90(img[1][:, :, SLICE])      # channel 1 = T1CE
m = np.rot90(mask[:, :, SLICE])

# crop around the tumor with padding so the nested structure fills the panel
ys, xs = np.where(m > 0)
pad = 28
y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, m.shape[0])
x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, m.shape[1])
base = norm_disp(t1ce)[y0:y1, x0:x1]
mc = m[y0:y1, x0:x1]

fig = plt.figure(figsize=(14, 5.4))
gs = GridSpec(1, 2, width_ratios=[1.0, 1.15], wspace=0.04)

# ---- left: real MRI + segmentation overlay ----------------------------
axL = fig.add_subplot(gs[0])
axL.imshow(base, cmap="gray")
axL.imshow(mc, cmap=SEG_CMAP, norm=SEG_NORM, interpolation="nearest",
           alpha=0.50)
axL.set_title("Real MRI scan — the three regions inside one tumor",
              fontsize=15, fontweight="bold", pad=10)
axL.axis("off")

# horizontal legend strip under the image (centroid labels would collide)
handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=14,
                       markerfacecolor=c, markeredgecolor="none", label=l)
           for c, l in [(NCR, "NCR — necrotic core"),
                        (ET, "ET — enhancing rim"),
                        (ED, "ED — infiltrative edema")]]
axL.legend(handles=handles, loc="upper center",
           bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False,
           fontsize=12.5, handletextpad=0.4, columnspacing=1.4)

# ---- right: nested aggregate -> clinical action -----------------------
axR = fig.add_subplot(gs[1])
axR.set_xlim(0, 10)
axR.set_ylim(0, 10)
axR.axis("off")


_renderer = fig.canvas.get_renderer()


def colored_formula(x0, y, tokens, fontsize):
    """Draw a sequence of (text, color) tokens left-to-right, color-coded."""
    x = x0
    inv = axR.transData.inverted()
    for txt, color in tokens:
        t = axR.text(x, y, txt, ha="left", va="center", fontsize=fontsize,
                     fontweight="bold", color=color)
        bb = t.get_window_extent(renderer=_renderer)
        w = inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0]
        x += w


def action_row(yc, accent, agg, formula_tokens, paren, action):
    axR.add_patch(FancyBboxPatch((0.3, yc - 1.75), 9.5, 3.5,
                  boxstyle="round,pad=0.05,rounding_size=0.25",
                  linewidth=2.4, edgecolor=accent,
                  facecolor=accent + "1a"))
    axR.text(0.9, yc + 1.0, agg, ha="left", va="center",
             fontsize=18, fontweight="bold", color=accent)
    colored_formula(0.9, yc + 0.05, formula_tokens, fontsize=19)
    axR.text(9.4, yc + 0.05, f"({paren})", ha="right", va="center",
             fontsize=12.5, color="#9ca3af", style="italic")
    axR.text(0.9, yc - 0.95, action, ha="left", va="center",
             fontsize=15, color="#1f2937")


DARK = "#1f2937"
action_row(6.5, "#b8860b", "Tumor Core (TC)  →  Surgery",
           [("TC = ", DARK), ("NCR", NCR), (" + ", DARK), ("ET", ET)],
           "solid tumor mass",
           "Resect to debulk the tumor and relieve pressure")
action_row(2.5, "#6d28d9", "Whole Tumor (WT)  →  Radiotherapy",
           [("WT = ", DARK), ("NCR", NCR), (" + ", DARK), ("ET", ET),
            (" + ", DARK), ("ED", ED)],
           "mass + infiltration",
           "Irradiate the field to kill infiltrating cells")

fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.04)
fig.savefig(OUT / "clinical_regions.png", dpi=200, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT / 'clinical_regions.png'}  ({CASE}, slice {SLICE})")
