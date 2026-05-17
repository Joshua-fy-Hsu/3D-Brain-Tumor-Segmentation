"""Slide 12 — Cross-modal attention: one-concept schematic (no MRI).

Grounded in src/model/blocks/modality_stem.py + cross_modal_attn.py:
  4 independent per-modality stems (Conv-IN-LeakyReLU, separate weights)
  -> at each spatial position the 4 modalities are 4 tokens
  -> attention across modalities (each reads the other three)
  -> concat + 1x1 fuse (+ foreground) -> features feed the encoder.

No top label / bottom caption / in-box title (slide title already says it).

Output: docs/report_figures/crossmodal.png
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "docs" / "report_figures"

FC, EC = "#dbe7f6", "#2f6fb5"
INK, SUB = "#1f2937", "#6b7280"

fig, ax = plt.subplots(figsize=(13, 4.7))
ax.set_xlim(0, 13)
ax.set_ylim(0, 4.7)
ax.axis("off")


def box(x, y, w, h, title, sub=None, fs=15):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.03,rounding_size=0.10",
                 linewidth=2.2, edgecolor=EC, facecolor=FC, zorder=3))
    yt = y + h / 2 + (0.20 if sub else 0)
    ax.text(x + w / 2, yt, title, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=INK, zorder=4)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.26, sub, ha="center", va="center",
                fontsize=10.5, color=SUB, zorder=4)


# --- column 1: per-modality stems -----------------------------------------
mods = ["T1", "T1CE", "T2", "FLAIR"]
SX, sw, sh = 0.55, 2.0, 0.84
ys = [3.55, 2.62, 1.69, 0.76]
for m, y in zip(mods, ys):
    box(SX, y, sw, sh, m, "stem", fs=15)

# --- column 2: cross-modal attention (glyph only, no redundant title) -----
AX, aw, ah = 4.05, 3.7, 3.95
ay = 0.45
ax.add_patch(FancyBboxPatch((AX, ay), aw, ah,
             boxstyle="round,pad=0.03,rounding_size=0.10",
             linewidth=2.4, edgecolor=EC, facecolor=FC, zorder=3))
ax.text(AX + aw / 2, ay + ah - 0.42,
        "every scan attends to the others",
        ha="center", va="center", fontsize=12, color=INK,
        fontweight="bold", zorder=5)

# all-to-all glyph with double-headed links (information flows both ways)
cx, cy, R = AX + aw / 2, ay + 1.55, 1.05
angd = np.deg2rad([90, 0, -90, 180])
nodes = [(cx + R * np.cos(a), cy + R * np.sin(a)) for a in angd]
for i, (px, py) in enumerate(nodes):
    for j, (qx, qy) in enumerate(nodes):
        if i < j:
            ax.add_patch(FancyArrowPatch((px, py), (qx, qy),
                         arrowstyle="<|-|>", mutation_scale=11, lw=1.5,
                         color="#7f93ad", shrinkA=14, shrinkB=14, zorder=4))
for (px, py), lbl in zip(nodes, mods):
    ax.add_patch(plt.Circle((px, py), 0.36, facecolor="#ffffff",
                 edgecolor=EC, lw=2.0, zorder=6))
    ax.text(px, py, lbl, ha="center", va="center", fontsize=10,
            fontweight="bold", color=INK, zorder=7)

for y in ys:
    ax.add_patch(FancyArrowPatch((SX + sw, y + sh / 2),
                 (AX, y + sh / 2), arrowstyle="-|>", mutation_scale=15,
                 lw=1.8, color=EC, zorder=2))

# --- column 3: fuse -> encoder --------------------------------------------
FX, fw, fh = 7.95, 2.4, 1.7
fy = 1.50
box(FX, fy, fw, fh, "Learned fusion", "combines all four", fs=14)
ax.add_patch(FancyArrowPatch((AX + aw, fy + fh / 2),
             (FX, fy + fh / 2), arrowstyle="-|>", mutation_scale=16,
             lw=2.0, color=EC, zorder=2))

EXX = FX + fw + 0.6
box(EXX, fy, 1.7, fh, "Encoder", "backbone", fs=14)
ax.add_patch(FancyArrowPatch((FX + fw, fy + fh / 2),
             (EXX, fy + fh / 2), arrowstyle="-|>", mutation_scale=16,
             lw=2.0, color="#374151", zorder=2))

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT / "crossmodal.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT / 'crossmodal.png'}")
