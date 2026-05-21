"""Slide 9 — end-to-end project pipeline (the orienting map for the talk).

Six stages: raw MRI -> preprocess -> AURAS model -> training -> evaluation
-> web workstation. The model box and the deploy box are highlighted because
the rest of the deck zooms into them.

Output: docs/report_figures/pipeline_overview.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "docs" / "report_figures"

# (title, core idea, concrete how) — one neutral colour for every stage
FC, EC = "#eaf0f8", "#3b6fb5"
STAGES = [
    ("Raw MRI", "Four MRI views", "T1 · T1CE · T2 · FLAIR"),
    ("Preprocess", "Standardize & focus", "5-channel · 128³ patches"),
    ("AURA model", "Tumor segmentation", "CNN + transformer"),
    ("Training", "Stable optimization", "EMA · mixed precision"),
    ("Evaluation", "Validated accuracy", "Dice / HD95 · paired stats"),
    ("Web App", "Deployed tool", "upload → segment → report"),
]

fig, ax = plt.subplots(figsize=(15, 4.0))
n = len(STAGES)
bw, bh, gap = 2.65, 2.5, 0.58
x = 0.0
ax.set_xlim(-0.3, n * (bw + gap) - gap + 0.3)
ax.set_ylim(-1.6, 1.6)
ax.set_aspect("equal")
ax.axis("off")

for i, (title, idea, how) in enumerate(STAGES):
    box = FancyBboxPatch((x, -bh / 2), bw, bh,
                         boxstyle="round,pad=0.04,rounding_size=0.20",
                         linewidth=2.2, edgecolor=EC, facecolor=FC, zorder=2)
    ax.add_patch(box)
    cx0 = x + bw / 2
    ax.text(cx0, 0.78, title, ha="center", va="center",
            fontsize=16, fontweight="bold", color="#13315c", zorder=3)
    ax.text(cx0, 0.06, idea, ha="center", va="center",
            fontsize=12.5, color="#1f2937", zorder=3)
    ax.text(cx0, -0.66, how, ha="center", va="center",
            fontsize=10, color="#7b8aa0", zorder=3)
    if i < n - 1:
        ax.add_patch(FancyArrowPatch(
            (x + bw + 0.05, 0), (x + bw + gap - 0.05, 0),
            arrowstyle="-|>", mutation_scale=24, linewidth=2.4,
            color="#64748b", zorder=1))
    x += bw + gap

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT / "pipeline_overview.png", dpi=130, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT / 'pipeline_overview.png'}")
