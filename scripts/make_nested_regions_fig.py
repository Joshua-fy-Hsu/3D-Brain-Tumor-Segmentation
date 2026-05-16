"""Slide 3 — nested BraTS region schematic: WT ⊃ TC ⊃ ET.

Renders three concentric rounded rectangles with the class→region mapping.
Output: docs/report_figures/nested_regions.png  (+ .pdf for vector use).
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "docs" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)

# region: (width, height, face, edge, set-label, class-label, label-y-offset)
REGIONS = [
    ("WT = {1, 2, 3}", 10.0, 6.6, "#dbeafe", "#2563eb"),
    ("TC = {1, 3}",     7.0, 4.2, "#fde68a", "#d97706"),
    ("ET = {3}",        4.8, 1.9, "#fecaca", "#dc2626"),
]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.set_xlim(-5.6, 5.6)
ax.set_ylim(-4.2, 3.9)
ax.set_aspect("equal")
ax.axis("off")

for set_lbl, w, h, face, edge in REGIONS:
    box = FancyBboxPatch(
        (-w / 2, -h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.35",
        linewidth=2.4, edgecolor=edge, facecolor=face, zorder=1,
    )
    ax.add_patch(box)
    # set-notation label sits just inside the top edge of each box
    ax.text(0, h / 2 - 0.38, set_lbl, ha="center", va="center",
            fontsize=15, fontweight="bold", color=edge, zorder=3)

# class-tissue labels, each in the visible band of its own region
ax.text(0, 2.55, "edema (ED)", ha="center", va="center",
        fontsize=13, color="#1e3a5f", zorder=3)
ax.text(0, 1.22, "necrotic core (NCR)", ha="center", va="center",
        fontsize=13, color="#7c4a03", zorder=3)
ax.text(0, -0.28, "enhancing tumor (ET)", ha="center", va="center",
        fontsize=13, fontweight="bold", color="#7f1d1d", zorder=3)

ax.text(0, -3.80, "Regions are nested:  ET  ⊂  TC  ⊂  WT",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color="#334155")

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"nested_regions.{ext}", dpi=200, bbox_inches="tight",
                facecolor="white")
print(f"wrote {OUT / 'nested_regions.png'}")
