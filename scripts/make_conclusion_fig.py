"""Slide 30 — conclusion: three-pillar takeaway card.

Accurate · Trustworthy · Deployed — one line of evidence each, then a
"Questions?" footer. Wording stays defensible (no specific p-values, since
the cross-variant stats may still be finishing).

Output: docs/report_figures/conclusion.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "docs" / "report_figures"

PILLARS = [
    ("Accurate", "#2f9e4f", "#e3f1e6",
     "Strong region-wise\nsegmentation (ET · TC · WT)"),
    ("Trustworthy", "#2666d9", "#e7eefb",
     "Per-voxel uncertainty that\ntracks error · calibrated"),
    ("Deployed", "#6d28d9", "#ede9fe",
     "Live browser workstation\nrunning the validated model"),
]

fig, ax = plt.subplots(figsize=(13, 4.6))
bw, bh, gap = 3.6, 3.2, 0.7
x = 0.0
ax.set_xlim(-0.4, 3 * bw + 2 * gap + 0.4)
ax.set_ylim(-2.6, 2.2)
ax.set_aspect("equal")
ax.axis("off")

for (title, ec, fc, sub) in PILLARS:
    ax.add_patch(FancyBboxPatch((x, -bh / 2), bw, bh,
                 boxstyle="round,pad=0.05,rounding_size=0.25",
                 linewidth=2.6, edgecolor=ec, facecolor=fc, zorder=2))
    cx0 = x + bw / 2
    ax.text(cx0, 0.78, title, ha="center", va="center", fontsize=22,
            fontweight="bold", color=ec, zorder=3)
    ax.text(cx0, -0.55, sub, ha="center", va="center", fontsize=13,
            color="#1f2937", zorder=3)
    x += bw + gap

ax.text((3 * bw + 2 * gap) / 2, -2.15, "Thank you — Questions?",
        ha="center", va="center", fontsize=20, fontweight="bold",
        color="#13315c")

fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01)
fig.savefig(OUT / "conclusion.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT / 'conclusion.png'}")
