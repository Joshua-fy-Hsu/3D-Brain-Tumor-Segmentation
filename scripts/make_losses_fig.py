"""Slide 17 — AURAS training objective: loss composition.

Three additive terms, grounded in src/training/losses.py:
  - RegionWiseDiceFocalLoss   : Dice + Focal(gamma=2) over WT/TC/ET region
                                 prob maps, deep-supervised (1 / 0.5 / 0.25).
  - UncertaintyAwareLoss      : + lambda_unc * |variance.mean() - target|
                                 (lambda_unc = 0.05, stabiliser).
  - BoundaryAwareLoss         : + lambda_b(t) * (BCE + edge-Dice), with a
                                 per-epoch ramp on lambda_b.
Visual = composition only (the "what plugs into what"); the slide bullets
carry the "why" so the two layers don't repeat.

Output: docs/report_figures/losses.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "docs" / "report_figures"

TERM_FC, TERM_EC = "#dbe7f6", "#2f6fb5"          # neutral blue, uniform
TOT_FC, TOT_EC = "#ede9fe", "#6d28d9"            # distinct result box

fig, ax = plt.subplots(figsize=(13, 3.7))
ax.set_xlim(0, 13)
ax.set_ylim(1.05, 4.55)
ax.axis("off")

bw, bh = 3.05, 1.55
Y = 2.55
# (x, title, sub line 1, sub line 2)
TERMS = [
    (0.55, "Region Dice + Focal",
     "WT / TC / ET prob maps", "deep-supervised  1 · ½ · ¼"),
    (4.55, "Uncertainty term",
     "λ_unc · |var − 0|", "λ_unc = 0.05  (constant)"),
    (8.55, "Boundary term",
     "λ_b(t) · (BCE + edge-Dice)", "λ_b ramps up per epoch"),
]


def box(x, y, title, s1, s2, fc, ec, w=bw, h=bh):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.04,rounding_size=0.12",
                 linewidth=2.0, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(x + w / 2, y + h - 0.42, title, ha="center", va="center",
            fontsize=13.5, fontweight="bold", color="#1f2937", zorder=4)
    ax.text(x + w / 2, y + h / 2 - 0.18, s1, ha="center", va="center",
            fontsize=10.5, color="#374151", zorder=4)
    ax.text(x + w / 2, y + 0.32, s2, ha="center", va="center",
            fontsize=9.0, color="#6b7280", style="italic", zorder=4)


for x, t, s1, s2 in TERMS:
    box(x, Y, t, s1, s2, TERM_FC, TERM_EC)

# "+" between the three term boxes
for cx in (4.05, 8.05):
    ax.text(cx, Y + bh / 2, "+", ha="center", va="center",
            fontsize=26, fontweight="bold", color="#6b7280", zorder=4)

# arrow from the three terms into the total
ax.add_patch(FancyArrowPatch((11.6, Y + bh / 2), (12.05, Y + bh / 2),
             arrowstyle="-|>", mutation_scale=18, lw=2.0,
             color="#374151", zorder=2))

# total loss box (distinct, like the backbone head)
ax.add_patch(FancyBboxPatch((12.15, Y), 0.78, bh,
             boxstyle="round,pad=0.04,rounding_size=0.12",
             linewidth=2.0, edgecolor=TOT_EC, facecolor=TOT_FC, zorder=3))
ax.text(12.54, Y + bh / 2 + 0.16, "L", ha="center", va="center",
        fontsize=15, fontweight="bold", color="#4c1d95", zorder=5)
ax.text(12.54, Y + bh / 2 - 0.22, "total", ha="center", va="center",
        fontsize=9.5, color="#6d28d9", zorder=5)

# role strip under each term (figure layer = what each term acts on)
roles = ["accuracy under imbalance", "per-voxel trust signal",
         "tumor-margin sharpness"]
for (x, *_), role in zip(TERMS, roles):
    ax.text(x + bw / 2, Y - 0.45, role, ha="center", va="center",
            fontsize=9.5, color="#475569")

ax.text(6.5, 1.35,
        "One objective, composed. Each term sits on top of the one "
        "before it, monotonic with the ablation.",
        ha="center", va="center", fontsize=11, color="#475569",
        style="italic")

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT / "losses.png", dpi=115, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT / 'losses.png'}")
