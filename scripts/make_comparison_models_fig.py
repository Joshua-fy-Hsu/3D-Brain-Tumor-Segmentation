"""Slide 14 — Comparison models: Baseline / Complex / AURA (three-column card layout).

Output: docs/report_figures/comparison_models.png
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "docs" / "report_figures"

# ── Colour palette ─────────────────────────────────────────────────────────────
C_BASE  = ("#eaf0f8", "#3b6fb5")   # Baseline  — neutral blue
C_COMP  = ("#fef3c7", "#b45309")   # Complex   — amber
C_AURA  = ("#d1f0d9", "#2f9e4f")   # AURA      — green (hero)

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")

CW, CH = 3.8, 5.6    # card width / height
GAP    = 0.7
X0     = (14 - 3 * CW - 2 * GAP) / 2   # left edge of first card

CARDS = [
    ("Baseline",  C_BASE,
     [("Modules", "Residual blocks"),
      ("",        "Instance Norm"),
      ("",        "LeakyReLU"),
      ("",        "Skip connections"),
      ("Role",    "Lower bound")]),
    ("Complex",   C_COMP,
     [("Modules", "Modality stems"),
      ("",        "Cross-modal attn"),
      ("",        "Spectral-Swin"),
      ("",        "Frequency filter"),
      ("",        "Uncertainty head"),
      ("",        "Boundary head"),
      ("",        "Multi-scale fusion"),
      ("Role",    "Upper bound")]),
    ("AURA",      C_AURA,
     [("Modules", "CNN encoder"),
      ("",        "Transformer bottleneck"),
      ("",        "CNN decoder"),
      ("",        "Skip connections"),
      ("",        "Deep supervision"),
      ("Role",    "Our model")]),
]

for i, (title, (fc, ec), rows) in enumerate(CARDS):
    x = X0 + i * (CW + GAP)
    lw = 3.0 if title == "AURA" else 1.8

    # Card background
    ax.add_patch(FancyBboxPatch(
        (x, 0.5), CW, CH,
        boxstyle="round,pad=0.04,rounding_size=0.15",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))

    # Title bar
    ax.add_patch(FancyBboxPatch(
        (x, 0.5 + CH - 0.78), CW, 0.78,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        linewidth=0, edgecolor=ec, facecolor=ec, zorder=3,
        clip_on=True))
    ax.text(x + CW / 2, 0.5 + CH - 0.39, title,
            ha="center", va="center",
            fontsize=17, fontweight="bold", color="white", zorder=4)

    # Content rows
    y_start = 0.5 + CH - 1.05
    dy = 0.52
    prev_label = None
    for label, value in rows:
        display_label = label if label != prev_label else ""
        if display_label and display_label != "":
            ax.text(x + 0.22, y_start, display_label + ":",
                    ha="left", va="top",
                    fontsize=9.5, color="#6b7280",
                    fontstyle="italic", zorder=4)
        ax.text(x + 0.22, y_start - 0.22, value,
                ha="left", va="top",
                fontsize=10.5, color="#1f2937",
                fontweight="bold" if label == "Role" else "normal",
                zorder=4)
        y_start -= dy
        prev_label = label

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT / "comparison_models.png", dpi=130, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT / 'comparison_models.png'}")
