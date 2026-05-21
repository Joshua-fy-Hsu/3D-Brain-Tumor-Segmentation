"""Slide 16 — Why not Complex? Per-region Dice bar chart: AURA vs Complex.

Output: docs/report_figures/why_not_complex.png
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "docs" / "report_figures"

# TTA + Post-process numbers
regions = ["Dice ET", "Dice TC", "Dice WT", "Mean Dice"]
complex_vals = [0.7888, 0.7808, 0.9291, round((0.7888+0.7808+0.9291)/3, 4)]
aura_vals    = [0.7855, 0.8038, 0.9284, round((0.7855+0.8038+0.9284)/3, 4)]

x = np.arange(len(regions))
w = 0.32

fig, ax = plt.subplots(figsize=(10, 5))

bars_c = ax.bar(x - w/2, complex_vals, w, label="Complex",
                color="#93c5fd", edgecolor="none")
bars_a = ax.bar(x + w/2, aura_vals,    w, label="AURA",
                color="#b91c1c", edgecolor="none")

# Value labels
for bar in bars_c:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{bar.get_height():.3f}", ha="center", va="bottom",
            fontsize=10, color="#1d4ed8", fontweight="bold")
for bar in bars_a:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{bar.get_height():.3f}", ha="center", va="bottom",
            fontsize=10, color="#b91c1c", fontweight="bold")

# Annotate TC gap
tc_x = x[1]
y_text = max(complex_vals[1], aura_vals[1]) + 0.014
ax.text(tc_x, y_text, "AURA +0.023", ha="center", fontsize=11,
        color="#111827", fontweight="bold")

# Annotate Mean Dice gap
mean_gap = round(aura_vals[3] - complex_vals[3], 3)
y_text2 = max(complex_vals[3], aura_vals[3]) + 0.014
ax.text(x[3], y_text2, f"AURA +{mean_gap:.3f}", ha="center", fontsize=11,
        color="#111827", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(regions, fontsize=13)
ax.set_ylim(0.72, 0.97)
ax.set_ylabel("Dice Score", fontsize=12)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=12, frameon=False)

fig.tight_layout()
fig.savefig(OUT / "why_not_complex.png", dpi=130, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT / 'why_not_complex.png'}")
