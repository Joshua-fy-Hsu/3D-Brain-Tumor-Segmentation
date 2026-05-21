"""Slide 19 — Trustworthiness: uncertainty panel + Dice@Coverage bar chart.

Output: docs/report_figures/trustworthiness.png
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import numpy as np

OUT  = Path(__file__).resolve().parent.parent / "docs" / "report_figures"
HERO = OUT / "conclusion_hero.png"

# ── Data ────────────────────────────────────────────────────────────────────
dice_vals = [0.8299, 0.8456, 0.8489, 0.8417]
labels    = ["100%\n(all)", "90%", "80%", "70%"]

# ── Load hero image and crop uncertainty panel (rightmost quarter) ───────────
hero = mpimg.imread(str(HERO))
H, W = hero.shape[:2]
# crop: rightmost quarter, skip top ~8% (suptitle row)
top_skip = int(H * 0.08)
unc_panel = hero[top_skip:, W*3//4:, :]

# ── Figure: 2 columns ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 4.5))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.12, width_ratios=[1, 1.4])

# Left — uncertainty image
ax_l = fig.add_subplot(gs[0])
ax_l.imshow(unc_panel)
ax_l.set_title("Uncertainty Map  (BraTS2021_01418)", fontsize=13,
               fontweight="bold", pad=8)
ax_l.axis("off")
ax_l.text(0.5, -0.04,
          "High uncertainty concentrates at region boundaries and error zones",
          ha="center", va="top", transform=ax_l.transAxes,
          fontsize=10, color="#555", style="italic")

# Right — selective prediction bar chart
ax_r = fig.add_subplot(gs[1])

colors = ["#94a3b8", "#60a5fa", "#3b82f6", "#1d4ed8"]
bars = ax_r.bar(labels, dice_vals, color=colors, edgecolor="none", width=0.55)
bars[0].set_color("#cbd5e1")

for bar, val in zip(bars, dice_vals):
    ax_r.text(bar.get_x() + bar.get_width()/2, val + 0.0008,
              f"{val:.3f}", ha="center", va="bottom", fontsize=11,
              fontweight="bold", color="#1e293b")

y_base = dice_vals[0]
y_best = max(dice_vals)
best_idx = dice_vals.index(y_best)
ax_r.set_ylim(0.815, 0.868)
ax_r.text(best_idx, y_best + 0.006,
          f"+{y_best - y_base:.3f} vs all",
          ha="center", va="bottom", fontsize=11,
          color="#dc2626", fontweight="bold")

ax_r.set_ylabel("Mean Dice", fontsize=11)
ax_r.set_xlabel("Coverage (fraction of cases predicted)", fontsize=10.5)
ax_r.set_title("Selective Prediction  (TTA)", fontsize=13, fontweight="bold", pad=10)
ax_r.yaxis.grid(True, linestyle="--", alpha=0.5)
ax_r.set_axisbelow(True)
ax_r.spines[["top", "right"]].set_visible(False)

fig.subplots_adjust(left=0.04, right=0.97, top=0.92, bottom=0.14)
out = OUT / "trustworthiness.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
