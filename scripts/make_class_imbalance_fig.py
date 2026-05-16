"""Slide 6 — class imbalance. Uses the official full-cohort voxel counts
(1251 volumes) from the written report so the deck and report agree exactly.

Output: docs/report_figures/class_imbalance.png  (+ .pdf)
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = Path(__file__).resolve().parent.parent / "docs" / "report_figures"

# official counts (report §4, 1251 volumes) — ordered high -> low
CLASSES = ["Background", "ED", "ET", "NCR"]
COUNTS = [11_048_872_504, 75_328_509, 26_830_591, 17_896_396]
PCT = [98.93, 0.67, 0.24, 0.16]
COLORS = ["#9ca3af", "#2666d9", "#dc2929", "#22b34d"]  # match Slides 3-5

fig, ax = plt.subplots(figsize=(9.5, 4.8))
bars = ax.bar(CLASSES, COUNTS, color=COLORS, width=0.62,
              edgecolor="white", linewidth=1.2)
ax.set_yscale("log")
ax.set_ylabel("Voxel count  (log scale)", fontsize=12)
ax.set_ylim(1e6, 3e10)

ax.yaxis.set_major_formatter(FuncFormatter(
    lambda v, _: {1e6: "1M", 1e7: "10M", 1e8: "100M",
                   1e9: "1B", 1e10: "10B"}.get(v, "")))

for b, pct in zip(bars, PCT):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.35,
            f"{pct:.2f}%", ha="center", va="bottom",
            fontsize=13, fontweight="bold", color="#1f2937")

ax.set_title("Only ~1.07% of voxels are tumor — background dominates "
             "(note the log axis)", fontsize=13.5, fontweight="bold", pad=14)
ax.text(0.5, -0.20,
        "Background : ED  ≈ 147 : 1      "
        "Background : ET  ≈ 412 : 1      "
        "Background : NCR  ≈ 617 : 1",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=12, color="#374151")

ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", labelsize=12)
fig.subplots_adjust(bottom=0.20, top=0.86)
fig.savefig(OUT / "class_imbalance.png", dpi=200,
            bbox_inches="tight", facecolor="white")
print(f"wrote {OUT / 'class_imbalance.png'}")
