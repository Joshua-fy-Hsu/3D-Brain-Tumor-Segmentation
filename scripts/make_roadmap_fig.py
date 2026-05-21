"""Slide 29 — limitations & future-work roadmap.

A horizontal timeline: what is done (left, solid) -> what is next
(right, outlined). Honest scoping — single-cohort today, external
validation + an efficient distilled variant next.

Output: docs/report_figures/roadmap.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "docs" / "report_figures"

DONE_FC, DONE_EC = "#e3f1e6", "#2f9e4f"
NEXT_FC, NEXT_EC = "#eef2f7", "#94a3b8"

DONE = [
    ("Integrated", "AURA built up\ncomponent-by-component"),
    ("Validated", "paired tests,\none BraTS cohort"),
    ("Deployed", "live browser\nworkstation"),
]
NEXT = [
    ("External validation", "multi-dataset,\ncross-institution"),
    ("Compression", "quantization &\npruning"),
]

fig, ax = plt.subplots(figsize=(14, 4.2))
bw, bh = 2.75, 2.0
gap = 0.55
xs = []
x = 0.0
for _ in range(len(DONE) + len(NEXT)):
    xs.append(x)
    x += bw + gap
total_r = xs[-1] + bw

ax.set_xlim(-0.4, total_r + 0.4)
ax.set_ylim(-2.0, 1.9)
ax.set_aspect("equal")
ax.axis("off")

# spine arrow under the boxes
ax.add_patch(FancyArrowPatch((-0.2, -1.45), (total_r + 0.2, -1.45),
             arrowstyle="-|>", mutation_scale=26, linewidth=2.6,
             color="#64748b"))
ax.text(xs[1] + bw / 2, -1.85, "DONE", ha="center", va="center",
        fontsize=13, fontweight="bold", color=DONE_EC)
ax.text(xs[3] + bw / 2 + gap, -1.85, "NEXT", ha="center", va="center",
        fontsize=13, fontweight="bold", color="#64748b")

allboxes = [(t, s, True) for t, s in DONE] + [(t, s, False) for t, s in NEXT]
for (title, sub, done), x0 in zip(allboxes, xs):
    fc, ec = (DONE_FC, DONE_EC) if done else (NEXT_FC, NEXT_EC)
    style = "round,pad=0.04,rounding_size=0.20"
    ax.add_patch(FancyBboxPatch((x0, -bh / 2 + 0.25), bw, bh,
                 boxstyle=style, linewidth=2.4, edgecolor=ec,
                 facecolor=fc, linestyle="-" if done else (0, (4, 3)),
                 zorder=2))
    ax.text(x0 + bw / 2, 0.75, title, ha="center", va="center",
            fontsize=15, fontweight="bold",
            color=ec if done else "#475569", zorder=3)
    ax.text(x0 + bw / 2, -0.15, sub, ha="center", va="center",
            fontsize=11, color="#475569", zorder=3)
    # tick connecting box to the spine
    ax.plot([x0 + bw / 2, x0 + bw / 2], [-bh / 2 + 0.25, -1.45],
            color="#cbd5e1", linewidth=1.4, zorder=1)

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT / "roadmap.png", dpi=130, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT / 'roadmap.png'}")
