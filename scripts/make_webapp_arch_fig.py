"""Slide 25 — web workstation architecture callout.

Small horizontal diagram: browser UI -> FastAPI backend -> AURAS (the SAME
inference path as evaluation) -> 3D result + report. The point of the slide
is "what we evaluate is what we deploy", so the middle box is emphasised.

Output: docs/report_figures/webapp_arch.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "docs" / "report_figures"

FC, EC = "#eaf0f8", "#3b6fb5"

STAGES = [
    ("Browser UI", "Upload & explore in 3D", "Niivue viewer"),
    ("FastAPI backend", "Orchestrates each case", "preprocess → infer"),
    ("AURA model", "Segments + flags uncertainty", "4 regions + confidence"),
    ("Result", "Clinical-ready output", "volumes · report ZIP"),
]

fig, ax = plt.subplots(figsize=(14, 4.0))
n = len(STAGES)
bw, bh, gap = 3.0, 2.5, 0.7
x = 0.0
ax.set_xlim(-0.3, n * (bw + gap) - gap + 0.3)
ax.set_ylim(-1.6, 1.6)
ax.set_aspect("equal")
ax.axis("off")

for i, (title, idea, how) in enumerate(STAGES):
    ax.add_patch(FancyBboxPatch((x, -bh / 2), bw, bh,
                 boxstyle="round,pad=0.04,rounding_size=0.20",
                 linewidth=2.2, edgecolor=EC, facecolor=FC, zorder=2))
    cx0 = x + bw / 2
    ax.text(cx0, 0.78, title, ha="center", va="center", fontsize=17,
            fontweight="bold", color="#13315c", zorder=3)
    ax.text(cx0, 0.06, idea, ha="center", va="center", fontsize=13,
            color="#1f2937", zorder=3)
    ax.text(cx0, -0.66, how, ha="center", va="center", fontsize=11,
            color="#7b8aa0", zorder=3)
    if i < n - 1:
        ax.add_patch(FancyArrowPatch(
            (x + bw + 0.06, 0), (x + bw + gap - 0.06, 0),
            arrowstyle="-|>", mutation_scale=24, linewidth=2.4,
            color="#64748b", zorder=1))
    x += bw + gap

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT / "webapp_arch.png", dpi=130, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT / 'webapp_arch.png'}")
