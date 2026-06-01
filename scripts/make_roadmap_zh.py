"""TumorSeg 競賽簡報用 — 未來展望三階段 Roadmap

Output: docs/report_figures/roadmap_zh.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "Arial"]
rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent.parent / "docs" / "report_figures"

# Green gradient: done -> optimizing -> exploring
STAGES = [
    {
        "title": "初賽",
        "subtitle": "4 月企劃書",
        "items": [
            "● 基礎分割模型",
            "● 簡易展示介面",
            "● 完成模型訓練",
        ],
        "facecolor": "#cde9d3",
        "edgecolor": "#1f7a3a",
        "linestyle": "solid",
        "linewidth": 2.6,
    },
    {
        "title": "複賽",
        "subtitle": "6 月現在",
        "items": [
            "● 升級混合架構模型",
            "● 完整 Web 工作站",
            "● 模型自評信心度",
        ],
        "facecolor": "#e3f1da",
        "edgecolor": "#4f9d4f",
        "linestyle": "solid",
        "linewidth": 2.2,
    },
    {
        "title": "探索中",
        "subtitle": "比賽後",
        "items": [
            "• 推論速度優化",
            "• 模型體積壓縮",
            "• 多資料集驗證",
        ],
        "facecolor": "#eef7e5",
        "edgecolor": "#8bb37a",
        "linestyle": "dashed",
        "linewidth": 2.0,
    },
]

fig, ax = plt.subplots(figsize=(15, 6.0))
n = len(STAGES)
bw, bh, gap = 4.6, 5.0, 0.9
x = 0.0
ax.set_xlim(-0.3, n * (bw + gap) - gap + 0.3)
ax.set_ylim(-2.6, 2.6)
ax.set_aspect("equal")
ax.axis("off")

for i, s in enumerate(STAGES):
    box = FancyBboxPatch(
        (x, -bh / 2), bw, bh,
        boxstyle="round,pad=0.04,rounding_size=0.22",
        linewidth=s["linewidth"], edgecolor=s["edgecolor"],
        facecolor=s["facecolor"], linestyle=s["linestyle"], zorder=2,
    )
    ax.add_patch(box)
    cx0 = x + bw / 2

    # Stage title (Chinese)
    ax.text(cx0, 1.85, s["title"], ha="center", va="center",
            fontsize=26, fontweight="bold", color="#13315c", zorder=3)
    # Subtitle
    ax.text(cx0, 1.30, s["subtitle"], ha="center", va="center",
            fontsize=14, color="#5b6a82", zorder=3)

    # Items, left-aligned. Centered vertically around y = -0.55.
    item_x = x + 0.40
    n_items = len(s["items"])
    spacing = 0.60 if n_items >= 4 else 0.70
    center_y = -0.55
    y_top = center_y + (n_items - 1) * spacing / 2
    for k, it in enumerate(s["items"]):
        y = y_top - k * spacing
        ax.text(item_x, y, it, ha="left", va="center",
                fontsize=16 if n_items >= 4 else 18,
                color="#1f2937", zorder=3)

    # Arrow to next
    if i < n - 1:
        ax.add_patch(FancyArrowPatch(
            (x + bw + 0.05, 0), (x + bw + gap - 0.05, 0),
            arrowstyle="-|>", mutation_scale=28, linewidth=2.6,
            color="#64748b", zorder=1))
    x += bw + gap

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT / "roadmap_zh.png", dpi=130, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT / 'roadmap_zh.png'}")
