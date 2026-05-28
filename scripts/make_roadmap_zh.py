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
        "title": "已完成",
        "subtitle": "Done",
        "items": ["● 模型訓練與驗證", "● Web 工作站部署", "● 不確定性量化"],
        "facecolor": "#cde9d3",
        "edgecolor": "#1f7a3a",
        "linestyle": "solid",
        "linewidth": 2.6,
    },
    {
        "title": "優化中",
        "subtitle": "Optimizing",
        "items": ["• 跨資料集泛化測試", "• 模型架構改進", "• UI / UX 使用者測試"],
        "facecolor": "#e3f1da",
        "edgecolor": "#4f9d4f",
        "linestyle": "solid",
        "linewidth": 2.2,
    },
    {
        "title": "探索中",
        "subtitle": "Exploring",
        "items": ["• 模型量化與剪枝", "• 推論進一步加速", "• 多疾病應用探索"],
        "facecolor": "#eef7e5",
        "edgecolor": "#8bb37a",
        "linestyle": "dashed",
        "linewidth": 2.0,
    },
]

fig, ax = plt.subplots(figsize=(15, 5.6))
n = len(STAGES)
bw, bh, gap = 4.4, 4.6, 0.9
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
    ax.text(cx0, 1.55, s["title"], ha="center", va="center",
            fontsize=26, fontweight="bold", color="#13315c", zorder=3)
    # English subtitle
    ax.text(cx0, 1.00, s["subtitle"], ha="center", va="center",
            fontsize=14, color="#5b6a82", zorder=3)

    # Items, left-aligned inside box
    item_x = x + 0.45
    for k, it in enumerate(s["items"]):
        y = 0.20 - k * 0.65
        ax.text(item_x, y, it, ha="left", va="center",
                fontsize=18, color="#1f2937", zorder=3)

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
