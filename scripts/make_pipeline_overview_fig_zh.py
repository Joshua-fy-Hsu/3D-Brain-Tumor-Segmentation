"""TumorSeg 競賽簡報用 — 系統架構總覽 (5 階段，含時序)

Output: docs/report_figures/pipeline_overview_zh.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "Arial"]
rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent.parent / "docs" / "report_figures"

FC, EC = "#eaf0f8", "#3b6fb5"
TIME_FC, TIME_FG = "#dbeafe", "#1d4ed8"

STAGES = [
    ("原始 MRI",    "四種模態影像", "T1 · T1CE · T2 · FLAIR", "~2 秒"),
    ("資料前處理",  "標準化與聚焦", "5 通道 · 128³ patches",  "~1 秒"),
    ("TumorSeg 模型", "腫瘤分割",   "CNN + Transformer",      "0.24 秒"),
    ("推論加速",    "穩定預測",     "TTA · MC Dropout",       "~3 秒"),
    ("Web 工作站",  "臨床部署",     "上傳 → 分割 → 報告",      "即時呈現"),
]

fig, ax = plt.subplots(figsize=(15, 5.2))
n = len(STAGES)
bw, bh, gap = 2.85, 3.4, 0.62
x = 0.0
ax.set_xlim(-0.3, n * (bw + gap) - gap + 0.3)
ax.set_ylim(-2.6, 2.0)
ax.set_aspect("equal")
ax.axis("off")

for i, (title, idea, how, t) in enumerate(STAGES):
    box = FancyBboxPatch((x, -bh / 2), bw, bh,
                         boxstyle="round,pad=0.04,rounding_size=0.20",
                         linewidth=2.2, edgecolor=EC, facecolor=FC, zorder=2)
    ax.add_patch(box)
    cx0 = x + bw / 2
    ax.text(cx0, 1.20, title, ha="center", va="center",
            fontsize=24, fontweight="bold", color="#13315c", zorder=3)
    ax.text(cx0, 0.45, idea, ha="center", va="center",
            fontsize=19, color="#1f2937", zorder=3)
    ax.text(cx0, -0.28, how, ha="center", va="center",
            fontsize=14, color="#7b8aa0", zorder=3)

    # time badge near the bottom of each box
    badge_w, badge_h = 1.55, 0.65
    bx = cx0 - badge_w / 2
    by = -1.45
    badge = FancyBboxPatch((bx, by), badge_w, badge_h,
                           boxstyle="round,pad=0.02,rounding_size=0.18",
                           linewidth=0, facecolor=TIME_FC, zorder=3)
    ax.add_patch(badge)
    ax.text(cx0, by + badge_h / 2, t, ha="center", va="center",
            fontsize=14, fontweight="bold", color=TIME_FG, zorder=4)

    if i < n - 1:
        ax.add_patch(FancyArrowPatch(
            (x + bw + 0.05, 0.30), (x + bw + gap - 0.05, 0.30),
            arrowstyle="-|>", mutation_scale=24, linewidth=2.4,
            color="#64748b", zorder=1))
    x += bw + gap

# Total time banner under all boxes
total_y = -2.30
total_w = x - gap
total_x = 0.0
banner = FancyBboxPatch((total_x, total_y - 0.30), total_w, 0.62,
                        boxstyle="round,pad=0.02,rounding_size=0.20",
                        linewidth=0, facecolor="#fef3c7", zorder=2)
ax.add_patch(banner)
ax.text(total_x + total_w / 2, total_y, "全程約 10 秒",
        ha="center", va="center",
        fontsize=20, fontweight="bold", color="#b45309", zorder=3)

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT / "pipeline_overview_zh.png", dpi=130, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT / 'pipeline_overview_zh.png'}")
