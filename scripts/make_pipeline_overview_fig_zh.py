"""TumorSeg 競賽簡報用 — 系統架構總覽 (5 階段)

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
STAGES = [
    ("原始 MRI",   "四種模態影像",    "T1 · T1CE · T2 · FLAIR"),
    ("資料前處理", "標準化與聚焦",    "5 通道 · 128³ patches"),
    ("TumorSeg 模型", "腫瘤分割",     "CNN + Transformer"),
    ("推論加速",   "穩定預測",        "TTA · MC Dropout"),
    ("Web 工作站", "臨床部署",        "上傳 → 分割 → 報告"),
]

fig, ax = plt.subplots(figsize=(15, 4.0))
n = len(STAGES)
bw, bh, gap = 2.85, 2.5, 0.62
x = 0.0
ax.set_xlim(-0.3, n * (bw + gap) - gap + 0.3)
ax.set_ylim(-1.6, 1.6)
ax.set_aspect("equal")
ax.axis("off")

for i, (title, idea, how) in enumerate(STAGES):
    box = FancyBboxPatch((x, -bh / 2), bw, bh,
                         boxstyle="round,pad=0.04,rounding_size=0.20",
                         linewidth=2.2, edgecolor=EC, facecolor=FC, zorder=2)
    ax.add_patch(box)
    cx0 = x + bw / 2
    ax.text(cx0, 0.78, title, ha="center", va="center",
            fontsize=24, fontweight="bold", color="#13315c", zorder=3)
    ax.text(cx0, 0.06, idea, ha="center", va="center",
            fontsize=19, color="#1f2937", zorder=3)
    ax.text(cx0, -0.66, how, ha="center", va="center",
            fontsize=14, color="#7b8aa0", zorder=3)
    if i < n - 1:
        ax.add_patch(FancyArrowPatch(
            (x + bw + 0.05, 0), (x + bw + gap - 0.05, 0),
            arrowstyle="-|>", mutation_scale=24, linewidth=2.4,
            color="#64748b", zorder=1))
    x += bw + gap

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT / "pipeline_overview_zh.png", dpi=130, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT / 'pipeline_overview_zh.png'}")
