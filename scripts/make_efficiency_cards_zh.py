"""TumorSeg 競賽簡報用 — 四個效率指標卡片橫排

Output: docs/report_figures/efficiency_cards_zh.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch

rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "Arial"]
rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent.parent / "docs" / "report_figures"

FC, EC = "#eaf0f8", "#3b6fb5"
ACCENT = "#13315c"

CARDS = [
    ("45 M",   "參數量",     "Parameters"),
    ("0.24 秒", "推論時間",   "Inference time"),
    ("1.9 GB", "GPU 記憶體", "GPU memory"),
    ("172 MB", "模型大小",   "Model size"),
]

fig, ax = plt.subplots(figsize=(15, 4.2))
n = len(CARDS)
bw, bh, gap = 3.0, 3.0, 0.55
x = 0.0
ax.set_xlim(-0.3, n * (bw + gap) - gap + 0.3)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect("equal")
ax.axis("off")

for i, (value, label_zh, label_en) in enumerate(CARDS):
    box = FancyBboxPatch((x, -bh / 2), bw, bh,
                         boxstyle="round,pad=0.04,rounding_size=0.20",
                         linewidth=2.2, edgecolor=EC, facecolor=FC, zorder=2)
    ax.add_patch(box)
    cx0 = x + bw / 2
    # Big number
    ax.text(cx0, 0.45, value, ha="center", va="center",
            fontsize=46, fontweight="bold", color=ACCENT, zorder=3)
    # Chinese label
    ax.text(cx0, -0.45, label_zh, ha="center", va="center",
            fontsize=26, color="#1f2937", zorder=3)
    # English subtitle
    ax.text(cx0, -0.98, label_en, ha="center", va="center",
            fontsize=16, color="#7b8aa0", zorder=3)
    x += bw + gap

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT / "efficiency_cards_zh.png", dpi=130, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT / 'efficiency_cards_zh.png'}")
