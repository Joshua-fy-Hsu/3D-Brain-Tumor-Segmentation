"""TumorSeg 競賽簡報用 — 不確定性量化中文版本

Output: docs/report_figures/trustworthiness_zh.png
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "Arial"]
rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "docs" / "report_figures"
HERO = ROOT / "docs" / "figures" / "conclusion_hero.png"

# ── Data ────────────────────────────────────────────────────────────────────
dice_vals = [0.8299, 0.8456, 0.8489, 0.8417]
labels    = ["100%\n(全部)", "90%", "80%", "70%"]

# ── Load hero image and crop the four panels ─────────────────────────────────
hero = mpimg.imread(str(HERO))
H, W = hero.shape[:2]
top_skip = int(H * 0.22)      # skip suptitle AND panel header text
bottom_skip = int(H * 0.12)   # skip the NCR/ED/ET legend row
# Four panels in the source image, each occupying a horizontal quarter.
t1ce_panel = hero[top_skip:H - bottom_skip, :W//4, :]
unc_panel  = hero[top_skip:H - bottom_skip, W*3//4:, :]

# ── Figure: 3 columns ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 4.8))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.12, width_ratios=[1.0, 1.0, 2.0])

# Left — T1CE input image
ax_l = fig.add_subplot(gs[0])
ax_l.imshow(t1ce_panel)
ax_l.set_title("T1CE (輸入影像)", fontsize=14,
               fontweight="bold", pad=8)
ax_l.axis("off")
ax_l.text(0.5, -0.05,
          "案例 BraTS2021_01418",
          ha="center", va="top", transform=ax_l.transAxes,
          fontsize=12, color="#1e293b", fontweight="bold")

# Middle — uncertainty heatmap
ax_m = fig.add_subplot(gs[1])
ax_m.imshow(unc_panel)
ax_m.set_title("不確定性熱圖", fontsize=14,
               fontweight="bold", pad=8)
ax_m.axis("off")
ax_m.text(0.5, -0.05,
          "高不確定性集中於腫瘤邊界與易錯區域",
          ha="center", va="top", transform=ax_m.transAxes,
          fontsize=12, color="#1e293b", fontweight="bold")

# Right — selective prediction bar chart
ax_r = fig.add_subplot(gs[2])

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
          f"提升 +{y_best - y_base:.3f}",
          ha="center", va="bottom", fontsize=11,
          color="#dc2626", fontweight="bold")

ax_r.set_ylabel("平均 Dice", fontsize=12)
ax_r.set_xlabel("覆蓋率 (預測案例比例)", fontsize=11)
ax_r.set_title("選擇性預測 (Selective Prediction)", fontsize=14, fontweight="bold", pad=10)
ax_r.yaxis.grid(True, linestyle="--", alpha=0.5)
ax_r.set_axisbelow(True)
ax_r.spines[["top", "right"]].set_visible(False)

fig.subplots_adjust(left=0.04, right=0.97, top=0.92, bottom=0.14)
out = OUT / "trustworthiness_zh.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
