"""Dataset-overview figure for Slide 3 — grounded in PROJECT REPORT Section I.

Numbers are taken verbatim from the report so the slide and the written
report agree:
  - BraTS 2021 training set, N=1251 (1000 train / 251 val), 240x240x155, 1mm
  - Classes 0 bg / 1 NCR / 2 ED / 3 ET ; WT=1,2,3  TC=1,3  ET=3
  - Voxel %: bg 98.93, NCR 0.16, ED 0.67, ET 0.24  (tumor ~1.07%)
  - Imbalance: bg:NCR~617:1  bg:ED~147:1  bg:ET~412:1
Colors match make_sample_figures.py and the deck accent.

Output: docs/report_figures/dataset_overview.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

OUT = Path(__file__).resolve().parents[1] / "docs" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#2E6FE0"
NCR = "#E51A1A"
ED = "#1ACC33"
ET = "#1A4DF2"
BG = "#9AA0A6"
GRAY = "#5A5A5A"

fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5))
for ax in axes:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

# ---- Panel A: cohort + input ----
a = axes[0]
a.set_title("BraTS 2021 cohort", fontsize=13, fontweight="bold", pad=10)
a.add_patch(FancyBboxPatch((0.5, 7.2), 9.0, 1.9,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            linewidth=1.4, edgecolor=BLUE, facecolor="#EAF1FC"))
a.text(5.0, 8.55, "N = 1,251 patients", ha="center", fontsize=13,
       fontweight="bold", color="#111")
a.text(5.0, 7.75, "1,000 train   ·   251 validation", ha="center",
       fontsize=10.5, color=GRAY)
rows = [
    "4 MRI modalities: T1 · T1ce · T2 · FLAIR",
    "Skull-stripped, co-registered, 1 mm iso",
    "Volume 240 x 240 x 155  -->  128³ patches",
    "+ foreground mask  =  5-channel input",
]
for i, t in enumerate(rows):
    y = 6.0 - i * 1.25
    a.text(0.7, y, "•", fontsize=14, color=BLUE, va="center")
    a.text(1.2, y, t, fontsize=10.6, color="#111", va="center")
a.text(5.0, 0.35, "RSNA-ASNR-MICCAI 2021 challenge set", ha="center",
       fontsize=9, color=GRAY, style="italic")

# ---- Panel B: nested regions ----
b = axes[1]
b.set_title("Labels → three nested scoring regions", fontsize=13,
            fontweight="bold", pad=10)
b.add_patch(Circle((5, 4.7), 4.0, facecolor=ED, edgecolor="white",
                    linewidth=2, alpha=0.85))
b.add_patch(Circle((5, 4.0), 2.7, facecolor=NCR, edgecolor="white",
                    linewidth=2, alpha=0.92))
b.add_patch(Circle((5, 3.6), 1.5, facecolor=ET, edgecolor="white",
                    linewidth=2))
b.text(5, 8.55, "WT  =  NCR + ED + ET", ha="center", fontsize=11.5,
       fontweight="bold", color="#111")
b.text(5, 5.95, "TC = NCR + ET", ha="center", fontsize=10.5,
       fontweight="bold", color="white")
b.text(5, 3.55, "ET", ha="center", va="center", fontsize=11,
       fontweight="bold", color="white")
b.text(5, 0.35, "ET ⊂ TC ⊂ WT  —  each scored separately",
       ha="center", fontsize=9.2, color=GRAY, style="italic")

# ---- Panel C: class imbalance (real voxel %) ----
c = axes[2]
c.set_title("Heavily imbalanced  (voxel %, log scale)", fontsize=13,
            fontweight="bold", pad=10)
c.axis("on")
c.set_xlim(0.1, 200)
c.set_ylim(-0.6, 3.6)
c.set_xscale("log")
c.set_yticks([])
for s in ("top", "right", "left"):
    c.spines[s].set_visible(False)
labels = ["Background", "ED", "ET", "NCR"]
vals = [98.93, 0.67, 0.24, 0.16]
cols = [BG, ED, ET, NCR]
notes = ["98.93 %", "0.67 %  (bg:ED ≈ 147:1)",
         "0.24 %  (bg:ET ≈ 412:1)", "0.16 %  (bg:NCR ≈ 617:1)"]
for i, (v, cc, nt) in enumerate(zip(vals, cols, notes)):
    y = 3 - i
    c.barh(y, v, height=0.55, color=cc, edgecolor="none")
    c.text(v * 1.25, y, nt, va="center", fontsize=9.3, color="#111")
    c.text(0.092, y, labels[i], va="center", ha="right", fontsize=10,
           fontweight="bold", color="#111")
c.set_xlabel("share of all voxels (%)", fontsize=9.3, color=GRAY)
c.text(0.5, -0.45,
       "Only ~1.07 % of voxels are tumor → plain accuracy / CE useless",
       fontsize=8.8, color=GRAY, style="italic", transform=c.transData)

fig.tight_layout(rect=[0, 0, 1, 0.97])
p = OUT / "dataset_overview.png"
fig.savefig(p, dpi=150)
print(f"Wrote {p}")
