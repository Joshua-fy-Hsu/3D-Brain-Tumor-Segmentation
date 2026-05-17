"""Slide 17 — AURAS training objective rendered as equations (Google-Slides
ready PNG; insert as an image).

Grounded in src/training/losses.py:
  L_seg : RegionWiseDiceFocalLoss  (Dice + Focal gamma=2 over WT/TC/ET,
          deep-supervised with weights 1 / 0.5 / 0.25)
  + lambda_unc * |v_bar - t|       (UncertaintyAwareLoss, lambda_unc=0.05, t=0)
  + lambda_b(t) * (w_bce*BCE + w_ed*EdgeDice)
          (BoundaryAwareLoss, lambda_b ramp 0.05 -> 0.25, w_bce=0.3, w_ed=0.2)

Output: docs/report_figures/loss_equation.png
"""
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "docs" / "report_figures"

INK = "#1f2937"
SUB = "#475569"
ACC = "#2666d9"

fig, ax = plt.subplots(figsize=(13, 5.4))
ax.set_xlim(0, 13)
ax.set_ylim(0, 5.4)
ax.axis("off")


def label(y, s):
    ax.text(0.55, y, s, ha="left", va="center", fontsize=13,
            fontweight="bold", color=ACC)


def eq(y, s, size=20, color=INK):
    ax.text(6.5, y, s, ha="center", va="center", fontsize=size, color=color)


# --- master equation -------------------------------------------------------
label(4.95, "Total objective")
eq(4.15,
   r"$\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{seg}}"
   r"+\lambda_{\mathrm{unc}}\,|\bar{v}-t|"
   r"+\lambda_b(t)\,(\,w_{\mathrm{bce}}\,\mathrm{BCE}"
   r"+w_{\mathrm{ed}}\,\mathrm{EdgeDice}\,)$",
   size=23)

# --- segmentation term -----------------------------------------------------
label(3.20, "Region term  (handles imbalance)")
eq(2.45,
   r"$\mathcal{L}_{\mathrm{seg}}="
   r"\sum_{s\,\in\,\{1,\,\frac{1}{2},\,\frac{1}{4}\}} w_s"
   r"\sum_{r\,\in\,\{\mathrm{WT,TC,ET}\}}"
   r"\left(\mathrm{Dice}_r+\mathrm{Focal}_r\right)$",
   size=21)

eq(1.55,
   r"$\mathrm{Dice}_r=1-\frac{2\sum p_r g_r}{\sum p_r+\sum g_r}"
   r"\qquad\qquad"
   r"\mathrm{Focal}_r=-\sum (1-p_t)^{\gamma}\log p_t,\ \ \gamma=2$",
   size=18)

# --- constants footnote ----------------------------------------------------
eq(0.60,
   r"$w=(1,\,0.5,\,0.25)\qquad"
   r"\lambda_{\mathrm{unc}}=0.05,\ \ t=0\qquad"
   r"\lambda_b:\ 0.05\!\rightarrow\!0.25\ \mathrm{ramp},\ \ "
   r"w_{\mathrm{bce}}=0.3,\ \ w_{\mathrm{ed}}=0.2$",
   size=14, color=SUB)

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT / "loss_equation.png", dpi=160, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT / 'loss_equation.png'}")
