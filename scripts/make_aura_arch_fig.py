"""Slide 10 — AURA architecture diagram (vertical U-Net style).

Layout mirrors the hand-drawn version:
  - Encoder column (left): Input → Enc1..4
  - Transformer Bottleneck (bottom centre)
  - Decoder column (right): Dec1..4 → Final / DS heads → Output
  - Dashed skip connections (horizontal)
  - Dropout markers on Dec3 + Dec4 (last two decoder stages)
  - Legend row at the bottom

Output: docs/report_figures/aura_arch.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "docs" / "report_figures"

# ── Colour palette ────────────────────────────────────────────────────────────
C_IN   = ("#e8edf2", "#5b6a7d")   # input / output  (grey-blue)
C_ENC  = ("#cfe0f5", "#2f6fb5")   # encoder         (blue)
C_BOT  = ("#fef3c7", "#b45309")   # transformer      (amber)
C_DEC  = ("#d1f0d9", "#2f9e4f")   # decoder          (green)
C_DROP = ("#fde68a", "#92400e")   # dropout label    (dark amber)
C_HEAD = ("#fde7e7", "#dc2929")   # output heads     (red)
C_OUT  = ("#ede9fe", "#6d28d9")   # output box       (purple)

fig, ax = plt.subplots(figsize=(11, 10))
ax.set_xlim(0, 11)
ax.set_ylim(0, 10.4)
ax.axis("off")

BW, BH = 2.2, 0.72      # standard box width / height
EX = 0.55               # encoder column left-x
DX = 6.30               # decoder column left-x
HX = DX + BW + 0.25     # head column left-x
HW = 1.55               # head box width

# vertical centres for each resolution level (top → bottom)
# index 0 = full-res (Enc1 / Dec4), index 3 = lowest (Enc4 / Dec1)
YS = [8.9, 7.6, 6.3, 5.0]   # y for each encoder/decoder stage
BOT_Y = 3.1                   # bottleneck y
INP_Y = YS[0] + 1.05         # input box y


def box(x, y, line1, line2, fc, ec, w=BW, h=BH, bold1=True):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.10",
        linewidth=2.0, edgecolor=ec, facecolor=fc, zorder=3))
    fs1 = 11.5 if bold1 else 10.5
    ax.text(x + w / 2, y + h * 0.64, line1,
            ha="center", va="center",
            fontsize=fs1, fontweight="bold" if bold1 else "normal",
            color="#1f2937", zorder=4)
    ax.text(x + w / 2, y + h * 0.26, line2,
            ha="center", va="center",
            fontsize=9, color="#4b5563", zorder=4)


def arrow(x0, y0, x1, y1, col, dashed=False, rad=0.0):
    ls = (0, (5, 3)) if dashed else "-"
    style = f"arc3,rad={rad}" if rad else "arc3,rad=0"
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=14, lw=1.8,
        color=col, linestyle=ls,
        connectionstyle=style, zorder=2))


# ── Input box ─────────────────────────────────────────────────────────────────
box(EX, INP_Y, "Input", "128³ × 5 ch", *C_IN)
arrow(EX + BW / 2, INP_Y, EX + BW / 2, YS[0] + BH, C_IN[1])

# ── Encoder stages ────────────────────────────────────────────────────────────
ENC = [
    ("Enc1", "128³ × 32"),
    ("Enc2",  "64³ × 64"),
    ("Enc3", "32³ × 128"),
    ("Enc4", "16³ × 256"),
]
for i, (lbl, dims) in enumerate(ENC):
    box(EX, YS[i], lbl, dims, *C_ENC)
    if i < 3:
        arrow(EX + BW / 2, YS[i], EX + BW / 2, YS[i + 1] + BH, C_ENC[1])

# Enc4 → Bottleneck
bot_cx = 4.3 + 1.4       # centre-x of bottleneck
arrow(EX + BW / 2, YS[3], bot_cx - 0.8, BOT_Y + BH,
      C_ENC[1], rad=-0.25)

# ── Transformer Bottleneck ────────────────────────────────────────────────────
BOT_X, BOT_W = 3.55, 2.9
box(BOT_X, BOT_Y, "Transformer Bottleneck",
    "4 layers · 8³ · 512 tokens · 8 heads",
    *C_BOT, w=BOT_W)

# Bottleneck → Dec1
arrow(BOT_X + BOT_W, BOT_Y + BH, DX + BW / 2, YS[3],
      C_DEC[1], rad=-0.25)

# ── Decoder stages ────────────────────────────────────────────────────────────
DEC = [
    ("Dec1", "16³ × 256", False),
    ("Dec2", "32³ × 128", False),
    ("Dec3",  "64³ × 64", True),    # Dropout
    ("Dec4", "128³ × 32", True),    # Dropout
]
for i, (lbl, dims, has_drop) in enumerate(DEC):
    ri = 3 - i   # Dec1 → YS[3], Dec4 → YS[0]
    fc = C_DROP[0] if has_drop else C_DEC[0]
    ec = C_DEC[1]
    box(DX, YS[ri], lbl, dims, fc, ec)
    if i < 3:
        arrow(DX + BW / 2, YS[ri] + BH, DX + BW / 2, YS[ri - 1], C_DEC[1])

# Dropout badge on Dec3 + Dec4
for ri in [0, 1]:   # YS[0] and YS[1]
    ax.text(DX + BW - 0.08, YS[ri] + BH - 0.13,
            "Dropout", ha="right", va="top",
            fontsize=7.5, color="#92400e",
            fontstyle="italic", zorder=5)

# ── Skip connections ──────────────────────────────────────────────────────────
for i in range(4):
    y_mid = YS[i] + BH / 2
    ri = 3 - i
    arrow(EX + BW, y_mid, DX, y_mid, "#9ca3af", dashed=True)
    ax.text((EX + BW + DX) / 2, y_mid + 0.14, "Skip",
            ha="center", fontsize=8, color="#9ca3af", style="italic", zorder=5)

# ── Output heads ──────────────────────────────────────────────────────────────
HEADS = [
    (YS[0], "Final",   "1×1×1 Conv"),
    (YS[1], "DS1",     "1/2 res"),
    (YS[2], "DS2",     "1/4 res"),
]
for y, lbl, sub in HEADS:
    arrow(DX + BW, y + BH / 2, HX, y + BH / 2, C_HEAD[1])
    box(HX, y + (BH - 0.60) / 2, lbl, sub, *C_HEAD, w=HW, h=0.60)

# Output box (top right)
OUT_Y = INP_Y
box(HX, OUT_Y, "Output", "128³ × 4 classes", *C_OUT, w=HW)
arrow(HX + HW / 2, HEADS[0][0] + BH, HX + HW / 2, OUT_Y, C_OUT[1])

# ── Legend ────────────────────────────────────────────────────────────────────
LEG_Y = 1.75
ax.add_patch(FancyBboxPatch((0.3, 0.25), 10.4, 1.55,
             boxstyle="round,pad=0.05,rounding_size=0.10",
             linewidth=1.4, edgecolor="#d1d5db", facecolor="#f9fafb", zorder=2))

legend_items = [
    (C_ENC,  "CNN  —  Residual → IN → LeakyReLU(0.01)"),
    (C_BOT,  "Transformer  —  4 layers · 512 tokens · 3D pos. embed"),
    (C_DEC,  "Decoder  —  ConvTranspose3d + Concat (skip) + Res Block"),
    (C_DROP, "Dropout  —  Dropout3d(0.10), last 2 decoder stages"),
]
for j, ((fc, ec), txt) in enumerate(legend_items):
    lx = 0.6 + (j % 2) * 5.2
    ly = LEG_Y - (j // 2) * 0.72
    ax.add_patch(FancyBboxPatch((lx, ly - 0.18), 0.36, 0.32,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=4))
    ax.text(lx + 0.50, ly - 0.02, txt,
            ha="left", va="center", fontsize=8.8, color="#374151", zorder=4)

fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
fig.savefig(OUT / "aura_arch.png", dpi=130, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT / 'aura_arch.png'}")
