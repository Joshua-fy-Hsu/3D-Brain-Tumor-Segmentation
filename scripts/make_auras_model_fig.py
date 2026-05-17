"""Slide 10 (replaces the build-up strip) — the full AURAS model.

Minimal backbone U (enc / bottleneck / dec) in neutral blue + the five AURAS
components in one accent colour, each attached right where it plugs in
(grounded in src/model/trans_resunet.py). Skip-connection / deep-sup detail
is deliberately omitted — that is Slide 11's job; this slide answers only
"what does AURAS add and where".

Output: docs/report_figures/auras_model.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "docs" / "report_figures"

ENC_FC, ENC_EC = "#dbe7f6", "#2f6fb5"
DEC_FC, DEC_EC = "#e3f1e6", "#2f9e4f"
BOT_FC, BOT_EC = "#ede9fe", "#6d28d9"
ADD_FC, ADD_EC = "#fff1da", "#e08a00"
INK, SUB, AINK = "#1f2937", "#6b7280", "#8a5200"

fig, ax = plt.subplots(figsize=(13, 7.0))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7.0)
ax.axis("off")

bw, bh = 1.7, 0.74
EX, DX = 4.35, 6.95
ROWS = [("32", "full", 5.55), ("64", "1/2", 4.55),
        ("128", "1/4", 3.55), ("256", "1/8", 2.55)]
BOT_Y = 1.35


def box(x, y, txt, sub, fc, ec, w=bw, h=bh):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.03,rounding_size=0.10",
                 linewidth=2.0, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(x + w / 2, y + h / 2 + 0.10, txt, ha="center", va="center",
            fontsize=12, fontweight="bold", color=INK, zorder=4)
    ax.text(x + w / 2, y + h / 2 - 0.18, sub, ha="center", va="center",
            fontsize=7.5, color=SUB, zorder=4)


def callout(x, y, w, h, txt, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.04,rounding_size=0.10",
                 linewidth=2.0, edgecolor=ADD_EC, facecolor=ADD_FC, zorder=5))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=AINK, zorder=6)


def aarrow(p, q, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12,
                 lw=1.7, color=ADD_EC, zorder=5,
                 connectionstyle=f"arc3,rad={rad}"))


# --- backbone U ------------------------------------------------------------
for c, r, y in ROWS:
    box(EX, y, f"{c} ch", f"enc · {r}", ENC_FC, ENC_EC)
    box(DX, y, f"{c} ch", f"dec · {r}", DEC_FC, DEC_EC)
box((EX + DX) / 2 + (bw - bw) / 2, BOT_Y, "512", "bottleneck",
    BOT_FC, BOT_EC)
BOTX = (EX + DX) / 2

exc, dxc = EX + bw / 2, DX + bw / 2
for i in range(3):
    ax.add_patch(FancyArrowPatch((exc, ROWS[i][2]), (exc, ROWS[i+1][2] + bh),
                 arrowstyle="-|>", mutation_scale=13, lw=1.6,
                 color=ENC_EC, zorder=2))
    ax.add_patch(FancyArrowPatch((dxc, ROWS[i+1][2] + bh), (dxc, ROWS[i][2]),
                 arrowstyle="-|>", mutation_scale=13, lw=1.6,
                 color=DEC_EC, zorder=2))
ax.add_patch(FancyArrowPatch((exc, ROWS[3][2]), (BOTX + 0.4, BOT_Y + bh),
             arrowstyle="-|>", mutation_scale=13, lw=1.6, color=ENC_EC,
             connectionstyle="arc3,rad=-0.25", zorder=2))
ax.add_patch(FancyArrowPatch((BOTX + bw - 0.4, BOT_Y + bh),
             (dxc, ROWS[3][2]), arrowstyle="-|>", mutation_scale=13,
             lw=1.6, color=DEC_EC, connectionstyle="arc3,rad=-0.25", zorder=2))

ax.add_patch(FancyArrowPatch((EX - 0.55, ROWS[0][2] + bh / 2),
             (EX, ROWS[0][2] + bh / 2), arrowstyle="-|>",
             mutation_scale=13, lw=1.7, color="#374151", zorder=2))
ax.add_patch(FancyArrowPatch((DX + bw, ROWS[0][2] + bh / 2),
             (DX + bw + 0.55, ROWS[0][2] + bh / 2), arrowstyle="-|>",
             mutation_scale=13, lw=1.7, color="#374151", zorder=2))
ax.text(DX + bw + 0.62, ROWS[0][2] + bh / 2, "segmentation", ha="left",
        va="center", fontsize=9, color="#374151")

# --- AURAS components, each attached locally (no long crossings) -----------
# 1. modality stems + cross-modal — left of enc top
callout(0.30, ROWS[0][2] - 0.10, 3.35, 0.94,
        "Modality stems\n+ Cross-modal attn")
aarrow((3.65, ROWS[0][2] + 0.37), (EX, ROWS[0][2] + bh / 2))
# 2. frequency block — left of enc, mid
callout(0.55, ROWS[2][2] + 0.02, 2.65, 0.74, "Frequency block")
aarrow((3.20, ROWS[2][2] + 0.37), (EX, ROWS[2][2] + bh / 2))
# 3. spectral-swin — under enc4 -> bottleneck
callout(EX - 0.55, 0.10, 3.0, 0.74, "Spectral-Swin stage")
aarrow((EX + 0.9, 0.84), (exc, ROWS[3][2]), rad=0.15)
aarrow((EX + 2.0, 0.84), (BOTX + 0.5, BOT_Y), rad=-0.15)
# 4. uncertainty head — right of bottleneck
callout(DX + 0.75, BOT_Y - 0.05, 2.6, 0.74, "Uncertainty head")
aarrow((DX + 0.75, BOT_Y + 0.32), (BOTX + bw, BOT_Y + bh / 2), rad=0.0)
ax.text(DX + 2.05, BOT_Y - 0.42, "→ variance", ha="center", va="center",
        fontsize=8.5, color=AINK, style="italic")
# 5. boundary head — right of decoder, mid
callout(DX + bw + 0.45, ROWS[2][2] + 0.02, 2.7, 0.74, "Boundary head")
aarrow((DX + bw + 0.45, ROWS[2][2] + 0.37), (DX + bw, ROWS[2][2] + bh / 2))
ax.text(DX + bw + 1.8, ROWS[2][2] - 0.40, "→ boundary", ha="center",
        va="center", fontsize=8.5, color=AINK, style="italic")
# 6. multiscale fusion + deep supervision — right of decoder top
callout(DX + bw + 0.45, ROWS[1][2] + 0.02, 2.7, 0.74, "Fusion + deep sup.")
aarrow((DX + bw + 0.45, ROWS[1][2] + 0.37), (DX + bw, ROWS[1][2] + bh / 2))

ax.text(6.5, 6.75,
        "Blue = backbone   ·   Amber = what AURAS adds on top",
        ha="center", va="center", fontsize=11.5, color="#475569",
        style="italic")

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT / "auras_model.png", dpi=140, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT / 'auras_model.png'}")
