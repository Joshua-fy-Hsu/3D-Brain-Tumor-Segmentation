"""Slide 11 — AURAS backbone: 3D Residual U-Net schematic.

Symmetric encoder/decoder U: 4 stages 32->64->128->256, x16=512 bottleneck,
skip connections at matching resolution, deep-supervision heads
(final / ds1 1/2 / ds2 1/4). Grounded in src/model/model.py (ResUnet3D):
Conv3D-InstanceNorm-LeakyReLU(0.01) residual blocks, 4-class softmax head.

Output: docs/report_figures/backbone.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "docs" / "report_figures"

ENC_FC, ENC_EC = "#dbe7f6", "#2f6fb5"
DEC_FC, DEC_EC = "#e3f1e6", "#2f9e4f"
BOT_FC, BOT_EC = "#ede9fe", "#6d28d9"
HEAD_FC, HEAD_EC = "#fde7e7", "#dc2929"

fig, ax = plt.subplots(figsize=(13, 6.8))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7.4)
ax.axis("off")

bw, bh = 1.95, 0.78
EX, DX = 1.4, 9.3                       # encoder / decoder column x
# rows: (channels, resolution, y)
ROWS = [("32", "full", 5.3), ("64", "1/2", 4.25),
        ("128", "1/4", 3.2), ("256", "1/8", 2.15)]
BOT_Y = 0.85


def box(x, y, txt, sub, fc, ec, w=bw):
    ax.add_patch(FancyBboxPatch((x, y), w, bh,
                 boxstyle="round,pad=0.03,rounding_size=0.10",
                 linewidth=2.0, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(x + w / 2, y + bh / 2 + 0.10, txt, ha="center", va="center",
            fontsize=12.5, fontweight="bold", color="#1f2937", zorder=4)
    ax.text(x + w / 2, y + bh / 2 - 0.19, sub, ha="center", va="center",
            fontsize=8.5, color="#6b7280", zorder=4)


# encoder + decoder boxes
for c, r, y in ROWS:
    box(EX, y, f"{c} ch", f"enc · {r}", ENC_FC, ENC_EC)
    box(DX, y, f"{c} ch", f"dec · {r}", DEC_FC, DEC_EC)
box(5.55, BOT_Y, "512 ch", "bottleneck · 1/16", BOT_FC, BOT_EC)

# encoder downsample (centered under each box)
exc = EX + bw / 2
for i in range(3):
    ax.add_patch(FancyArrowPatch((exc, ROWS[i][2]),
                 (exc, ROWS[i+1][2] + bh), arrowstyle="-|>",
                 mutation_scale=15, lw=1.8, color=ENC_EC, zorder=2))
ax.add_patch(FancyArrowPatch((exc, ROWS[3][2]), (5.7, BOT_Y + bh),
             arrowstyle="-|>", mutation_scale=15, lw=1.8, color=ENC_EC,
             connectionstyle="arc3,rad=-0.2", zorder=2))
# decoder upsample
dxc = DX + bw / 2
ax.add_patch(FancyArrowPatch((5.55 + bw, BOT_Y + bh), (dxc, ROWS[3][2]),
             arrowstyle="-|>", mutation_scale=15, lw=1.8, color=DEC_EC,
             connectionstyle="arc3,rad=-0.2", zorder=2))
for i in range(3, 0, -1):
    ax.add_patch(FancyArrowPatch((dxc, ROWS[i][2] + bh),
                 (dxc, ROWS[i-1][2]), arrowstyle="-|>",
                 mutation_scale=15, lw=1.8, color=DEC_EC, zorder=2))

# skip connections (same resolution, horizontal)
for _, _, y in ROWS:
    ax.add_patch(FancyArrowPatch((EX + bw, y + bh / 2),
                 (DX, y + bh / 2), arrowstyle="-|>", mutation_scale=13,
                 lw=1.5, color="#9ca3af", linestyle=(0, (5, 3)), zorder=1))
ax.text((EX + bw + DX) / 2, ROWS[0][2] + bh + 0.28, "skip connections",
        ha="center", fontsize=10.5, color="#6b7280", style="italic")

# input
ax.add_patch(FancyArrowPatch((EX - 0.75, ROWS[0][2] + bh / 2),
             (EX, ROWS[0][2] + bh / 2), arrowstyle="-|>",
             mutation_scale=14, lw=1.8, color="#374151", zorder=2))
ax.text(EX - 0.78, ROWS[0][2] + bh / 2 + 0.95,
        "5-ch input\nT1·T1CE·T2·FLAIR·fg", ha="center", va="center",
        fontsize=9.5, color="#374151")

# output head + deep-supervision heads (final / ds1 / ds2)
HX = DX + bw + 0.95
ds = [("final", ROWS[0][2], "#dc2929"),
      ("ds1 · ½", ROWS[1][2], "#dc2929"),
      ("ds2 · ¼", ROWS[2][2], "#dc2929")]
for lbl, y, col in ds:
    ax.add_patch(FancyArrowPatch((DX + bw, y + bh / 2),
                 (HX, y + bh / 2), arrowstyle="-|>", mutation_scale=13,
                 lw=1.6, color=col, zorder=2))
    ax.text(HX + 0.08, y + bh / 2, lbl, ha="left", va="center",
            fontsize=10, fontweight="bold", color="#b91c1c", zorder=4)
box(HX + 1.2, ROWS[0][2] - 0.05, "4-class", "softmax head",
    HEAD_FC, HEAD_EC, w=1.7)
ax.text(HX + 0.55, ROWS[2][2] - 0.7,
        "Deep supervision loss:\nfinal + ½·ds1 + ¼·ds2", ha="center",
        va="center", fontsize=9.5, color="#b91c1c")

# residual-block callout — top centre, clear of all skip/flow lines
ax.add_patch(FancyBboxPatch((4.55, 6.2), 4.0, 1.05,
             boxstyle="round,pad=0.05,rounding_size=0.12",
             linewidth=1.6, edgecolor="#94a3b8", facecolor="#f8fafc",
             zorder=3))
ax.text(6.55, 6.95, "Residual block — reused by every variant",
        ha="center", fontsize=10.5, fontweight="bold", color="#1f2937",
        zorder=4)
ax.text(6.55, 6.5, "[ Conv3D → InstanceNorm → LeakyReLU(0.01) ] ×2  "
        "+ identity skip", ha="center", va="center", fontsize=9.5,
        color="#475569", zorder=4)

ax.text(6.5, 0.32, "All component flags OFF  →  this is exactly a plain "
        "3D Residual U-Net", ha="center", va="center", fontsize=11,
        color="#475569", style="italic")

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT / "backbone.png", dpi=115, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT / 'backbone.png'}")
