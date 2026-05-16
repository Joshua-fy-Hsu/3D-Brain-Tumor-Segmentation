"""Slide 5 — the three sub-regions shown in isolation, same case as Slide 4.

NCR | ED | ET each overlaid alone on FLAIR, so the audience sees that ET is
the smallest / most fragmented region (sets up Slides 20, 23, 24).

Colors match nested_regions.png / modality_panel.png:
  NCR green · ED blue · ET red.

Output: docs/report_figures/region_panel.png  (+ .pdf)
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "report_figures"

CASE = "BraTS2021_00018"   # same case as the Slide 4 modality panel
SLICE = 91

REGIONS = [
    ("NCR", 1, (0.13, 0.70, 0.30), "necrotic core"),
    ("ED",  2, (0.15, 0.40, 0.85), "edema"),
    ("ET",  3, (0.86, 0.16, 0.16), "enhancing tumor"),
]


def norm_disp(img2d):
    fg = img2d[img2d != 0]
    if fg.size == 0:
        return np.zeros_like(img2d)
    lo, hi = np.percentile(fg, (1, 99))
    return np.clip((img2d - lo) / (hi - lo + 1e-6), 0, 1)


def main():
    folder = ROOT / "data" / CASE
    img = np.asarray(np.load(folder / "image.npy")).astype(np.float32)
    mask = np.asarray(np.load(folder / "mask.npy"))

    flair = np.rot90(img[3][:, :, SLICE])      # channel 3 = FLAIR
    m = np.rot90(mask[:, :, SLICE])
    base = norm_disp(flair)
    total = max(int((mask[:, :, SLICE] > 0).sum()), 1)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0))
    for ax, (abbr, lab, color, full) in zip(axes, REGIONS):
        ax.imshow(base, cmap="gray")
        ov = np.zeros((*m.shape, 4))
        ov[m == lab] = to_rgba(color, 0.75)
        ax.imshow(ov, interpolation="nearest")
        pct = 100.0 * (m == lab).sum() / total
        ax.set_title(f"{abbr} — {full}\n{pct:.0f}% of tumor on this slice",
                     fontsize=14, fontweight="bold", color="#222", pad=8)
        ax.axis("off")

    fig.suptitle(f"The three sub-regions shown separately  ·  "
                 f"case {CASE}, one axial slice",
                 fontsize=13, color="#555", y=1.02)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.02,
                        wspace=0.06)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"region_panel.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT / 'region_panel.png'}  ({CASE}, slice {SLICE})")


if __name__ == "__main__":
    main()
