"""Slide 7 — tumor size & location heterogeneity across the cohort.

Left:  histogram of whole-tumor volume (cm^3), 1 mm iso voxels.
Right: axial centroid scatter — where tumors sit in the brain.

Computed over the full cohort (all preprocessed cases).
Output: docs/report_figures/tumor_distribution.png
"""
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "report_figures"

dirs = sorted(glob.glob(str(ROOT / "data" / "BraTS2021_*")))

vols, cx, cy = [], [], []
for i, d in enumerate(dirs):
    if i % 200 == 0:
        print(f"  {i}/{len(dirs)}")
    m = np.asarray(np.load(Path(d) / "mask.npy"))
    if not (m > 0).any():
        continue
    vols.append((m > 0).sum() / 1000.0)          # cm^3
    _, ys, xs = np.where(m > 0)
    cx.append(xs.mean())
    cy.append(ys.mean())

vols = np.array(vols)
med = np.median(vols)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# ---- left: volume histogram ------------------------------------------
ax1.hist(vols, bins=30, color="#2666d9", edgecolor="white", linewidth=0.6)
ax1.axvline(med, color="#dc2929", linewidth=2.2, linestyle="--")
ax1.text(med + 6, ax1.get_ylim()[1] * 0.92, f"median ≈ {med:.0f} cm³",
         color="#dc2929", fontsize=12, fontweight="bold")
ax1.set_xlabel("Whole-tumor volume  (cm³)", fontsize=13)
ax1.set_ylabel("Number of patients", fontsize=13)
ax1.tick_params(labelsize=11)
ax1.spines[["top", "right"]].set_visible(False)

# ---- right: centroid location ----------------------------------------
# window covers the full observed centroid range (all 1251 cases) + pad
hb = ax2.hist2d(cx, cy, bins=28, range=[[10, 140], [60, 200]],
                cmap="magma")
ax2.scatter(cx, cy, s=6, c="white", alpha=0.35, linewidths=0)
ax2.invert_yaxis()
ax2.set_xlabel("In-plane position X  (voxels)", fontsize=13)
ax2.set_ylabel("In-plane position Y  (voxels)", fontsize=13)
ax2.tick_params(labelsize=11)
cb = fig.colorbar(hb[3], ax=ax2, fraction=0.046, pad=0.04)
cb.set_label("patients per bin", fontsize=11)

fig.subplots_adjust(left=0.07, right=0.97, top=0.97, bottom=0.12, wspace=0.28)
fig.savefig(OUT / "tumor_distribution.png", dpi=200, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT / 'tumor_distribution.png'}  (n={len(vols)})")
