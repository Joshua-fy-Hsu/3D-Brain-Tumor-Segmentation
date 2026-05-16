"""Slide 30 conclusion hero — real AURAS prediction vs ground truth.

Runs the pinned `full` (AURAS) checkpoint on the median-Dice representative
case and renders a 4-panel:  T1CE | Ground Truth | AURAS Prediction |
Uncertainty (predictive entropy).  Single sliding-window pass (no TTA/MC),
so it is tractable on CPU but fast on GPU — run it in the CUDA conda env.

Usage (in the env that has CUDA):
    python scripts/make_conclusion_hero.py
    python scripts/make_conclusion_hero.py --case BraTS2021_01418

Output: docs/report_figures/conclusion_hero.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgba

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))            # for `import web.inference`
sys.path.append(str(ROOT / "src"))

from web import inference as INF       # noqa: E402
import torch                           # noqa: E402

NCR, ED, ET = "#22b34d", "#2666d9", "#dc2929"
SEG_CMAP = ListedColormap([(0, 0, 0, 0), to_rgba(NCR), to_rgba(ED),
                           to_rgba(ET)])
SEG_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], SEG_CMAP.N)


def norm_disp(a):
    fg = a[a != 0]
    if fg.size == 0:
        return a
    lo, hi = np.percentile(fg, (1, 99))
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="BraTS2021_01418",
                    help="median-Dice representative case (tta_post)")
    ap.add_argument("--overlap", type=float, default=0.5)
    args = ap.parse_args()

    cdir = ROOT / "data" / args.case
    img = np.asarray(np.load(cdir / "image.npy")).astype(np.float32)  # (5,X,Y,Z)
    gt = np.asarray(np.load(cdir / "mask.npy"))                       # (X,Y,Z)

    print("loading AURAS (full) checkpoint ...")
    loaded = INF.load_model()
    print(f"  checkpoint: {loaded.checkpoint_path}")
    print(f"  device: {loaded.device}")

    x = torch.from_numpy(img).unsqueeze(0).to(loaded.device)
    print("running sliding-window inference (single pass) ...")
    res = INF.run(loaded, x, overlap=args.overlap)
    pred = res.labels                                   # (X,Y,Z) uint8
    probs = res.probs_4ch                               # (4,X,Y,Z)

    # predictive entropy as the uncertainty map, normalised to [0,1]
    p = np.clip(probs, 1e-6, 1.0)
    ent = (-(p * np.log(p)).sum(0)) / np.log(probs.shape[0])
    fg = img[4] > 0
    ent = np.where(fg, ent, 0.0)

    # axial slice with the most tumor in the ground truth
    z = int((gt > 0).sum(axis=(0, 1)).argmax())

    # square brain bounding box on this slice so the tumor fills the panels
    fy, fx = np.where(np.rot90(fg[:, :, z]))
    pad = 10
    H = fg.shape[1]
    y0, y1 = max(fy.min() - pad, 0), min(fy.max() + pad, H)
    x0, x1 = max(fx.min() - pad, 0), min(fx.max() + pad, H)
    side = max(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = max(cy - side // 2, 0); y1 = y0 + side
    x0 = max(cx - side // 2, 0); x1 = x0 + side

    def S(a):
        return np.rot90(a[:, :, z])[y0:y1, x0:x1]

    t1ce = norm_disp(S(img[1]))
    ent_s = S(ent)
    evmax = max(np.percentile(ent_s[ent_s > 0], 99) if (ent_s > 0).any()
                else 1e-3, 1e-3)
    panels = [
        ("T1CE (input)", t1ce, None),
        ("Ground truth", t1ce, S(gt)),
        ("AURAS prediction", t1ce, S(pred)),
        ("Uncertainty", None, None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    for ax, (title, base, seg) in zip(axes, panels):
        if title == "Uncertainty":
            ax.imshow(np.zeros_like(t1ce), cmap="gray", vmin=0, vmax=1)
            um = np.where(S(fg) & (ent_s > 0), ent_s, np.nan)
            ax.imshow(um ** 0.6, cmap="inferno", vmin=0,
                      vmax=evmax ** 0.6)
        else:
            ax.imshow(base, cmap="gray")
            if seg is not None:
                ax.imshow(seg, cmap=SEG_CMAP, norm=SEG_NORM,
                          interpolation="nearest", alpha=0.55)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=8)
        ax.axis("off")

    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=13,
               markerfacecolor=c, markeredgecolor="none", label=l)
               for c, l in [(NCR, "NCR"), (ED, "ED"), (ET, "ET")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=12.5, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Representative case {args.case} — AURAS prediction "
                 f"matches the expert, and uncertainty flags the errors",
                 fontsize=13, color="#555", y=1.06)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.10,
                        wspace=0.05)
    out = ROOT / "docs" / "report_figures" / "conclusion_hero.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
