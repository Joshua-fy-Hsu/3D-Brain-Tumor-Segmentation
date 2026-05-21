"""Conclusion hero figure using the hybrid (AURA) checkpoint.

Output: docs/report_figures/conclusion_hero.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgba

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

import torch
from monai.inferers import sliding_window_inference
from model.registry import build_variant
from evaluation._core import DictToSegAdapter

OUT  = ROOT / "docs" / "report_figures"
CKPT = ROOT / "logs" / "run_hybrid_20260518-194903" / "best_model.pth"
CASE = "BraTS2021_01418"

NCR, ED, ET = "#22b34d", "#2666d9", "#dc2929"
SEG_CMAP = ListedColormap([(0,0,0,0), to_rgba(NCR), to_rgba(ED), to_rgba(ET)])
SEG_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], SEG_CMAP.N)


def norm_disp(a):
    fg = a[a != 0]
    if fg.size == 0:
        return a
    lo, hi = np.percentile(fg, (1, 99))
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    cdir = ROOT / "data" / CASE
    img = np.load(cdir / "image.npy").astype(np.float32)
    gt  = np.load(cdir / "mask.npy")

    print("loading AURA (hybrid) checkpoint ...")
    model = build_variant("hybrid")
    sd = torch.load(CKPT, map_location="cpu")
    if any(k.startswith("ema_model.") for k in sd):
        sd = {k.replace("ema_model.", ""): v for k, v in sd.items()
              if k.startswith("ema_model.")}
    model.load_state_dict(sd, strict=False)
    model = DictToSegAdapter(model)
    model.eval().to(device)

    x = torch.from_numpy(img).unsqueeze(0).to(device)
    print("running inference ...")
    with torch.no_grad(), torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
        out = sliding_window_inference(
            x, roi_size=(128,128,128), sw_batch_size=2,
            predictor=model, overlap=0.5, mode="gaussian",
        )
    probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    pred  = np.argmax(probs, axis=0).astype(np.uint8)

    # predictive entropy
    p   = np.clip(probs, 1e-6, 1.0)
    ent = (-(p * np.log(p)).sum(0)) / np.log(probs.shape[0])
    fg  = img[4] > 0
    ent = np.where(fg, ent, 0.0)

    # best axial slice
    z = int((gt > 0).sum(axis=(0,1)).argmax())

    # brain bounding box (square)
    fy, fx = np.where(np.rot90(fg[:,:,z]))
    pad = 10
    H = fg.shape[1]
    y0, y1 = max(fy.min()-pad, 0), min(fy.max()+pad, H)
    x0, x1 = max(fx.min()-pad, 0), min(fx.max()+pad, H)
    side = max(y1-y0, x1-x0)
    cy, cx = (y0+y1)//2, (x0+x1)//2
    y0 = max(cy-side//2, 0); y1 = y0+side
    x0 = max(cx-side//2, 0); x1 = x0+side

    def S(a):
        return np.rot90(a[:,:,z])[y0:y1, x0:x1]

    t1ce  = norm_disp(S(img[1]))
    ent_s = S(ent)
    evmax = max(np.percentile(ent_s[ent_s>0], 99) if (ent_s>0).any() else 1e-3, 1e-3)

    panels = [
        ("T1CE (input)",    t1ce, None),
        ("Ground Truth",    t1ce, S(gt)),
        ("AURA Prediction", t1ce, S(pred)),
        ("Uncertainty",     None, None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    for ax, (title, base, seg) in zip(axes, panels):
        if title == "Uncertainty":
            ax.imshow(np.zeros_like(t1ce), cmap="gray", vmin=0, vmax=1)
            um = np.where(S(fg) & (ent_s > 0), ent_s, np.nan)
            ax.imshow(um**0.6, cmap="inferno", vmin=0, vmax=evmax**0.6)
        else:
            ax.imshow(base, cmap="gray")
            if seg is not None:
                ax.imshow(seg, cmap=SEG_CMAP, norm=SEG_NORM,
                          interpolation="nearest", alpha=0.55)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=8)
        ax.axis("off")

    handles = [plt.Line2D([0],[0], marker="s", linestyle="", markersize=13,
               markerfacecolor=c, markeredgecolor="none", label=l)
               for c, l in [(NCR,"NCR"), (ED,"ED"), (ET,"ET")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=12.5, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Representative case {CASE} — AURA prediction matches the expert, "
                 f"and uncertainty flags the errors",
                 fontsize=13, color="#555", y=1.06)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.10, wspace=0.05)

    out = OUT / "conclusion_hero.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
