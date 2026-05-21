"""Slide 18 — Qualitative failure: same layout as qualitative_success but for hard cases.

Output: docs/report_figures/qualitative_failure.png
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

NCR, ED, ET = "#22b34d", "#2666d9", "#dc2929"
SEG_CMAP = ListedColormap([(0,0,0,0), to_rgba(NCR), to_rgba(ED), to_rgba(ET)])
SEG_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], SEG_CMAP.N)

# Worst cases by mean Dice (tta_post)
CASES = [
    ("BraTS2021_01628", "Patient BraTS2021_01628  |  ET Dice ≈ 0 · TC Dice 0.25"),
    ("BraTS2021_01508", "Patient BraTS2021_01508  |  ET Dice ≈ 0 · TC Dice 0.02"),
]


def norm_disp(a):
    fg = a[a != 0]
    if fg.size == 0:
        return a
    lo, hi = np.percentile(fg, (1, 99))
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def load_model(device):
    model = build_variant("hybrid")
    sd = torch.load(CKPT, map_location="cpu")
    if any(k.startswith("ema_model.") for k in sd):
        sd = {k.replace("ema_model.", ""): v for k, v in sd.items()
              if k.startswith("ema_model.")}
    model.load_state_dict(sd, strict=False)
    model = DictToSegAdapter(model)
    model.eval().to(device)
    return model


def infer(model, img_np, device):
    x = torch.from_numpy(img_np).unsqueeze(0).to(device)
    with torch.no_grad(), torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
        out = sliding_window_inference(
            x, roi_size=(128,128,128), sw_batch_size=2,
            predictor=model, overlap=0.5, mode="gaussian",
        )
    probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    return np.argmax(probs, axis=0).astype(np.uint8)


def crop_box(fg_slice, H, pad=10):
    fy, fx = np.where(np.rot90(fg_slice))
    if len(fy) == 0:
        return 0, H, 0, H
    y0, y1 = max(fy.min()-pad, 0), min(fy.max()+pad, H)
    x0, x1 = max(fx.min()-pad, 0), min(fx.max()+pad, H)
    side = max(y1-y0, x1-x0)
    cy, cx = (y0+y1)//2, (x0+x1)//2
    y0 = max(cy-side//2, 0); y1 = y0+side
    x0 = max(cx-side//2, 0); x1 = x0+side
    return y0, y1, x0, x1


def render_case(fig, axes_row, img, gt, pred, case_label, show_col_titles=True):
    fg = img[4] > 0
    z  = int((gt > 0).sum(axis=(0,1)).argmax())
    y0, y1, x0, x1 = crop_box(fg[:,:,z], fg.shape[1])

    def S(a):
        return np.rot90(a[:,:,z])[y0:y1, x0:x1]

    t1ce = norm_disp(S(img[1]))
    panels = [
        ("T1CE (input)",    t1ce, None),
        ("Ground Truth",    t1ce, S(gt)),
        ("AURA Prediction", t1ce, S(pred)),
    ]
    for ax, (title, base, seg) in zip(axes_row, panels):
        ax.imshow(base, cmap="gray")
        if seg is not None:
            ax.imshow(seg, cmap=SEG_CMAP, norm=SEG_NORM,
                      interpolation="nearest", alpha=0.55)
        if show_col_titles:
            ax.set_title(title, fontsize=13, fontweight="bold", pad=6)
        ax.axis("off")

    axes_row[1].annotate(
        case_label,
        xy=(0.5, 1.0), xycoords="axes fraction",
        xytext=(0, 38), textcoords="offset points",
        ha="center", va="bottom", fontsize=11, fontweight="bold",
        color="#222", annotation_clip=False,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print("loading AURA (hybrid) checkpoint ...")
    model = load_model(device)

    results = []
    for pid, label in CASES:
        print(f"inferring {pid} ...")
        d = ROOT / "data" / pid
        img = np.load(d / "image.npy").astype(np.float32)
        gt  = np.load(d / "mask.npy")
        pred = infer(model, img, device)
        tumor_voxels = int((gt > 0).sum())
        full_label = f"{label}  ·  Tumor {tumor_voxels:,} voxels"
        results.append((img, gt, pred, full_label))

    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.5*n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (img, gt, pred, label) in enumerate(results):
        render_case(fig, axes[row], img, gt, pred, label, show_col_titles=True)

    handles = [plt.Line2D([0],[0], marker="s", linestyle="", markersize=12,
               markerfacecolor=c, markeredgecolor="none", label=l)
               for c, l in [(NCR,"NCR"), (ED,"ED"), (ET,"ET")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, -0.01))

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.06, wspace=0.04, hspace=0.55)
    out = OUT / "qualitative_failure.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
