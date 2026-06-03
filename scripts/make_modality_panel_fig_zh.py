"""中文版 — 四模態 MRI + 標準分割面板，用於 TumorSeg 競賽簡報。

Output: docs/report_figures/modality_panel_zh.png
"""
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import ListedColormap, BoundaryNorm

rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "Arial"]
rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)

MOD_NAMES = ["T1", "T1CE", "T2", "FLAIR"]
SEG_COLORS = [(0, 0, 0, 0),
              (0.13, 0.70, 0.30, 1),  # 1 NCR  綠
              (0.15, 0.40, 0.85, 1),  # 2 ED   藍
              (0.86, 0.16, 0.16, 1)]  # 3 ET   紅
SEG_CMAP = ListedColormap(SEG_COLORS)
SEG_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], SEG_CMAP.N)


def pick_case(max_scan=60):
    cands = sorted(glob.glob(str(ROOT / "data" / "BraTS2021_*")))[:max_scan]
    best = None
    for d in cands:
        try:
            m = np.load(Path(d) / "mask.npy", mmap_mode="r")
        except Exception:
            continue
        m = np.asarray(m)
        labs = np.unique(m)
        if not {1, 2, 3}.issubset(set(labs.tolist())):
            continue
        tumor_frac = (m > 0).mean()
        if not (0.012 <= tumor_frac <= 0.045):
            continue
        per_slice = (m > 0).sum(axis=(0, 1))
        z = int(per_slice.argmax())
        et_on_slice = int((m[:, :, z] == 3).sum())
        if et_on_slice < 80:
            continue
        score = et_on_slice
        if best is None or score > best[0]:
            best = (score, d, z, tumor_frac)
    if best is None:
        d = cands[0]
        m = np.asarray(np.load(Path(d) / "mask.npy"))
        z = int((m > 0).sum(axis=(0, 1)).argmax())
        return d, z
    return best[1], best[2]


def norm_disp(img2d):
    fg = img2d[img2d != 0]
    if fg.size == 0:
        return np.zeros_like(img2d)
    lo, hi = np.percentile(fg, (1, 99))
    return np.clip((img2d - lo) / (hi - lo + 1e-6), 0, 1)


def main():
    folder, z = pick_case()
    pid = Path(folder).name
    img = np.asarray(np.load(Path(folder) / "image.npy")).astype(np.float32)
    mask = np.asarray(np.load(Path(folder) / "mask.npy"))

    def slc(a):
        return np.rot90(a[:, :, z])

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
    for i, name in enumerate(MOD_NAMES):
        axes[i].imshow(norm_disp(slc(img[i])), cmap="gray")
        axes[i].set_title(name, fontsize=17, fontweight="bold", pad=8)
        axes[i].axis("off")

    axes[4].imshow(norm_disp(slc(img[1])), cmap="gray")
    axes[4].imshow(slc(mask), cmap=SEG_CMAP, norm=SEG_NORM,
                   interpolation="nearest", alpha=0.55)
    axes[4].set_title("標準分割答案", fontsize=17,
                      fontweight="bold", pad=8)
    axes[4].axis("off")

    handles = [plt.Line2D([0], [0], marker="s", linestyle="",
                          markersize=13, markerfacecolor=c, markeredgecolor="none",
                          label=l)
               for c, l in [(SEG_COLORS[3], "ET — 強化腫瘤"),
                            (SEG_COLORS[1], "NCR — 壞死核心"),
                            (SEG_COLORS[2], "ED — 水腫")]]
    fig.suptitle(f"同一軸狀切面，四種 MRI 模態 — 案例 {pid}",
                 fontsize=13, color="#555555", y=0.99)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.16,
                        wspace=0.05)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=13, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(OUT / "modality_panel_zh.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT / 'modality_panel_zh.png'}  (case {pid}, slice {z})")


if __name__ == "__main__":
    main()
