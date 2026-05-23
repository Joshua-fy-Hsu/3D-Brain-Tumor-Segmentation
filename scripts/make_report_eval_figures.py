"""One-off: generate report figures (confusion matrix + ROC) and print the
per-region metric / significance numbers for the AURA (=hybrid) report.

Outputs:
  docs/figures/confusion_matrix.png
  docs/figures/roc_curves.png
and prints a STATS block (Dice/Precision/Recall/F1/Specificity/AUC per region
with bootstrap 95% CIs, plus paired Wilcoxon AURA-vs-Baseline / AURA-vs-Complex).

Confusion matrix + ROC use the cached baseline softmax logits + GT in
results/hybrid/_resume_cache/*_tslog.npy / *_tstgt.npy (subsampled voxels,
softmax head). Region defs: WT={1,2,3}, TC={1,3}, ET={3}.
"""
import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "hybrid", "_resume_cache")
FIGDIR = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIGDIR, exist_ok=True)
CLASSES = ["BG", "NCR", "ED", "ET"]
rng = np.random.default_rng(67)

# ---------------------------------------------------------------- load cache
logs, gts = [], []
for f in sorted(glob.glob(os.path.join(CACHE, "*_tslog.npy"))):
    g = f.replace("_tslog.npy", "_tstgt.npy")
    if not os.path.exists(g):
        continue
    logs.append(np.load(f)); gts.append(np.load(g))
L = np.concatenate(logs).astype(np.float64)   # (N,4) logits
G = np.concatenate(gts).astype(np.int64)       # (N,)
# softmax
P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
pred = P.argmax(1)
print(f"[cache] {len(logs)} cases, {len(G)} voxels, GT class counts:",
      {CLASSES[c]: int((G == c).sum()) for c in range(4)})

# ---------------------------------------------------------------- confusion matrix (row-normalized)
cm = confusion_matrix(G, pred, labels=[0, 1, 2, 3]).astype(float)
cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
fig, ax = plt.subplots(figsize=(4.2, 3.6))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "confusion_matrix.png"), dpi=200)
plt.close(fig)
print("[fig] confusion_matrix.png written")

# ---------------------------------------------------------------- ROC per region
REGIONS = {"ET": {3}, "TC": {1, 3}, "WT": {1, 2, 3}}
fig, ax = plt.subplots(figsize=(4.4, 3.8))
for name, cls in REGIONS.items():
    score = P[:, sorted(cls)].sum(1)
    y = np.isin(G, list(cls)).astype(int)
    fpr, tpr, _ = roc_curve(y, score)
    a = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {a:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "roc_curves.png"), dpi=200)
plt.close(fig)
print("[fig] roc_curves.png written")

# ---------------------------------------------------------------- per-case metric stats
def load_mode(variant, mode):
    df = pd.read_csv(os.path.join(ROOT, "results", variant,
                     glob.glob(os.path.join(ROOT, "results", variant, "*"))[0].split(os.sep)[-1]
                     if False else "")) if False else None
    # robust path: find the per_case csv under results/<variant>/**
    cand = glob.glob(os.path.join(ROOT, "results", variant, "**", "per_case_metrics.csv"),
                     recursive=True) + glob.glob(os.path.join(ROOT, "results", variant, "per_case_metrics.csv"))
    df = pd.read_csv(cand[0])
    return df[df["mode"] == mode].copy()

def boot_ci(x, n=10000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

MODE = "tta_post"
variants = {"AURA": "hybrid", "Baseline": "base_cnn", "Complex": "full"}
dfm = {k: load_mode(v, MODE) for k, v in variants.items()}

print(f"\n===== PER-REGION METRICS ({MODE}) =====")
for label, df in dfm.items():
    print(f"\n--- {label} ({variants[label]}) ---")
    for reg in ["ET", "TC", "WT"]:
        d = df[f"dice_{reg}"]; pr = df[f"precision_{reg}"]; rc = df[f"recall_{reg}"]
        sp = df[f"specificity_{reg}"]; au = df[f"auc_{reg}"]
        dm, dl, dh = boot_ci(d)
        f1 = (2 * pr * rc / (pr + rc)).replace([np.inf, -np.inf], np.nan)
        print(f"{reg}: Dice {dm:.3f} [{dl:.3f},{dh:.3f}]  P {pr.mean():.3f}  "
              f"R {rc.mean():.3f}  F1 {f1.mean():.3f}  Spec {sp.mean():.4f}  AUC {au.mean():.3f}")
    md = df[["dice_ET", "dice_TC", "dice_WT"]].mean(1)
    mm, ml, mh = boot_ci(md)
    print(f"mean Dice (per-case avg of 3 regions): {mm:.3f} [{ml:.3f},{mh:.3f}]")

# ---------------------------------------------------------------- paired Wilcoxon: AURA vs others
if HAVE_SCIPY:
    print(f"\n===== PAIRED WILCOXON ({MODE}, Dice) =====")
    a = dfm["AURA"].set_index("patient_id")
    for opp in ["Baseline", "Complex"]:
        b = dfm[opp].set_index("patient_id")
        ids = a.index.intersection(b.index)
        print(f"\nAURA vs {opp}  (n={len(ids)} paired)")
        for reg in ["ET", "TC", "WT"]:
            xa = a.loc[ids, f"dice_{reg}"].values; xb = b.loc[ids, f"dice_{reg}"].values
            try:
                stat, p = wilcoxon(xa, xb)
            except Exception as e:
                p = float("nan")
            print(f"  {reg}: dDice {xa.mean()-xb.mean():+.4f}  p={p:.4g}")
        ma = a.loc[ids, ["dice_ET", "dice_TC", "dice_WT"]].mean(1).values
        mb = b.loc[ids, ["dice_ET", "dice_TC", "dice_WT"]].mean(1).values
        stat, p = wilcoxon(ma, mb)
        print(f"  mean: dDice {ma.mean()-mb.mean():+.4f}  p={p:.4g}")
else:
    print("\n[warn] scipy not available — skipping Wilcoxon")
