"""Generate figures for the project report.

Outputs land in docs/report_figures/:
  - training_curves_dice.png     mean val Dice vs epoch, one line per variant
  - training_curves_loss.png     train + val loss curves for the `full` model
  - ablation_dice.png            grouped bar chart (Dice ET/TC/WT) across variants
  - ablation_hd95_nsd.png        twin bar charts: HD95 (lower=better) and NSD
  - complexity_tradeoff.png      Dice ET vs Latency scatter, sized by params
  - calibration_ts_effect.png    ECE_pos before/after temperature scaling for `full`
"""
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
OUT = ROOT / "docs" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    ("base_cnn",      "run_base_cnn_phase1_matched_20260509-024641", "Base CNN",       "#888888"),
    ("cross_modal",   "run_cross_modal_phase1_20260508-183620",      "+ Cross-Modal",  "#1f77b4"),
    ("frequency",     "run_frequency_phase2_20260509-135015",        "+ Frequency",    "#2ca02c"),
    ("spectral_swin", "run_spectral_swin_phase3_20260510-003028",     "+ Spectral Swin","#9467bd"),
    ("uncertainty",   "run_uncertainty_phase4_20260510-164216",       "+ Uncertainty",  "#ff7f0e"),
    ("boundary",      "run_boundary_phase5_20260511-075902",           "+ Boundary",     "#d62728"),
    ("full",          "run_full_phase6_20260512-205520",           "Full",           "#000000"),
]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def training_curves_dice():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for variant, run_dir, label, color in VARIANTS:
        log_path = LOGS / run_dir / "training_log.csv"
        if not log_path.exists():
            print(f"[skip] {label}: no training_log.csv at {run_dir}")
            continue
        rows = read_csv(log_path)
        epochs = [int(r["epoch"]) for r in rows]
        dice   = [float(r["val_dice_mean"]) for r in rows]
        lw = 2.2 if variant == "full" else 1.3
        ax.plot(epochs, dice, label=label, color=color, linewidth=lw)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean validation Dice (ET / TC / WT)")
    ax.set_title("Training progression of each variant")
    ax.set_ylim(0.0, 0.85)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves_dice.png", dpi=150)
    plt.close(fig)


def training_curves_loss():
    rows = read_csv(LOGS / VARIANTS[-1][1] / "training_log.csv")
    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss   = [float(r["val_loss"])   for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.0))
    ax.plot(epochs, train_loss, label="Train loss", color="#1f77b4", linewidth=1.5)
    ax.plot(epochs, val_loss,   label="Val loss",   color="#d62728", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train / validation loss — Full variant")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "training_curves_loss.png", dpi=150)
    plt.close(fig)


def ablation_dice():
    rows = read_csv(ROOT / "results" / "final" / "ablation_table.csv")
    labels = [r["display"] for r in rows]
    et = [float(r["Dice_ET"]) for r in rows]
    tc = [float(r["Dice_TC"]) for r in rows]
    wt = [float(r["Dice_WT"]) for r in rows]

    import numpy as np
    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.bar(x - w, et, w, label="Dice ET", color="#1f77b4")
    ax.bar(x,     tc, w, label="Dice TC", color="#2ca02c")
    ax.bar(x + w, wt, w, label="Dice WT", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.70, 0.95)
    ax.set_ylabel("Dice")
    ax.set_title("Per-region Dice across ablation variants (TTA + post-process)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "ablation_dice.png", dpi=150)
    plt.close(fig)


def ablation_hd95_nsd():
    rows = read_csv(ROOT / "results" / "final" / "ablation_table.csv")
    labels = [r["display"] for r in rows]
    hd95 = [float(r["HD95"]) for r in rows]
    nsd  = [float(r["NSD"])  for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    colors = ["#888888"] * len(labels)
    colors[-1] = "#000000"

    axes[0].bar(labels, hd95, color=colors)
    axes[0].set_ylabel("HD95 (mm, lower is better)")
    axes[0].set_title("Boundary distance — HD95")
    axes[0].set_ylim(5.5, 8.0)
    axes[0].grid(True, axis="y", alpha=0.3)
    for tick in axes[0].get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha("right")

    axes[1].bar(labels, nsd, color=colors)
    axes[1].set_ylabel("NSD (higher is better)")
    axes[1].set_title("Surface agreement — NSD")
    axes[1].set_ylim(0.77, 0.81)
    axes[1].grid(True, axis="y", alpha=0.3)
    for tick in axes[1].get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha("right")

    fig.tight_layout()
    fig.savefig(OUT / "ablation_hd95_nsd.png", dpi=150)
    plt.close(fig)


def complexity_tradeoff():
    rows = read_csv(ROOT / "results" / "final" / "ablation_table.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for r in rows:
        params = float(r["Params_M"])
        lat    = float(r["Latency_ms"])
        dice_et = float(r["Dice_ET"])
        size = max(60, params * 14)
        is_full = (r["variant"] == "full")
        ax.scatter(lat, dice_et,
                   s=size,
                   color=("#000000" if is_full else "#1f77b4"),
                   alpha=0.85 if is_full else 0.6,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(r["display"], (lat, dice_et),
                    xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Inference latency (ms / volume, single forward pass)")
    ax.set_ylabel("Dice ET (TTA + post-process)")
    ax.set_title("Accuracy vs. cost — marker area ∝ parameter count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "complexity_tradeoff.png", dpi=150)
    plt.close(fig)


def calibration_ts_effect():
    """ECE_pos for ET / TC / WT, baseline vs. + Temperature Scale, for the Full variant."""
    rows = read_csv(ROOT / "results" / "full" / "eval_phase6" / "summary.csv")
    baseline = next(r for r in rows if r["Method"] == "Transformer baseline")
    ts       = next(r for r in rows if r["Method"] == "+ Temperature Scale")
    # ECE_pos is a single mean across regions in summary.csv. We get it from evaluation_meta.json
    # if available; here we just plot the single mean value pair.
    import json
    meta_path = ROOT / "results" / "full" / "eval_phase6" / "evaluation_meta.json"
    meta = json.loads(meta_path.read_text())

    cal_pos_base = meta["calibration_pos"]["baseline"]
    cal_pos_ts   = meta["calibration_pos"]["ts"]

    regions = ["ET", "TC", "WT"]
    base_vals = [cal_pos_base[r]["ece"] for r in regions]
    ts_vals   = [cal_pos_ts[r]["ece"]   for r in regions]

    import numpy as np
    x = np.arange(len(regions))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(x - w/2, base_vals, w, label="Baseline",          color="#888888")
    ax.bar(x + w/2, ts_vals,   w, label="+ Temperature Scale", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylabel("Positive-only ECE (lower is better)")
    ax.set_title("Calibration improvement from temperature scaling (Full)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "calibration_ts_effect.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # Each figure is independent; a missing input for one must not block the rest.
    for fn in (training_curves_dice, training_curves_loss, ablation_dice,
               ablation_hd95_nsd, complexity_tradeoff, calibration_ts_effect):
        try:
            fn()
        except Exception as e:
            print(f"[skip] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"Wrote available figures to {OUT}")
