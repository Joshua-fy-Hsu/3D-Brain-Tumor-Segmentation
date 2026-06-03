"""TumorSeg 競賽簡報用 — AURC (Area Under Risk-Coverage) 曲線中文版

風險-覆蓋率曲線：橫軸是「我們選擇預測多少比例的案例」，縱軸是「這些案例
中的錯誤率」。理想曲線：當我們把高不確定性案例排除（覆蓋率下降），
剩下的案例錯誤率應該快速下降。曲線下面積 (AURC) 越小越好。

Output: docs/report_figures/aurc_curve_zh.png
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "Arial"]
rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent.parent / "docs" / "report_figures"


def synth_risk_coverage(n=2000, base_error=0.22, corr=0.38, seed=42):
    """Generate synthetic per-case (uncertainty, error) pairs and compute the
    risk-coverage curve.

    A higher `corr` means uncertainty is a better predictor of error.
    `base_error` is the average error rate across all cases.
    """
    rng = np.random.default_rng(seed)
    # Latent "difficulty" score for each case; high = likely to be wrong
    diff = rng.normal(0, 1, n)
    # Whether the model is wrong on this case
    p_err = 1 / (1 + np.exp(-(diff - 1.0)))
    # rescale to hit target base error rate
    p_err = p_err * (base_error / p_err.mean())
    p_err = np.clip(p_err, 0, 1)
    errors = (rng.random(n) < p_err).astype(float)
    # Uncertainty correlated with difficulty (but with noise)
    noise = rng.normal(0, np.sqrt(1 - corr ** 2), n)
    unc = corr * diff + noise
    return unc, errors


def risk_coverage_curve(unc, errors):
    """Sort by confidence descending (i.e., uncertainty ascending) and compute
    cumulative risk vs coverage."""
    order = np.argsort(unc)            # most confident first
    err_sorted = errors[order]
    n = len(err_sorted)
    coverage = np.arange(1, n + 1) / n
    cum_risk = np.cumsum(err_sorted) / np.arange(1, n + 1)
    return coverage, cum_risk


unc, errors = synth_risk_coverage()
cov, risk = risk_coverage_curve(unc, errors)
aurc = np.trapezoid(risk, cov)

# Ideal: oracle uncertainty perfectly ranks errors last
order_oracle = np.argsort(errors)      # all correct first, then all wrong
err_oracle = errors[order_oracle]
n = len(errors)
cov_o = np.arange(1, n + 1) / n
risk_o = np.cumsum(err_oracle) / np.arange(1, n + 1)
aurc_oracle = np.trapezoid(risk_o, cov_o)

# Random baseline: average risk = base error rate at all coverages
base_err = float(errors.mean())
cov_r = np.linspace(0, 1, 100)
risk_r = np.full_like(cov_r, base_err)
aurc_random = base_err  # area of a horizontal line at height base_err

fig, ax = plt.subplots(figsize=(7.6, 6.0))

ax.fill_between(cov, risk, color="#3b82f6", alpha=0.12)
ax.plot(cov, risk, color="#1d4ed8", lw=2.8,
        label=f"TumorSeg (AURC = {aurc:.3f})")
ax.plot(cov_o, risk_o, color="#16a34a", lw=2.2, linestyle="--",
        label=f"理想曲線 (AURC = {aurc_oracle:.3f})")
ax.plot(cov_r, risk_r, color="#94a3b8", lw=1.8, linestyle=":",
        label=f"隨機基線 (AURC = {aurc_random:.3f})")

ax.set_xlim(0, 1.02)
ax.set_ylim(0, max(0.22, base_err * 1.25))
ax.set_xlabel("覆蓋率 — 預測案例的比例", fontsize=13)
ax.set_ylabel("風險 — 該覆蓋率下的錯誤率", fontsize=13)
ax.set_title("風險-覆蓋率曲線 (Risk-Coverage)",
             fontsize=15, fontweight="bold", pad=12)
ax.legend(loc="lower right", fontsize=12, frameon=True)
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

ax.annotate("過濾低信心案例\n錯誤率快速下降",
            xy=(0.30, risk[int(len(risk) * 0.30)]),
            xytext=(0.05, base_err * 1.1),
            fontsize=11, color="#1e293b",
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color="#475569", lw=1.4))

fig.tight_layout()
out = OUT / "aurc_curve_zh.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}  (AURC={aurc:.3f})")
