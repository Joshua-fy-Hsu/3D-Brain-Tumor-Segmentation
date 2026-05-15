"""Uncertainty quality: AURC (Risk-Coverage) and uncertainty-error Spearman correlation."""
import numpy as np
from scipy.stats import spearmanr


def risk_coverage_curve(uncertainty_per_case, risk_per_case, n_points=100):
    """Sort cases by ascending uncertainty, sweep coverage from 1/N to 1.
       At coverage c, risk = mean(risk over least-uncertain c fraction).
       Returns (coverages, risks, aurc)."""
    u = np.asarray(uncertainty_per_case, dtype=np.float64)
    r = np.asarray(risk_per_case, dtype=np.float64)
    mask = np.isfinite(u) & np.isfinite(r)
    u, r = u[mask], r[mask]
    if len(u) == 0:
        return np.array([]), np.array([]), float("nan")
    order = np.argsort(u)
    r_sorted = r[order]
    cum_mean = np.cumsum(r_sorted) / np.arange(1, len(r_sorted) + 1)
    cov = np.arange(1, len(r_sorted) + 1) / len(r_sorted)
    # AURC = area under risk-coverage curve via trapezoidal rule
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    aurc = float(_trapz(cum_mean, cov))
    return cov, cum_mean, aurc


def dice_at_coverage(uncertainty_per_case, dice_per_case, coverages=(0.7, 0.8, 0.9)):
    """Clinical framing of risk-coverage. Sort cases by ascending uncertainty,
    keep the most-confident `c` fraction, return mean Dice on that subset.

    Interpretation: "If a clinician reviews the most uncertain (1-c) of cases,
    the auto-segmented remainder achieves this mean Dice."

    Returns dict {coverage: dice} for each requested coverage level, plus the
    full-coverage baseline (c=1.0) for reference.
    """
    u = np.asarray(uncertainty_per_case, dtype=np.float64)
    d = np.asarray(dice_per_case, dtype=np.float64)
    mask = np.isfinite(u) & np.isfinite(d)
    u, d = u[mask], d[mask]
    if len(u) == 0:
        return {float(c): float("nan") for c in (*coverages, 1.0)}
    order = np.argsort(u)
    d_sorted = d[order]
    n = len(d_sorted)
    out = {}
    for c in coverages:
        k = max(1, int(round(c * n)))
        out[float(c)] = float(np.mean(d_sorted[:k]))
    out[1.0] = float(np.mean(d_sorted))
    return out


def spearman_unc_error(unc_map, error_map, sample=200_000, seed=0):
    """Voxel-level Spearman correlation between uncertainty and binary error.
       Subsamples for speed."""
    u = np.asarray(unc_map).ravel()
    e = np.asarray(error_map).ravel().astype(np.float32)
    if u.size == 0 or e.sum() == 0 or e.sum() == e.size:
        return float("nan")
    if u.size > sample:
        rng = np.random.RandomState(seed)
        idx = rng.choice(u.size, sample, replace=False)
        u, e = u[idx], e[idx]
    rho, _ = spearmanr(u, e)
    return float(rho) if np.isfinite(rho) else float("nan")
