"""Statistical testing for segmentation results.

Two questions this answers:
  1. "Is method A better than method B?"  → paired Wilcoxon signed-rank
     over per-case Dice/HD95/NSD. Returns p-values.
  2. "How tight is the mean estimate?"  → bootstrap 95% CI over per-case
     metrics. Reports mean ± std and [CI_low, CI_high].

Operates on the `per_case_metrics.csv` files written by run_evaluation.
Required columns: `patient_id`, `mode`, and one of the metric columns
(e.g. `dice_ET`, `hd95_TC`, `nsd_WT`, `precision_ET`, ...).

CLI:
  python -m evaluation.stats compare \
      results/cnn/eval_*/per_case_metrics.csv \
      results/transformer/eval_*/per_case_metrics.csv \
      --metric dice_ET --mode tta_post

  python -m evaluation.stats ci \
      results/transformer/eval_*/per_case_metrics.csv \
      --metric dice_ET --mode tta_post

  python -m evaluation.stats summarize-folder \
      results/transformer/eval_*/per_case_metrics.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


REGIONS = ("ET", "TC", "WT")
DEFAULT_METRICS = (
    "dice", "hd95", "nsd",
    "precision", "recall", "sensitivity", "specificity",
)


# ------------------------------------------------------------------------
# Bootstrap CI
# ------------------------------------------------------------------------
def bootstrap_ci(values, n_boot: int = 5000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float, float]:
    """Returns (mean, std, ci_low, ci_high). NaN-safe — drops NaNs first."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"),) * 4
    if v.size == 1:
        return (float(v[0]), 0.0, float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (float(v.mean()), float(v.std(ddof=1)), lo, hi)


# ------------------------------------------------------------------------
# Paired Wilcoxon
# ------------------------------------------------------------------------
def paired_wilcoxon(a, b, alternative: str = "two-sided"
                    ) -> tuple[float, float, int]:
    """Wilcoxon signed-rank on paired samples. Returns (statistic, pvalue, n).
    Pairs with NaN in either side are dropped. n is the post-drop pair count."""
    if not _HAS_SCIPY:
        raise ImportError("scipy is required for Wilcoxon. pip install scipy")
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: a={a.shape} vs b={b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 5:
        return (float("nan"), float("nan"), int(a.size))
    diff = a - b
    if np.allclose(diff, 0):
        return (0.0, 1.0, int(a.size))
    stat, p = wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
    return (float(stat), float(p), int(a.size))


# ------------------------------------------------------------------------
# CSV helpers
# ------------------------------------------------------------------------
def load_per_case(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "patient_id" not in df.columns or "mode" not in df.columns:
        raise ValueError(f"{csv_path} missing patient_id or mode column")
    df["patient_id"] = df["patient_id"].astype(str)
    return df


def select(df: pd.DataFrame, mode: str, metric: str) -> pd.Series:
    """Filter to one (mode, metric) and return values indexed by patient_id."""
    sub = df[df["mode"] == mode]
    if sub.empty:
        raise KeyError(f"mode='{mode}' not found in dataframe. "
                       f"Available: {sorted(df['mode'].unique())}")
    if metric not in sub.columns:
        raise KeyError(f"metric '{metric}' not in columns. "
                       f"Available: {[c for c in sub.columns if c not in ('patient_id','mode')]}")
    return sub.set_index("patient_id")[metric]


def align_pairs(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Inner-join two series by patient_id, return (a_arr, b_arr, pids)."""
    pids = sorted(set(a.index) & set(b.index))
    if not pids:
        raise ValueError("No overlapping patient_ids between the two series.")
    return (a.loc[pids].to_numpy(dtype=np.float64),
            b.loc[pids].to_numpy(dtype=np.float64),
            pids)


# ------------------------------------------------------------------------
# Public API: summarize one CSV across all default metrics × regions × modes
# ------------------------------------------------------------------------
def summarize_per_case(csv_path: str, modes: Iterable[str] | None = None,
                       metrics: Iterable[str] = DEFAULT_METRICS,
                       regions: Iterable[str] = REGIONS,
                       n_boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Walks one per_case_metrics.csv and reports mean ± std + 95% CI for
    every (mode, region, metric) triple. One row per triple.

    Output columns: mode, metric, region, n, mean, std, ci_low, ci_high
    """
    df = load_per_case(csv_path)
    rows = []
    if modes is None:
        modes = sorted(df["mode"].unique())
    for mode in modes:
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        for metric in metrics:
            for region in regions:
                col = f"{metric}_{region}"
                if col not in sub.columns:
                    continue
                v = sub[col].to_numpy(dtype=np.float64)
                m, s, lo, hi = bootstrap_ci(v, n_boot=n_boot, seed=seed)
                rows.append(dict(
                    mode=mode, metric=metric, region=region,
                    n=int(np.isfinite(v).sum()),
                    mean=m, std=s, ci_low=lo, ci_high=hi,
                ))
    return pd.DataFrame(rows)


def compare_two(csv_a: str, csv_b: str, mode: str, metric: str, region: str,
                label_a: str = "A", label_b: str = "B",
                alternative: str = "two-sided", n_boot: int = 5000,
                seed: int = 0) -> dict:
    """Paired Wilcoxon + per-method bootstrap CI for one (mode, metric, region)."""
    df_a = load_per_case(csv_a)
    df_b = load_per_case(csv_b)
    col = f"{metric}_{region}"
    sa = select(df_a, mode, col)
    sb = select(df_b, mode, col)
    a, b, pids = align_pairs(sa, sb)
    stat, p, n = paired_wilcoxon(a, b, alternative=alternative)
    mean_a, std_a, lo_a, hi_a = bootstrap_ci(a, n_boot=n_boot, seed=seed)
    mean_b, std_b, lo_b, hi_b = bootstrap_ci(b, n_boot=n_boot, seed=seed)
    return dict(
        method_a=label_a, method_b=label_b, mode=mode,
        metric=metric, region=region, n_pairs=n,
        a_mean=mean_a, a_std=std_a, a_ci_low=lo_a, a_ci_high=hi_a,
        b_mean=mean_b, b_std=std_b, b_ci_low=lo_b, b_ci_high=hi_b,
        diff_mean=float(mean_a - mean_b),
        wilcoxon_stat=stat, wilcoxon_p=p, alternative=alternative,
    )


def compare_two_full(csv_a: str, csv_b: str, mode: str,
                     label_a: str = "A", label_b: str = "B",
                     metrics: Iterable[str] = DEFAULT_METRICS,
                     regions: Iterable[str] = REGIONS,
                     alternative: str = "two-sided",
                     n_boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Run compare_two over all (metric, region) pairs. One row each."""
    rows = []
    for metric in metrics:
        for region in regions:
            try:
                row = compare_two(csv_a, csv_b, mode, metric, region,
                                  label_a=label_a, label_b=label_b,
                                  alternative=alternative,
                                  n_boot=n_boot, seed=seed)
                rows.append(row)
            except (KeyError, ValueError):
                continue
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------
def _cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_sum = sub.add_parser("summarize", help="mean ± std + 95% CI for one CSV")
    p_sum.add_argument("csv")
    p_sum.add_argument("--mode", default=None,
                       help="If omitted, summarize ALL modes in the CSV.")
    p_sum.add_argument("--out", default=None,
                       help="Write a CSV next to the input. Default: stdout.")

    p_cmp = sub.add_parser("compare", help="paired Wilcoxon between two CSVs")
    p_cmp.add_argument("csv_a")
    p_cmp.add_argument("csv_b")
    p_cmp.add_argument("--mode", required=True)
    p_cmp.add_argument("--label-a", default="A")
    p_cmp.add_argument("--label-b", default="B")
    p_cmp.add_argument("--alternative", default="two-sided",
                       choices=["two-sided", "less", "greater"])
    p_cmp.add_argument("--out", default=None)

    args = ap.parse_args()

    if args.cmd == "summarize":
        modes = [args.mode] if args.mode else None
        out = summarize_per_case(args.csv, modes=modes)
        out_path = args.out or os.path.join(
            os.path.dirname(os.path.abspath(args.csv)), "stats_summary.csv"
        )
        out.to_csv(out_path, index=False)
        print(f"wrote {out_path} ({len(out)} rows)")
        print(out.to_string(index=False))

    elif args.cmd == "compare":
        out = compare_two_full(args.csv_a, args.csv_b, args.mode,
                               label_a=args.label_a, label_b=args.label_b,
                               alternative=args.alternative)
        out_path = args.out or os.path.join(
            os.path.dirname(os.path.abspath(args.csv_a)),
            f"compare_{args.label_a}_vs_{args.label_b}.csv"
        )
        out.to_csv(out_path, index=False)
        print(f"wrote {out_path} ({len(out)} rows)")
        print(out.to_string(index=False))


if __name__ == "__main__":
    _cli()
