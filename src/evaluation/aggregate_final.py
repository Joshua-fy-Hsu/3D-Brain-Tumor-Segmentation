"""Phase 8 — final aggregator across all trained variants.

Walks `results/<variant>/eval_*/per_case_metrics.csv` for each variant in the
ablation study, produces report-ready CSVs and figures under `results/final/`.

Skips Phase 7 baselines — only the 7 ablation variants are aggregated:
    base_cnn, cross_modal, frequency, spectral_swin, uncertainty, boundary, full

Outputs (all under `results/final/`):
    final_stats.csv           mean/std/95% CI per (variant, mode, region, metric)
    wilcoxon_pvalues.csv      paired Wilcoxon: full vs each other variant on tta_post,
                              Bonferroni-corrected across the 3 regions per metric
    ablation_table.csv        7-row table: Dice ET/TC/WT, HD95, NSD, Params, FLOPs
    complexity_combined.csv   passthrough of results/complexity.csv, indexed by variant
    boxplots/dice_by_method.png   multi-method Dice boxplot (tta_post mode)

Usage:
    python -m evaluation.aggregate_final
    python -m evaluation.aggregate_final --mode tta_post --out results/final
    python -m evaluation.aggregate_final --variants base_cnn cross_modal full
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Iterable

import numpy as np
import pandas as pd

CURR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(CURR)
if SRC not in sys.path:
    sys.path.append(SRC)

from evaluation.stats import (
    bootstrap_ci,
    compare_two_full,
    load_per_case,
    summarize_per_case,
)
from evaluation.visualize import dice_boxplot_multi


# Default variant order for the ablation table. Matches Phase 1→6 progression.
DEFAULT_VARIANTS = (
    "base_cnn",
    "cross_modal",
    "frequency",
    "spectral_swin",
    "uncertainty",
    "boundary",
    "full",
)

REGIONS = ("ET", "TC", "WT")
METRICS = ("dice", "hd95", "nsd",
           "precision", "recall", "sensitivity", "specificity")

# Pretty-name mapping for figures/tables.
DISPLAY_NAMES = {
    "base_cnn": "Base CNN",
    "cross_modal": "+ Cross-Modal",
    "frequency": "+ Frequency",
    "spectral_swin": "+ Spectral Swin",
    "uncertainty": "+ Uncertainty",
    "boundary": "+ Boundary",
    "full": "Full",
}


# --------------------------------------------------------------------------
# Eval-folder discovery
# --------------------------------------------------------------------------
def latest_eval_dir(results_root: str, variant: str) -> str | None:
    """Find the most recent eval_* folder for a variant that contains a
    per_case_metrics.csv. Newest by mtime wins."""
    variant_dir = os.path.join(results_root, variant)
    if not os.path.isdir(variant_dir):
        return None
    candidates = []
    for d in os.listdir(variant_dir):
        full = os.path.join(variant_dir, d)
        csv = os.path.join(full, "per_case_metrics.csv")
        if os.path.isdir(full) and d.startswith("eval") and os.path.isfile(csv):
            candidates.append((os.path.getmtime(full), full))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def discover_variants(results_root: str, variants: Iterable[str]
                      ) -> dict[str, str]:
    """Map variant -> path to its per_case_metrics.csv. Missing variants
    are skipped with a warning."""
    out = {}
    for v in variants:
        d = latest_eval_dir(results_root, v)
        if d is None:
            print(f"  [warn] no eval folder found for variant '{v}' under "
                  f"{os.path.join(results_root, v)} — skipping", file=sys.stderr)
            continue
        out[v] = os.path.join(d, "per_case_metrics.csv")
        print(f"  {v:18s} -> {os.path.relpath(d, results_root)}")
    return out


# --------------------------------------------------------------------------
# 1) final_stats.csv — mean/std/CI per (variant, mode, region, metric)
# --------------------------------------------------------------------------
def build_final_stats(variant_to_csv: dict[str, str], n_boot: int, seed: int
                      ) -> pd.DataFrame:
    """Run summarize_per_case on each variant; concat with a `variant` column."""
    frames = []
    for v, csv in variant_to_csv.items():
        df = summarize_per_case(csv, metrics=METRICS, regions=REGIONS,
                                n_boot=n_boot, seed=seed)
        df.insert(0, "variant", v)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------
# 2) wilcoxon_pvalues.csv — paired Wilcoxon: reference vs each other variant
# --------------------------------------------------------------------------
def build_wilcoxon(variant_to_csv: dict[str, str], reference: str, mode: str,
                   n_boot: int, seed: int) -> pd.DataFrame:
    """Paired Wilcoxon: `reference` vs every other variant on (mode).
    Per-metric Bonferroni correction across the 3 regions."""
    if reference not in variant_to_csv:
        print(f"  [warn] reference variant '{reference}' missing — skipping "
              f"Wilcoxon", file=sys.stderr)
        return pd.DataFrame()
    csv_ref = variant_to_csv[reference]
    rows = []
    for v, csv in variant_to_csv.items():
        if v == reference:
            continue
        df = compare_two_full(csv_ref, csv, mode=mode,
                              label_a=reference, label_b=v,
                              metrics=METRICS, regions=REGIONS,
                              n_boot=n_boot, seed=seed)
        if df.empty:
            continue
        # Bonferroni across 3 regions per metric
        df["bonferroni_p"] = (df["wilcoxon_p"] * len(REGIONS)).clip(upper=1.0)
        df["sig_0.05"] = df["bonferroni_p"] < 0.05
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# --------------------------------------------------------------------------
# 3) ablation_table.csv — per-variant Dice/HD95/NSD + Params/FLOPs
# --------------------------------------------------------------------------
def build_ablation_table(variant_to_csv: dict[str, str], mode: str,
                         complexity_csv: str | None) -> pd.DataFrame:
    """One row per variant. Means come from per_case_metrics.csv at the chosen
    mode. HD95 / NSD are averaged across regions for the single-column view."""
    # Load complexity if available
    cx = None
    if complexity_csv and os.path.isfile(complexity_csv):
        cx = pd.read_csv(complexity_csv).set_index("variant")

    rows = []
    for v, csv in variant_to_csv.items():
        df = load_per_case(csv)
        sub = df[df["mode"] == mode]
        if sub.empty:
            print(f"  [warn] variant '{v}' has no rows for mode='{mode}' — "
                  f"skipping in ablation table", file=sys.stderr)
            continue
        row = {"variant": v, "display": DISPLAY_NAMES.get(v, v),
               "n_cases": int(sub["patient_id"].nunique())}
        for region in REGIONS:
            col = f"dice_{region}"
            if col in sub.columns:
                row[f"Dice_{region}"] = float(sub[col].mean())
        # HD95 / NSD averaged across regions
        for metric in ("hd95", "nsd"):
            vals = []
            for region in REGIONS:
                col = f"{metric}_{region}"
                if col in sub.columns:
                    vals.append(sub[col].to_numpy(dtype=np.float64))
            if vals:
                stacked = np.concatenate(vals)
                stacked = stacked[np.isfinite(stacked)]
                row[metric.upper()] = float(stacked.mean()) if stacked.size else float("nan")
        # Complexity join
        if cx is not None and v in cx.index:
            row["Params_M"] = float(cx.loc[v, "params_total"]) / 1e6
            row["GFLOPs"]   = float(cx.loc[v, "gflops"])
            row["VRAM_MB"]  = float(cx.loc[v, "peak_mem_mb"])
            row["Latency_ms"] = float(cx.loc[v, "latency_mean_ms"])
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4) Multi-method Dice boxplot
# --------------------------------------------------------------------------
def build_dice_boxplot(variant_to_csv: dict[str, str], mode: str,
                       save_path: str) -> None:
    """One box per variant per region, tta_post mode."""
    method_to_df = {}
    for v, csv in variant_to_csv.items():
        df = load_per_case(csv)
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        method_to_df[DISPLAY_NAMES.get(v, v)] = sub
    if not method_to_df:
        print(f"  [warn] no data for boxplot at mode='{mode}'", file=sys.stderr)
        return
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    dice_boxplot_multi(method_to_df, regions=REGIONS, save_path=save_path,
                       title=f"Per-case Dice by variant ({mode})")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default="results",
                    help="Root folder containing per-variant subfolders.")
    ap.add_argument("--out", default="results/final",
                    help="Output directory for aggregated artifacts.")
    ap.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS),
                    help="Variants to aggregate (order preserved in tables).")
    ap.add_argument("--mode", default="tta_post",
                    help="Mode used for tables/boxplot/Wilcoxon. Default tta_post.")
    ap.add_argument("--reference", default="full",
                    help="Reference variant for paired Wilcoxon (full vs each other).")
    ap.add_argument("--complexity-csv", default="results/complexity.csv",
                    help="Optional per-variant complexity CSV to merge into ablation table.")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[1/5] Discovering eval folders under {args.results_root}/ ...")
    variant_to_csv = discover_variants(args.results_root, args.variants)
    if not variant_to_csv:
        print("No variants found. Nothing to do.", file=sys.stderr)
        sys.exit(1)

    print(f"\n[2/5] Building final_stats.csv (mean ± std + 95% CI, "
          f"n_boot={args.n_boot}) ...")
    stats = build_final_stats(variant_to_csv, n_boot=args.n_boot, seed=args.seed)
    stats_path = os.path.join(args.out, "final_stats.csv")
    stats.to_csv(stats_path, index=False)
    print(f"  wrote {stats_path} ({len(stats)} rows)")

    print(f"\n[3/5] Paired Wilcoxon — {args.reference} vs each other variant "
          f"on mode='{args.mode}' ...")
    wil = build_wilcoxon(variant_to_csv, reference=args.reference,
                         mode=args.mode, n_boot=args.n_boot, seed=args.seed)
    wil_path = os.path.join(args.out, "wilcoxon_pvalues.csv")
    wil.to_csv(wil_path, index=False)
    print(f"  wrote {wil_path} ({len(wil)} rows)")

    print(f"\n[4/5] Ablation table (mode='{args.mode}') ...")
    abl = build_ablation_table(variant_to_csv, mode=args.mode,
                               complexity_csv=args.complexity_csv)
    abl_path = os.path.join(args.out, "ablation_table.csv")
    abl.to_csv(abl_path, index=False)
    print(f"  wrote {abl_path} ({len(abl)} rows)")
    if not abl.empty:
        print(abl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Complexity passthrough
    if os.path.isfile(args.complexity_csv):
        cx = pd.read_csv(args.complexity_csv)
        # Keep only the variants we aggregated, preserve their order
        cx = cx[cx["variant"].isin(variant_to_csv.keys())]
        order = {v: i for i, v in enumerate(args.variants)}
        cx = cx.assign(_o=cx["variant"].map(order)).sort_values("_o").drop(columns="_o")
        cx_path = os.path.join(args.out, "complexity_combined.csv")
        cx.to_csv(cx_path, index=False)
        print(f"  wrote {cx_path} ({len(cx)} rows)")

    print(f"\n[5/5] Multi-variant Dice boxplot ...")
    box_path = os.path.join(args.out, "boxplots", "dice_by_method.png")
    build_dice_boxplot(variant_to_csv, mode=args.mode, save_path=box_path)
    print(f"  wrote {box_path}")

    # Manifest
    manifest = {
        "results_root": os.path.abspath(args.results_root),
        "variants": list(variant_to_csv.keys()),
        "missing_variants": [v for v in args.variants if v not in variant_to_csv],
        "mode": args.mode,
        "reference": args.reference,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "sources": {v: os.path.relpath(csv, args.results_root)
                    for v, csv in variant_to_csv.items()},
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone. Output dir: {args.out}/")


if __name__ == "__main__":
    main()
