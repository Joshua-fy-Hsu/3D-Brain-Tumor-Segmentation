"""Merge CNN and Transformer evaluation runs into a single comparison table.

Auto-detects the latest `eval_*` folder under `results/cnn/` and
`results/transformer/`, loads each run's `summary.csv`, prefixes the Method
column with the architecture, concatenates them, and writes a unified
`results/compare/compare_YYYYMMDD-HHMMSS/summary_combined.csv` (+ a printed
table on stdout).

Also merges `dice_at_coverage.csv` and `evaluation_meta.json` from each run
when present, so the report can pull a single source of truth.

Usage:
    # auto-detect latest eval_* folders
    python src/evaluation/compare.py

    # point at specific runs
    python src/evaluation/compare.py \\
        --cnn-dir results/cnn/eval_20260101-120000 \\
        --transformer-dir results/transformer/eval_20260101-130000

    # write to a custom output dir (no timestamp subfolder)
    python src/evaluation/compare.py --out results/compare/latest --no-subfolder
"""
import argparse
import datetime
import json
import os
import sys

import pandas as pd

CURR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(CURR)
ROOT = os.path.dirname(SRC)


def find_latest_eval_dir(arch_dir):
    """Newest results/<arch>/eval_*/ folder, by mtime. Returns None if missing."""
    if not os.path.isdir(arch_dir):
        return None
    runs = [os.path.join(arch_dir, d) for d in os.listdir(arch_dir)
            if d.startswith("eval_") and os.path.isdir(os.path.join(arch_dir, d))]
    runs = [r for r in runs if os.path.exists(os.path.join(r, "summary.csv"))]
    if not runs:
        return None
    runs.sort(key=os.path.getmtime, reverse=True)
    return runs[0]


def load_summary(eval_dir, arch_label):
    """Load summary.csv and prefix Method with the arch label."""
    path = os.path.join(eval_dir, "summary.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.insert(0, "Arch", arch_label)
    df["Method"] = arch_label + " " + df["Method"].astype(str)
    return df


def load_dice_at_cov(eval_dir, arch_label):
    path = os.path.join(eval_dir, "dice_at_coverage.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.insert(0, "Arch", arch_label)
    return df


def load_meta(eval_dir):
    path = os.path.join(eval_dir, "evaluation_meta.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(
        description="Merge CNN + Transformer eval runs into one comparison table.")
    ap.add_argument("--cnn-dir", default=None,
                    help="Path to a results/cnn/eval_* folder. Auto-detected if omitted.")
    ap.add_argument("--transformer-dir", default=None,
                    help="Path to a results/transformer/eval_* folder. Auto-detected if omitted.")
    ap.add_argument("--out", default=None,
                    help="Output directory. Default: results/compare/compare_YYYYMMDD-HHMMSS/")
    ap.add_argument("--no-subfolder", action="store_true",
                    help="Write directly into --out without a timestamped subfolder.")
    args = ap.parse_args()

    cnn_dir = args.cnn_dir or find_latest_eval_dir(os.path.join(ROOT, "results", "cnn"))
    tr_dir = args.transformer_dir or find_latest_eval_dir(os.path.join(ROOT, "results", "transformer"))

    if cnn_dir is None and tr_dir is None:
        sys.exit("[compare] no eval runs found under results/cnn/ or results/transformer/")
    print(f"[compare] CNN run         : {cnn_dir}")
    print(f"[compare] Transformer run : {tr_dir}")

    base_out = args.out or os.path.join(ROOT, "results", "compare")
    if args.no_subfolder:
        out_dir = base_out
    else:
        out_dir = os.path.join(
            base_out, "compare_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[compare] output dir      : {out_dir}")

    # --- summary.csv merge ---
    parts = []
    if cnn_dir is not None:
        df = load_summary(cnn_dir, "CNN")
        if df is not None:
            parts.append(df)
    if tr_dir is not None:
        df = load_summary(tr_dir, "Transformer")
        if df is not None:
            parts.append(df)

    if not parts:
        sys.exit("[compare] neither run had a summary.csv")

    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(os.path.join(out_dir, "summary_combined.csv"), index=False)
    pd.options.display.float_format = "{:.4f}".format
    print("\n========== COMBINED SUMMARY ==========")
    print(combined.to_string(index=False))

    # --- dice_at_coverage.csv merge ---
    cov_parts = []
    if cnn_dir is not None:
        df = load_dice_at_cov(cnn_dir, "CNN")
        if df is not None:
            cov_parts.append(df)
    if tr_dir is not None:
        df = load_dice_at_cov(tr_dir, "Transformer")
        if df is not None:
            cov_parts.append(df)
    if cov_parts:
        cov_df = pd.concat(cov_parts, ignore_index=True)
        cov_df.to_csv(os.path.join(out_dir, "dice_at_coverage_combined.csv"), index=False)
        print("\n========== DICE @ COVERAGE ==========")
        print(cov_df.to_string(index=False))

    # --- meta merge ---
    combined_meta = {}
    if cnn_dir is not None:
        m = load_meta(cnn_dir)
        if m is not None:
            combined_meta["cnn"] = {"run_dir": cnn_dir, **m}
    if tr_dir is not None:
        m = load_meta(tr_dir)
        if m is not None:
            combined_meta["transformer"] = {"run_dir": tr_dir, **m}
    if combined_meta:
        with open(os.path.join(out_dir, "evaluation_meta_combined.json"), "w") as f:
            json.dump(combined_meta, f, indent=2, default=float)

    print(f"\n[compare] wrote: {out_dir}")


if __name__ == "__main__":
    main()
