"""Print the best (et_vmin, tc_vmin) per mode from a vmin_sweep.csv.

Usage:
    python src/evaluation/summarize_vmin_sweep.py
        # auto-discovers the latest results/{cnn,transformer}/eval_*/

    python src/evaluation/summarize_vmin_sweep.py results/transformer/eval_20260507-120000/
"""
import argparse
import json
import os
import sys

import pandas as pd

CURR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CURR))


def latest_eval_dir(arch: str) -> str | None:
    base = os.path.join(ROOT, "results", arch)
    if not os.path.isdir(base):
        return None
    runs = [os.path.join(base, d) for d in os.listdir(base)
            if d.startswith("eval_") and os.path.isdir(os.path.join(base, d))]
    if not runs:
        return None
    runs.sort(key=os.path.getmtime, reverse=True)
    return runs[0]


def summarize(eval_dir: str) -> None:
    sweep_csv = os.path.join(eval_dir, "vmin_sweep.csv")
    if not os.path.exists(sweep_csv):
        print(f"[skip] no vmin_sweep.csv in {eval_dir} — re-run with --vmin-sweep")
        return

    df = pd.read_csv(sweep_csv)
    if "dice_mean" not in df.columns:
        df["dice_mean"] = df[["dice_ET", "dice_TC", "dice_WT"]].mean(axis=1)

    print(f"\n=== {eval_dir} ===")

    meta_path = os.path.join(eval_dir, "evaluation_meta.json")
    cur_et = cur_tc = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        pp = meta.get("postprocess") or {}
        cur_et, cur_tc = pp.get("et_vmin"), pp.get("tc_vmin")
        if cur_et is not None:
            print(f"current run used: et_vmin={cur_et}  tc_vmin={cur_tc}")

    best = df.loc[df.groupby("mode")["dice_mean"].idxmax()].sort_values("mode")
    print("\nBest (et_vmin, tc_vmin) per mode by mean Dice:")
    cols = ["mode", "et_vmin", "tc_vmin", "dice_ET", "dice_TC", "dice_WT", "dice_mean"]
    pd.options.display.float_format = "{:.4f}".format
    print(best[cols].to_string(index=False))

    # If the current-run thresholds aren't already best, show the lift.
    if cur_et is not None:
        cur = df[(df["et_vmin"] == cur_et) & (df["tc_vmin"] == cur_tc)]
        if not cur.empty:
            cur_means = cur.groupby("mode")["dice_mean"].mean()
            best_means = best.set_index("mode")["dice_mean"]
            shared = cur_means.index.intersection(best_means.index)
            if len(shared):
                print("\nGain from switching to per-mode best (vs current run):")
                lift = (best_means.loc[shared] - cur_means.loc[shared]).sort_values(ascending=False)
                for mode, delta in lift.items():
                    print(f"  {mode:>20s}  +{delta:.4f} mean Dice")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", nargs="?", default=None,
                    help="Path to an eval_*/ folder. If omitted, summarises "
                         "the latest under results/cnn/ AND results/transformer/.")
    args = ap.parse_args()

    if args.eval_dir is not None:
        if not os.path.isdir(args.eval_dir):
            print(f"not a directory: {args.eval_dir}", file=sys.stderr)
            sys.exit(2)
        summarize(args.eval_dir)
        return

    found = False
    for arch in ("cnn", "transformer"):
        d = latest_eval_dir(arch)
        if d is not None:
            summarize(d)
            found = True
    if not found:
        print("no eval_*/ folders found under results/{cnn,transformer}/")
        sys.exit(1)


if __name__ == "__main__":
    main()
