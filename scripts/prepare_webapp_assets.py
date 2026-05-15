"""One-time preparation of the web GUI's auxiliary data files.

Outputs (under web/data/):
  population_stats.json   — WT/TC/ET volume percentiles over the BraTS val split.
  aal_brats.nii.gz        — AAL atlas resampled to the BraTS canonical grid.
                            (Generated from an existing AAL NIfTI passed via --aal-source;
                             otherwise we only print instructions and skip.)
  aal_labels.json         — {"label_int": "Region_Name", ...} matching aal_brats.

Usage:
    # Build population stats from the locally cached BraTS val split.
    python scripts/prepare_webapp_assets.py

    # Also resample an external AAL atlas (NN) to the BraTS grid.
    python scripts/prepare_webapp_assets.py \\
        --aal-source path/to/AAL3v1.nii.gz \\
        --aal-labels path/to/aal_labels.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.append(SRC)

from configs import config  # noqa: E402
from evaluation._core import list_val_patients  # noqa: E402
from evaluation.metrics import labels_to_regions  # noqa: E402


def list_all_patients(data_root: str) -> list[str]:
    """Every patient folder that has a ground-truth mask.npy. Used for the
    population reference — it is descriptive (GT volumes, no model involved),
    so the full cohort is a better, more stable reference than the val split."""
    folders = sorted(f for f in os.listdir(data_root)
                     if os.path.isdir(os.path.join(data_root, f)))
    return [p for p in folders
            if os.path.exists(os.path.join(data_root, p, "mask.npy"))]

WEBAPP_DATA = os.path.join(ROOT, "web", "data")
BRATS_REF_SHAPE = (240, 240, 155)


def _voxel_volume_ml_from_dir(pat_dir: str, pid: str) -> float:
    """Read t1.nii to get voxel spacing → mL. Falls back to 0.001 (1mm^3)."""
    for ext in (".nii", ".nii.gz"):
        for mod in ("t1", "flair", "t2", "t1ce"):
            cand = os.path.join(pat_dir, f"{pid}_{mod}{ext}")
            if os.path.exists(cand):
                affine = nib.load(cand).affine
                spacing = np.linalg.norm(affine[:3, :3], axis=0)
                v = float(np.prod(spacing))
                if np.isfinite(v) and v > 0:
                    return v / 1000.0
                return 0.001
    return 0.001


def build_population_stats(out_path: str, scope: str = "all") -> None:
    data_root = config.TRAIN_DATA_PATH
    if not os.path.isdir(data_root):
        print(f"[skip] TRAIN_DATA_PATH not found: {data_root}")
        return

    if scope == "val":
        pids = list_val_patients(data_root, config.TRAIN_COUNT)
    else:
        pids = list_all_patients(data_root)
    if not pids:
        print(f"[skip] no patients ({scope}) under {data_root}")
        return

    print(f"Scanning {len(pids)} patients ({scope}) for tumor volumes ...")
    rows = {"ET": [], "TC": [], "WT": []}
    for pid in tqdm(pids):
        pat_dir = os.path.join(data_root, pid)
        mask_path = os.path.join(pat_dir, "mask.npy")
        if not os.path.exists(mask_path):
            continue
        mask = np.load(mask_path)
        vox_ml = _voxel_volume_ml_from_dir(pat_dir, pid)
        regions = labels_to_regions(mask)
        for r in rows:
            rows[r].append(float(regions[r].sum()) * vox_ml)

    if not rows["WT"]:
        print("[skip] no volumes computed")
        return

    out = {}
    for r, vals in rows.items():
        vals = sorted(float(v) for v in vals)
        out[r] = {
            "p10": float(np.percentile(vals, 10)),
            "p33": float(np.percentile(vals, 33)),
            "p67": float(np.percentile(vals, 67)),
            "p90": float(np.percentile(vals, 90)),
            "n": len(vals),
            "values": vals,
        }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}  (n={out['WT']['n']})")
    for r in ("ET", "TC", "WT"):
        d = out[r]
        print(f"  {r}: p33={d['p33']:.1f} p67={d['p67']:.1f} p90={d['p90']:.1f} mL")


def resample_aal(aal_src: str, aal_labels_src: str | None,
                 out_atlas: str, out_labels: str) -> None:
    """Resample AAL atlas (NN, order=0) to BraTS_REF_SHAPE. Drops affine
    info — the BraTS grid is the runtime target."""
    from scipy.ndimage import zoom
    if not os.path.exists(aal_src):
        print(f"[skip] AAL source not found: {aal_src}")
        return

    img = nib.load(aal_src)
    data = np.asarray(img.dataobj).astype(np.int32)
    if data.ndim != 3:
        print(f"[skip] AAL has unexpected shape {data.shape}")
        return

    factors = tuple(t / s for t, s in zip(BRATS_REF_SHAPE, data.shape))
    resampled = zoom(data, factors, order=0, mode="constant", cval=0)
    resampled = resampled.astype(np.int16)
    os.makedirs(os.path.dirname(out_atlas), exist_ok=True)
    nib.save(nib.Nifti1Image(resampled, np.eye(4)), out_atlas)
    print(f"wrote {out_atlas}  shape={resampled.shape}")

    labels: dict[int, str] = {}
    if aal_labels_src and os.path.exists(aal_labels_src):
        with open(aal_labels_src, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        lid = int(parts[0])
                    except ValueError:
                        try:
                            lid = int(parts[1])
                            name = parts[0]
                            labels[lid] = name
                            continue
                        except ValueError:
                            continue
                    name = parts[1] if len(parts) >= 2 else f"label_{lid}"
                    labels[lid] = name
    else:
        for lid in np.unique(resampled):
            if int(lid) == 0:
                continue
            labels[int(lid)] = f"AAL_{int(lid)}"

    with open(out_labels, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in labels.items()}, f, indent=2)
    print(f"wrote {out_labels}  ({len(labels)} regions)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aal-source", default=None,
                    help="Path to an AAL NIfTI to resample to the BraTS grid.")
    ap.add_argument("--aal-labels", default=None,
                    help="Whitespace-separated label-id<TAB>name file for the AAL atlas.")
    ap.add_argument("--skip-population", action="store_true")
    ap.add_argument("--skip-atlas", action="store_true")
    ap.add_argument("--population-scope", choices=("all", "val"), default="all",
                    help="Cohort for the size-vs-population reference. "
                         "'all' = every patient with a GT mask (default, more "
                         "stable); 'val' = held-out split only.")
    args = ap.parse_args()

    os.makedirs(WEBAPP_DATA, exist_ok=True)

    if not args.skip_population:
        build_population_stats(os.path.join(WEBAPP_DATA, "population_stats.json"),
                               scope=args.population_scope)

    if not args.skip_atlas and args.aal_source:
        resample_aal(
            args.aal_source,
            args.aal_labels,
            os.path.join(WEBAPP_DATA, "aal_brats.nii.gz"),
            os.path.join(WEBAPP_DATA, "aal_labels.json"),
        )
    elif not args.skip_atlas:
        print("[info] --aal-source not given; AAL atlas not generated. "
              "The web UI will render without anatomy until you provide one.")


if __name__ == "__main__":
    main()
