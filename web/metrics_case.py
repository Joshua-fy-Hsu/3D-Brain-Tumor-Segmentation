"""Per-case derived metrics for the web report: volumes + per-region confidence.

Reuses src.evaluation.metrics for region masks and probability maps.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.append(SRC)

from evaluation import metrics as M  # noqa: E402

REGIONS = ("ET", "TC", "WT")


def region_volumes_ml(labels: np.ndarray, voxel_volume_ml: float) -> dict[str, float]:
    """Returns per-region tumor volume in mL."""
    masks = M.labels_to_regions(labels)
    return {r: float(masks[r].sum()) * float(voxel_volume_ml) for r in REGIONS}


def region_volume_pct(volumes: dict[str, float],
                      brain_ml: float | None) -> dict[str, dict]:
    """Percentage each region occupies, mixed denominators.

    WT is reported as a fraction of total brain volume (tumour burden); TC and
    ET as a fraction of WT (nested-region composition). Each entry is
    {"of": "brain"|"wt", "pct": float|None}.
    """
    wt = float(volumes.get("WT", 0.0) or 0.0)
    out: dict[str, dict] = {}
    out["WT"] = {
        "of": "brain",
        "pct": round(100.0 * wt / brain_ml, 2) if brain_ml and brain_ml > 0 else None,
    }
    for r in ("TC", "ET"):
        v = float(volumes.get(r, 0.0) or 0.0)
        out[r] = {"of": "wt", "pct": round(100.0 * v / wt, 1) if wt > 0 else None}
    return out


def region_confidence(probs_4ch: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    """Mean predicted probability over the predicted region voxels.

    Returns None for a region if its predicted mask is empty.
    """
    masks = M.labels_to_regions(labels)
    region_probs = M.probs_to_region_probs(probs_4ch)
    out: dict[str, float | None] = {}
    for r in REGIONS:
        m = masks[r]
        if not m.any():
            out[r] = None
        else:
            out[r] = round(float(region_probs[r][m].mean()), 4)
    return out
