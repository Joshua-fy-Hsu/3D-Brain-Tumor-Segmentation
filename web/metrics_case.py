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
