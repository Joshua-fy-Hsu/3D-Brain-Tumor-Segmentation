"""On-the-fly preprocessing for uploaded NIfTI files.

Replicates the per-patient normalization in src/preprocessing/optimizing.py
(build_npy_for_patient) without writing a stats.json to disk: per-modality
mean/std are computed from the upload itself, brain-masked, same as
optimize_from_raw() at line 154-158.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import nibabel as nib
import numpy as np
import torch

MODALITIES = ("t1", "t1ce", "t2", "flair")


class PreprocessError(ValueError):
    """Raised on shape mismatch or invalid upload."""


@dataclass
class PreprocessedCase:
    x: torch.Tensor                # (1, 5, D, H, W) float32 on `device`
    ref_affine: np.ndarray         # affine of the T1CE NIfTI (carried for save_nifti)
    ref_header: nib.Nifti1Header   # T1CE header
    spatial_shape: Tuple[int, int, int]
    voxel_volume_ml: float         # product of |spacing| / 1000


def _load_modality(path: str) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    img = nib.load(path)
    data = np.asarray(img.dataobj).astype(np.float32)
    return data, img.affine, img.header


def _voxel_volume_ml(affine: np.ndarray) -> float:
    """Voxel volume in mL from a 4x4 affine. BraTS 1 mm^3 → 0.001 mL."""
    try:
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        vol_mm3 = float(np.prod(spacing))
        if not np.isfinite(vol_mm3) or vol_mm3 <= 0:
            return 0.001
        return vol_mm3 / 1000.0
    except Exception:
        return 0.001


def build_input(modality_paths: dict[str, str], device: torch.device) -> PreprocessedCase:
    """Load 4 NIfTI modalities, z-score per-modality with brain-masked stats,
    stack with a foreground channel, return a (1, 5, D, H, W) float32 tensor.

    Args:
        modality_paths: {"t1": "...", "t1ce": "...", "t2": "...", "flair": "..."}.
        device: torch.device for the output tensor.
    """
    missing = [m for m in MODALITIES if m not in modality_paths]
    if missing:
        raise PreprocessError(f"Missing modalities: {missing}")

    raw, affines, headers = [], [], []
    for m in MODALITIES:
        path = modality_paths[m]
        if not os.path.exists(path):
            raise PreprocessError(f"File not found for {m}: {path}")
        data, affine, header = _load_modality(path)
        raw.append(data)
        affines.append(affine)
        headers.append(header)

    shapes = [r.shape for r in raw]
    if len(set(shapes)) > 1:
        raise PreprocessError(
            f"Modality shape mismatch: {dict(zip(MODALITIES, shapes))}"
        )
    if len(shapes[0]) != 3:
        raise PreprocessError(f"Expected 3D volumes, got shape {shapes[0]}")

    # Foreground mask: any non-zero voxel across the 4 raw modalities.
    foreground = np.any([c != 0 for c in raw], axis=0).astype(np.float32)

    # Per-modality z-score on brain-masked voxels (matches optimize_from_raw).
    normed = []
    for r in raw:
        m = r > 0
        if m.any():
            mean = float(r[m].mean())
            std = float(r[m].std())
        else:
            mean, std = 0.0, 1.0
        ch = (r - mean) / (std + 1e-8)
        ch[foreground == 0] = 0.0
        normed.append(ch.astype(np.float32))
    normed.append(foreground.astype(np.float32))

    arr = np.stack(normed, axis=0)  # (5, D, H, W)
    x = torch.from_numpy(arr).unsqueeze(0).contiguous().to(device, non_blocking=True)

    # T1CE is the reference space for save + Niivue render (index 1 in MODALITIES).
    ref_idx = MODALITIES.index("t1ce")
    return PreprocessedCase(
        x=x,
        ref_affine=affines[ref_idx],
        ref_header=headers[ref_idx],
        spatial_shape=tuple(shapes[0]),
        voxel_volume_ml=_voxel_volume_ml(affines[ref_idx]),
    )
