"""Imaging-based malignancy-tendency indicator (DESCRIPTIVE, NOT A DIAGNOSIS).

The BraTS2021 training data this model was built on carries *no* tumour grade
labels (HGG/LGG was dropped after BraTS2020), so a trained, validated grade
classifier is impossible here. Instead we surface a transparent radiomics
heuristic derived purely from the segmentation the model already produces.

Rationale: the classical radiological signature separating high-grade from
low-grade glioma is (1) presence/extent of contrast-enhancing tumour and
(2) central necrosis. High-grade gliomas typically show substantial enhancement
with a necrotic core; low-grade gliomas are usually non-enhancing. We turn the
predicted ET / NCR / ED sub-region volumes into two saturating signals and
combine them into an index in [0, 1].

This is an imaging-feature *description*, explicitly labelled non-diagnostic and
not validated against grade ground truth. It must never be presented as
"predicted stage/grade".

BraTS label convention (post-remap): 1 = NCR (necrotic core), 2 = ED (edema),
3 = ET (enhancing tumour). TC = {1,3}, WT = {1,2,3}.
"""
from __future__ import annotations

import os

import numpy as np

# --- Tunable thresholds (conservative; documented in the UI/report) ----------
# Enhancing fraction (ET / WT) at which the enhancement signal saturates to 1.
ENH_SATURATE = 0.20
# Necrotic fraction (NCR / WT) at which the necrosis signal saturates to 1.
NEC_SATURATE = 0.10
# Relative weights of the two signals in the combined index.
W_ENH = 0.65
W_NEC = 0.35
# Index at/above which imaging features are flagged as high-grade-leaning.
HIGH_INDEX = 0.50
# Absolute ET volume (mL) below which we treat enhancement as negligible —
# guards against a few stray ET voxels in a small tumour reading as "high".
ET_MIN_ML = 1.0
# Necrotic fraction below which necrosis is treated as negligible.
NEC_MIN_FRAC = 0.02
# Whole-tumour volume (mL) below which the enhancement/necrosis *fractions* are
# computed over too few voxels to be reliable. Below this we never assert a
# confident "high-grade" call — a couple of voxels swing the percentages — and
# instead report "indeterminate (small lesion)". The index is still shown.
SMALL_WT_ML = float(os.environ.get("MALIGNANCY_SMALL_WT_ML", "5.0"))

COLORS = {
    "high": "#c0392b",     # high-grade-leaning imaging features
    "indeterminate": "#d9b026",
    "low": "#2d8f2d",      # low-grade-leaning imaging features
    "unknown": "#888888",
}


def _saturate(x: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return float(min(max(x, 0.0) / scale, 1.0))


def assess(labels: np.ndarray, voxel_volume_ml: float) -> dict:
    """Compute the imaging-malignancy-tendency indicator from a label volume.

    Parameters
    ----------
    labels : np.ndarray
        Integer label volume with BraTS classes {0,1,2,3}.
    voxel_volume_ml : float
        Physical volume of one voxel in mL.

    Returns
    -------
    dict with keys: label, label_zh, category, index, color, drivers (list of
    str), features (dict of raw numbers), disclaimer, disclaimer_zh.
    """
    vv = float(voxel_volume_ml)
    et_vox = int(np.count_nonzero(labels == 3))
    ncr_vox = int(np.count_nonzero(labels == 1))
    ed_vox = int(np.count_nonzero(labels == 2))
    wt_vox = et_vox + ncr_vox + ed_vox

    et_ml = et_vox * vv
    ncr_ml = ncr_vox * vv
    wt_ml = wt_vox * vv

    disclaimer = (
        "Imaging-feature description only — derived from the segmentation, "
        "NOT a tumour grade/stage and NOT validated against pathology. "
        "Do not use for diagnosis."
    )
    disclaimer_zh = (
        "僅為影像特徵描述，由分割結果推導，並非腫瘤分級/分期，"
        "且未經病理分級驗證，不可作為診斷依據。"
    )

    if wt_vox == 0:
        return {
            "label": "No tumour segmented",
            "label_zh": "未偵測到腫瘤",
            "category": "unknown",
            "index": None,
            "color": COLORS["unknown"],
            "reliability": "ok",
            "drivers": [],
            "drivers_zh": [],
            "features": {
                "enhancing_fraction": None,
                "necrotic_fraction": None,
                "et_volume_ml": round(et_ml, 2),
                "necrosis_volume_ml": round(ncr_ml, 2),
                "wt_volume_ml": round(wt_ml, 2),
            },
            "disclaimer": disclaimer,
            "disclaimer_zh": disclaimer_zh,
        }

    enh_frac = et_ml / wt_ml if wt_ml > 0 else 0.0
    nec_frac = ncr_ml / wt_ml if wt_ml > 0 else 0.0

    enh_signal = _saturate(enh_frac, ENH_SATURATE) if et_ml >= ET_MIN_ML else 0.0
    nec_signal = _saturate(nec_frac, NEC_SATURATE) if nec_frac >= NEC_MIN_FRAC else 0.0
    index = round(W_ENH * enh_signal + W_NEC * nec_signal, 3)

    negligible_enh = et_ml < ET_MIN_ML
    negligible_nec = nec_frac < NEC_MIN_FRAC

    if index >= HIGH_INDEX:
        category, label, label_zh = (
            "high", "High-grade", "高惡性度")
    elif negligible_enh and negligible_nec:
        category, label, label_zh = (
            "low", "Low-grade", "低惡性度")
    else:
        category, label, label_zh = (
            "indeterminate", "Indeterminate", "不確定")

    # Small-lesion reliability gate: at tiny whole-tumour volumes the ET/NCR
    # fractions are computed over too few voxels to trust, so we never assert a
    # confident "high" — downgrade it to indeterminate and flag limited
    # reliability. The numeric index is still returned for transparency.
    reliability = "ok"
    small_lesion = wt_ml < SMALL_WT_ML
    if small_lesion:
        reliability = "limited"
        if category == "high":
            category, label, label_zh = (
                "indeterminate",
                "Indeterminate (small lesion)",
                "不確定（病灶過小）")

    drivers: list[str] = []
    drivers_zh: list[str] = []
    if small_lesion:
        drivers.append(
            f"Small lesion (whole tumour {wt_ml:.1f} mL < {SMALL_WT_ML:.0f} mL) "
            f"— enhancement/necrosis fractions unreliable")
        drivers_zh.append(
            f"病灶過小（全腫瘤 {wt_ml:.1f} mL < {SMALL_WT_ML:.0f} mL），"
            f"強化／壞死比例不可靠")
    if not negligible_enh:
        drivers.append(
            f"Enhancing tumour {et_ml:.1f} mL ({enh_frac * 100:.0f}% of whole tumour)")
        drivers_zh.append(
            f"強化腫瘤 {et_ml:.1f} mL（占全腫瘤 {enh_frac * 100:.0f}%）")
    else:
        drivers.append("Little to no enhancing tumour")
        drivers_zh.append("幾乎無強化腫瘤")
    if not negligible_nec:
        drivers.append(
            f"Central necrosis {ncr_ml:.1f} mL ({nec_frac * 100:.0f}% of whole tumour)")
        drivers_zh.append(
            f"中央壞死 {ncr_ml:.1f} mL（占全腫瘤 {nec_frac * 100:.0f}%）")
    else:
        drivers.append("No appreciable central necrosis")
        drivers_zh.append("無明顯中央壞死")

    return {
        "label": label,
        "label_zh": label_zh,
        "category": category,
        "index": index,
        "color": COLORS[category],
        "reliability": reliability,
        "drivers": drivers,
        "drivers_zh": drivers_zh,
        "features": {
            "enhancing_fraction": round(enh_frac, 4),
            "necrotic_fraction": round(nec_frac, 4),
            "et_volume_ml": round(et_ml, 2),
            "necrosis_volume_ml": round(ncr_ml, 2),
            "wt_volume_ml": round(wt_ml, 2),
        },
        "disclaimer": disclaimer,
        "disclaimer_zh": disclaimer_zh,
    }
