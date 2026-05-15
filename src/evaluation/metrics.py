"""Segmentation metrics: Dice, HD95, NSD, AUC, FPV — over BraTS clinical regions ET/TC/WT."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

try:
    from monai.metrics import compute_hausdorff_distance, compute_surface_dice
    _HAS_MONAI = True
except Exception:
    _HAS_MONAI = False

# BraTS clinical regions in terms of label indices (after 4->3 remap):
#   WT = {1,2,3}   TC = {1,3}   ET = {3}
REGIONS = ("ET", "TC", "WT")


def labels_to_regions(label_map):
    """label_map: (D,H,W) int → dict of region binary masks."""
    lm = label_map
    return {
        "ET": (lm == 3),
        "TC": (lm == 1) | (lm == 3),
        "WT": (lm == 1) | (lm == 2) | (lm == 3),
    }


def probs_to_region_probs(probs):
    """probs: (4,D,H,W) softmax → dict of region probability maps (sum of class probs)."""
    return {
        "ET": probs[3],
        "TC": probs[1] + probs[3],
        "WT": probs[1] + probs[2] + probs[3],
    }


def dice_score(pred_bin, gt_bin, eps=1e-6):
    pred_bin = pred_bin.astype(bool)
    gt_bin = gt_bin.astype(bool)
    if not gt_bin.any() and not pred_bin.any():
        return 1.0
    inter = np.logical_and(pred_bin, gt_bin).sum()
    return float((2 * inter + eps) / (pred_bin.sum() + gt_bin.sum() + eps))


def hd95(pred_bin, gt_bin, spacing=(1.0, 1.0, 1.0)):
    if not _HAS_MONAI:
        return float("nan")
    if not pred_bin.any() or not gt_bin.any():
        return float("nan")
    p = torch.from_numpy(pred_bin.astype(np.uint8))[None, None]
    g = torch.from_numpy(gt_bin.astype(np.uint8))[None, None]
    v = compute_hausdorff_distance(p, g, percentile=95, spacing=spacing).item()
    return float(v)


def nsd(pred_bin, gt_bin, tolerance_mm=1.0, spacing=(1.0, 1.0, 1.0)):
    if not _HAS_MONAI:
        return float("nan")
    if not pred_bin.any() or not gt_bin.any():
        return float("nan")
    p = torch.from_numpy(pred_bin.astype(np.uint8))[None, None]
    g = torch.from_numpy(gt_bin.astype(np.uint8))[None, None]
    v = compute_surface_dice(p, g, class_thresholds=[tolerance_mm], spacing=spacing).item()
    return float(v)


def fpv_ml(pred_bin, gt_bin, voxel_volume_ml=0.001):
    """False positive volume in mL. Default 1mm³ voxel = 0.001 mL."""
    fp = np.logical_and(pred_bin, np.logical_not(gt_bin)).sum()
    return float(fp * voxel_volume_ml)


def confusion_counts(pred_bin, gt_bin):
    """Returns (TP, FP, FN, TN) as ints for a single binary region."""
    pb = pred_bin.astype(bool)
    gb = gt_bin.astype(bool)
    tp = int(np.logical_and(pb, gb).sum())
    fp = int(np.logical_and(pb, ~gb).sum())
    fn = int(np.logical_and(~pb, gb).sum())
    tn = int(np.logical_and(~pb, ~gb).sum())
    return tp, fp, fn, tn


def precision_recall_specificity(pred_bin, gt_bin, eps=1e-6):
    """Returns dict with precision, recall (= sensitivity), specificity.
    NaN-safe: if a denominator is 0 because the region is empty in both
    pred and gt, returns 1.0 for that metric (perfect agreement on absence)."""
    tp, fp, fn, tn = confusion_counts(pred_bin, gt_bin)
    if (tp + fp + fn) == 0:
        # Empty in both pred and gt — define as perfect.
        return dict(precision=1.0, recall=1.0, sensitivity=1.0, specificity=1.0)
    precision = tp / max(tp + fp, eps) if (tp + fp) > 0 else float("nan")
    recall = tp / max(tp + fn, eps) if (tp + fn) > 0 else float("nan")
    specificity = tn / max(tn + fp, eps) if (tn + fp) > 0 else float("nan")
    return dict(
        precision=float(precision),
        recall=float(recall),
        sensitivity=float(recall),  # alias — required by professor's spec
        specificity=float(specificity),
    )


def per_region_auc(region_prob, region_gt):
    rg = region_gt.astype(np.uint8).ravel()
    if rg.sum() == 0 or rg.sum() == rg.size:
        return float("nan")
    rp = region_prob.astype(np.float32).ravel()
    # Subsample for speed: 1M voxels is plenty for AUC
    if rp.size > 1_000_000:
        idx = np.random.RandomState(0).choice(rp.size, 1_000_000, replace=False)
        rp, rg = rp[idx], rg[idx]
    return float(roc_auc_score(rg, rp))


def all_metrics(pred_label, probs, gt_label, spacing=(1.0, 1.0, 1.0)):
    """Compute Dice/HD95/NSD/AUC/FPV/Precision/Recall/Sensitivity/Specificity
    per region for a single case."""
    pr_regions = labels_to_regions(pred_label)
    gt_regions = labels_to_regions(gt_label)
    prob_regions = probs_to_region_probs(probs)
    out = {}
    for r in REGIONS:
        pb, gb = pr_regions[r], gt_regions[r]
        out[f"dice_{r}"] = dice_score(pb, gb)
        out[f"hd95_{r}"] = hd95(pb, gb, spacing)
        out[f"nsd_{r}"] = nsd(pb, gb, spacing=spacing)
        out[f"auc_{r}"] = per_region_auc(prob_regions[r], gb)
        out[f"fpv_{r}"] = fpv_ml(pb, gb)
        prs = precision_recall_specificity(pb, gb)
        out[f"precision_{r}"]   = prs["precision"]
        out[f"recall_{r}"]      = prs["recall"]
        out[f"sensitivity_{r}"] = prs["sensitivity"]
        out[f"specificity_{r}"] = prs["specificity"]
    return out


def dice_only_metrics(pred_label, gt_label):
    """Fast Dice-only path (no MONAI HD95/NSD, no AUC). Used by V_min sweep
    where we evaluate dozens of (et_vmin, tc_vmin) grid points per case and
    can't afford the Hausdorff cost."""
    pr_regions = labels_to_regions(pred_label)
    gt_regions = labels_to_regions(gt_label)
    return {f"dice_{r}": dice_score(pr_regions[r], gt_regions[r])
            for r in REGIONS}
