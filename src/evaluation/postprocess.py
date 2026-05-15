"""Post-processing for ET predictions on BraTS.

Standard BraTS trick: if total predicted ET volume is below a threshold,
relabel all ET voxels (class 3) -> NCR (class 1). This rescues no-ET cases
where a handful of false-positive voxels would otherwise score Dice = 0.
"""
import numpy as np
import torch
from scipy import ndimage


def postprocess_et_labels_torch(labels: torch.Tensor, v_min: int = 1000) -> torch.Tensor:
    """Per-sample ET volume suppression for a (B, D, H, W) integer label tensor.
    If a sample's ET voxel count < v_min, demote ET (3) -> NCR (1).
    Vectorized — no per-sample CPU sync."""
    et = (labels == 3)
    et_counts = et.flatten(1).sum(dim=1)        # (B,) on GPU
    suppress = et_counts < v_min                 # (B,) bool on GPU
    # Broadcast (B,) to (B,D,H,W): demote ET voxels in suppressed samples
    demote_mask = et & suppress[:, None, None, None]
    out = torch.where(demote_mask, torch.full_like(labels, 1), labels)
    return out


def postprocess_et(probs, tau_et=0.5, v_min=1000):
    """Apply ET probability threshold + small-volume suppression.

    probs:   (4, D, H, W) softmax probabilities
    tau_et:  ET probability threshold. Default 0.5 reproduces argmax behaviour.
    v_min:   if predicted ET voxel count < v_min, demote all ET -> NCR (label 1).

    Returns: (D, H, W) uint8 label map with values in {0, 1, 2, 3}.
    """
    pred = probs.argmax(0).astype(np.uint8)

    # 1. ET probability threshold — voxels argmax-ed to ET but below tau become
    #    the runner-up among {0,1,2}.
    if tau_et != 0.5:
        et_mask = probs[3] > tau_et
        demote = (pred == 3) & ~et_mask
        if demote.any():
            runner = probs[[0, 1, 2]].argmax(0).astype(np.uint8)
            pred[demote] = runner[demote]

    # 2. Volume suppression
    et_count = int((pred == 3).sum())
    if et_count < v_min:
        pred[pred == 3] = 1  # demote to NCR (preserves TC region)

    return pred


def postprocess_tc(pred, v_min):
    """Per-component small TC component removal.

    For each connected component of TC = (NCR ∪ ET) in `pred`, if its voxel
    count is below `v_min`, demote NCR(1) and ET(3) within it -> ED(2).
    Demotion target is ED so the WT region (1∪2∪3) is preserved — only the
    core/enhancing labelling inside WT changes.

    `v_min <= 0` is a no-op. 6-connectivity (3D face-adjacent) by default.
    """
    pred = np.asarray(pred).copy()
    if v_min is None or v_min <= 0:
        return pred

    tc_mask = (pred == 1) | (pred == 3)
    if not tc_mask.any():
        return pred

    cc, n = ndimage.label(tc_mask)
    if n == 0:
        return pred

    sizes = ndimage.sum_labels(tc_mask.astype(np.uint8), cc,
                                index=np.arange(1, n + 1))
    small_labels = np.nonzero(sizes < v_min)[0] + 1
    if small_labels.size == 0:
        return pred

    small_mask = np.isin(cc, small_labels)
    pred[small_mask & ((pred == 1) | (pred == 3))] = 2
    return pred


def postprocess_full(probs, tau_et=0.5, et_vmin=1000, tc_vmin=0):
    """Compose ET volume rescue with TC small-component cleanup.

    Order matters: ET rescue (which can demote ET->NCR globally) runs first;
    then TC small-component cleanup operates on the post-rescue label map.
    `tc_vmin <= 0` skips the second pass.
    """
    pred = postprocess_et(probs, tau_et=tau_et, v_min=et_vmin)
    if tc_vmin and tc_vmin > 0:
        pred = postprocess_tc(pred, v_min=tc_vmin)
    return pred
