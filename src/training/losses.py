"""Training losses.

Exports the region-wise softmax/sigmoid Dice+Focal losses plus the
`UncertaintyAwareLoss` and `BoundaryAwareLoss` wrappers used by the `full`
model. Imported by `train_variant.py`.

UncertaintyAwareLoss
--------------------
Aleatoric-uncertainty regularizer for the Phase-4 variance head:

    L_total = L_seg + λ_unc * |variance.mean() - target|

The model emits a non-negative variance map (Softplus tail, see
`src/model/blocks/uncertainty_bottleneck.py`). We don't have access to a
per-voxel "ground-truth" variance, so we use a single scalar target as a soft
anchor: the loss penalizes the *mean* variance from drifting either too far
above or too far below the chosen target.

Default `target_unc_at_high_dice = 0.0`. With Softplus the variance head's
output is bounded below by 0; the mean can only be ≥ 0, so |mean − 0| =
mean and the regularizer simply discourages large variance overall. This
is a defensible default because:

  - At init the variance head is small (Kaiming) and softplus(small) ≈ ln(2)
    ≈ 0.69, so mean variance is bounded above.
  - λ_unc = 0.05 is small relative to the segmentation loss (typically ~1.0
    early in training), so the regularizer can't dominate.
  - The block's gating α starts at 0 and only departs from 0 if variance
    correlates with error. The penalty against large mean variance acts as a
    weak prior: if growing variance doesn't help α (i.e. doesn't reduce seg
    loss), it gets squeezed back toward 0. AURC/Spearman are the actual
    targets; this is a stabilizer, not the objective.

Larger non-zero targets (e.g. 0.1) bias the head toward emitting moderate
variance everywhere — useful only if you observe the head collapsing to ~0
and want to keep it alive. We default to 0 and expose `target` as a knob.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Region-wise softmax Dice + Focal — for 4-channel (BG/NCR/ED/ET) heads.
# ---------------------------------------------------------------------------
class RegionWiseDiceFocalLoss(nn.Module):
    """
    Combined Dice + Focal loss for three clinical tumor regions:
      WT (Whole Tumor):    labels 1+2+3
      TC (Tumor Core):     labels 1+3
      ET (Enhancing):      label 3

    Focal loss down-weights easy, well-classified voxels so training focuses
    on hard/uncertain ones (especially small ET regions).
    Supports deep supervision with weighted sum (1.0 / 0.5 / 0.25).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        smooth: float = 1e-5,
        ce_weight: float = 0.0,
        class_weights: Optional[Sequence[float]] = None,
        region_weights: Sequence[float] = (1.0, 1.0, 1.0),
    ):
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth
        self.ce_weight = float(ce_weight)
        # Per-region multiplier on (dice + focal), order = (WT, TC, ET) to
        # match the region loop in forward_single. Default (1,1,1) is a
        # no-op so every existing variant is byte-identical.
        rw = tuple(float(w) for w in region_weights)
        assert len(rw) == 3, "region_weights must have 3 entries (WT, TC, ET)"
        self.region_weights = rw
        if class_weights is not None:
            cw = torch.as_tensor(class_weights, dtype=torch.float32)
            assert cw.numel() == 4, "class_weights must have 4 entries"
            self.register_buffer("class_weights", cw)
        else:
            self.class_weights = None

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Binary focal loss on probability inputs (not logits)."""
        pred = pred.clamp(1e-7, 1.0 - 1e-7)
        bce = -(target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
        p_t = pred * target + (1.0 - pred) * (1.0 - target)
        focal_weight = (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()

    def forward_single(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(inputs, dim=1)
        targets_oh = F.one_hot(targets, num_classes=4).permute(0, 4, 1, 2, 3).float()

        pred_wt = probs[:, 1] + probs[:, 2] + probs[:, 3]
        pred_tc = probs[:, 1] + probs[:, 3]
        pred_et = probs[:, 3]

        tgt_wt = targets_oh[:, 1] + targets_oh[:, 2] + targets_oh[:, 3]
        tgt_tc = targets_oh[:, 1] + targets_oh[:, 3]
        tgt_et = targets_oh[:, 3]

        loss = 0.0
        regions = [(pred_wt, tgt_wt), (pred_tc, tgt_tc), (pred_et, tgt_et)]
        for w, (pred_r, tgt_r) in zip(self.region_weights, regions):
            loss = loss + w * (self._dice_loss(pred_r, tgt_r)
                               + self._focal_loss(pred_r, tgt_r))

        if self.ce_weight > 0.0:
            cw = self.class_weights
            if cw is not None and cw.device != inputs.device:
                cw = cw.to(inputs.device)
                self.class_weights = cw
            loss = loss + self.ce_weight * F.cross_entropy(inputs, targets, weight=cw)

        return loss

    def forward(self, inputs_list, targets: torch.Tensor) -> torch.Tensor:
        loss = self.forward_single(inputs_list[0], targets)

        if len(inputs_list) > 1:
            t_ds1 = F.interpolate(
                targets.unsqueeze(1).float(), scale_factor=0.5, mode='nearest'
            ).squeeze(1).long()
            loss = loss + 0.5 * self.forward_single(inputs_list[1], t_ds1)

        if len(inputs_list) > 2:
            t_ds2 = F.interpolate(
                targets.unsqueeze(1).float(), scale_factor=0.25, mode='nearest'
            ).squeeze(1).long()
            loss = loss + 0.25 * self.forward_single(inputs_list[2], t_ds2)

        return loss


# ---------------------------------------------------------------------------
# Region-wise sigmoid Dice + Focal — for 3-channel (ET/TC/WT) region heads.
# ---------------------------------------------------------------------------
class RegionWiseDiceFocalSigmoidLoss(nn.Module):
    """Model outputs (B, 3, D, H, W) logits — channel order: [ET, TC, WT]."""

    REGION_ORDER = ("ET", "TC", "WT")

    def __init__(self, gamma: float = 2.0, smooth: float = 1e-5):
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth

    @staticmethod
    def labels_to_regions(targets: torch.Tensor) -> torch.Tensor:
        """(B,D,H,W) long -> (B,3,D,H,W) float in {0,1}, channel order ET/TC/WT."""
        et = (targets == 3).float()
        tc = ((targets == 1) | (targets == 3)).float()
        wt = (targets > 0).float()
        return torch.stack([et, tc, wt], dim=1)

    def _dice(self, p, t):
        inter = (p * t).sum(dim=(2, 3, 4))
        union = p.sum(dim=(2, 3, 4)) + t.sum(dim=(2, 3, 4))
        return 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth)).mean()

    def _focal(self, p, t):
        # Bf16-safe clamp bounds.
        p = p.clamp(0.01, 0.99)
        bce = -(t * torch.log(p) + (1.0 - t) * torch.log(1.0 - p))
        p_t = p * t + (1.0 - p) * (1.0 - t)
        return ((1.0 - p_t) ** self.gamma * bce).mean()

    def forward_single(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        tgt = self.labels_to_regions(targets)
        return self._dice(probs, tgt) + self._focal(probs, tgt)

    def forward(self, inputs_list, targets: torch.Tensor) -> torch.Tensor:
        loss = self.forward_single(inputs_list[0], targets)
        if len(inputs_list) > 1:
            t_ds1 = F.interpolate(
                targets.unsqueeze(1).float(), scale_factor=0.5, mode='nearest'
            ).squeeze(1).long()
            loss = loss + 0.5 * self.forward_single(inputs_list[1], t_ds1)
        if len(inputs_list) > 2:
            t_ds2 = F.interpolate(
                targets.unsqueeze(1).float(), scale_factor=0.25, mode='nearest'
            ).squeeze(1).long()
            loss = loss + 0.25 * self.forward_single(inputs_list[2], t_ds2)
        return loss


def regions_to_labels(probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert (B, 3, D, H, W) sigmoid probs (ET/TC/WT) to (B, D, H, W) labels."""
    et = probs[:, 0] > threshold
    tc = probs[:, 1] > threshold
    wt = probs[:, 2] > threshold
    labels = torch.zeros_like(probs[:, 0], dtype=torch.long)
    labels[wt] = 2
    labels[tc] = 1
    labels[et] = 3
    return labels


# ---------------------------------------------------------------------------
# Phase 4 — Uncertainty-aware composite loss
# ---------------------------------------------------------------------------
class UncertaintyAwareLoss(nn.Module):
    """Wraps a base segmentation loss and adds a small variance regularizer.

        L = L_seg + λ_unc * |variance.mean() - target|

    Args:
        seg_loss: base segmentation loss (e.g. RegionWiseDiceFocalLoss). Must
            implement `forward(inputs_list, targets)` and `forward_single`.
        lambda_unc: weight on the variance term. Default 0.05 — small enough
            to not dominate L_seg early.
        target_unc_at_high_dice: scalar anchor for variance.mean(). Default
            0.0 (encourage low variance overall). Non-zero values bias the
            head toward a moderate baseline if you observe collapse to 0.

    The wrapped loss exposes `forward_single` so the val loop in
    `train_variant.py` (which calls `criterion.forward_single(preds, targets)`)
    keeps working unchanged for the seg portion.
    """

    def __init__(
        self,
        seg_loss: nn.Module,
        lambda_unc: float = 0.05,
        target_unc_at_high_dice: float = 0.0,
    ):
        super().__init__()
        self.seg_loss = seg_loss
        self.lambda_unc = float(lambda_unc)
        self.target = float(target_unc_at_high_dice)

    # ----- pass-through to the wrapped seg loss -----
    def forward_single(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.seg_loss.forward_single(inputs, targets)

    # ----- Phase-4 composite loss -----
    def forward(
        self,
        inputs_list,
        targets: torch.Tensor,
        variance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = self.seg_loss(inputs_list, targets)
        if variance is not None and self.lambda_unc > 0.0:
            v_mean = variance.float().mean()
            loss = loss + self.lambda_unc * (v_mean - self.target).abs()
        return loss


# ---------------------------------------------------------------------------
# Phase 5 — Boundary-aware composite loss
# ---------------------------------------------------------------------------
def _morph_edge_gt(targets: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    """Online morphological-gradient edge mask from a label volume.

    Args:
        targets: (B, D, H, W) long, label values in [0, num_classes).
    Returns:
        edge: (B, 1, D, H, W) float in {0, 1}. 1 where any class boundary lies.

    Implementation: one-hot → max-pool dilation (k=3, s=1, p=1) - one-hot
    yields a per-class non-negative difference (1 inside, dilation == identity
    in the bulk → 0; dilation > one-hot on the inside-edge of each class →
    positive). `any(dim=1)` ORs across classes, recovering a tight 1-voxel
    boundary mask.

    Computed inline so no preprocessing step is needed. F.max_pool3d on a
    one-hot float tensor is a fast surrogate for binary dilation.
    """
    onehot = F.one_hot(targets.long(), num_classes=num_classes)            # (B,D,H,W,C)
    onehot = onehot.permute(0, 4, 1, 2, 3).float()                          # (B,C,D,H,W)
    dilated = F.max_pool3d(onehot, kernel_size=3, stride=1, padding=1)
    diff = (dilated - onehot).abs()
    edge = (diff.amax(dim=1, keepdim=True) > 0).float()                     # (B,1,D,H,W)
    return edge


def _downsample_edge(edge: torch.Tensor, scale: int) -> torch.Tensor:
    """Downsample an edge mask by an integer factor using max-pool so thin
    1-voxel edges are preserved (nearest-interp would halve them)."""
    if scale == 1:
        return edge
    return F.max_pool3d(edge, kernel_size=scale, stride=scale)


def _softmax_region_probs(seg_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """4-class softmax logits → (WT, TC, ET) probability maps, each (B, D, H, W)."""
    probs = torch.softmax(seg_logits, dim=1)
    wt = probs[:, 1] + probs[:, 2] + probs[:, 3]
    tc = probs[:, 1] + probs[:, 3]
    et = probs[:, 3]
    return wt, tc, et


def _label_region_masks(targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(B,D,H,W) long → (WT, TC, ET) binary masks, each (B, D, H, W) float."""
    wt = (targets > 0).float()
    tc = ((targets == 1) | (targets == 3)).float()
    et = (targets == 3).float()
    return wt, tc, et


class BoundaryAwareLoss(nn.Module):
    """Wraps a base seg loss (or an UncertaintyAwareLoss-wrapped one) and adds
    per-stage boundary supervision:

        L_total = L_seg + λ_b · (bce_weight · BCE + edge_dice_weight · EdgeDice)

    `BCE` is binary cross-entropy with logits between each boundary head's
    output and the corresponding-resolution morphological-gradient edge GT.
    `EdgeDice` is an edge-restricted Dice on the seg path: build region probs
    (WT/TC/ET) from the seg deep-sup logit at the matching resolution,
    multiply both prediction and GT region maps by the edge mask, then 1 − Dice
    averaged over the three regions.

    Deep-supervision weights mirror the seg path: 1.0 / 0.5 / 0.25 for
    (full / 1/2 / 1/4) resolution.

    Args:
        base_loss: an `nn.Module` implementing `forward(inputs_list, targets)`
            and `forward_single`. Either `RegionWiseDiceFocalLoss(...)` for a
            boundary-only variant, or `UncertaintyAwareLoss(...)` for the
            `full` variant that stacks Phase-4 + Phase-5.
        lambda_boundary: steady-state λ_b. The trainer applies a linear warm-up
            from `lambda_boundary_start` → `lambda_boundary` over
            `ramp_epochs` epochs via `set_lambda` once per epoch.
        bce_weight / edge_dice_weight: relative weights of the two boundary
            terms before λ_b. Defaults (0.3 / 0.2) keep the magnitudes
            comparable to the seg Dice term.
        lambda_boundary_start: λ_b at epoch 0. Default 0.1.
        ramp_epochs: epochs to ramp from start → steady. Default 50.
        num_classes: 4 for the softmax-head pipeline (this loss is currently
            only used with `output_mode="softmax"` variants).

    `forward_single` remains a pure-seg pass (no boundary) so the validation
    loop in `train_variant.py` keeps working unchanged. Boundary supervision
    is a training-only signal — eval consumes `last_aux["boundary"]` for
    diagnostics if needed but the metric path ignores it.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        lambda_boundary: float = 0.3,
        bce_weight: float = 0.3,
        edge_dice_weight: float = 0.2,
        lambda_boundary_start: float = 0.1,
        ramp_epochs: int = 50,
        num_classes: int = 4,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.lambda_boundary = float(lambda_boundary)
        self.lambda_boundary_steady = float(lambda_boundary)
        self.lambda_boundary_start = float(lambda_boundary_start)
        self.ramp_epochs = int(ramp_epochs)
        self.bce_weight = float(bce_weight)
        self.edge_dice_weight = float(edge_dice_weight)
        self.num_classes = int(num_classes)
        self.smooth = float(smooth)

    # ----- runtime λ_b scheduling (called by trainer once per epoch) -----
    def set_lambda(self, value: float) -> None:
        self.lambda_boundary = float(value)

    def lambda_at_epoch(self, epoch: int) -> float:
        """Linear warm-up: start → steady over `ramp_epochs`."""
        if self.ramp_epochs <= 0:
            return self.lambda_boundary_steady
        frac = min(max(epoch, 0) / float(self.ramp_epochs), 1.0)
        return self.lambda_boundary_start + (
            self.lambda_boundary_steady - self.lambda_boundary_start
        ) * frac

    # ----- pass-through to the wrapped base loss for the val loop -----
    def forward_single(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.base_loss.forward_single(inputs, targets)

    # ----- per-stage edge-restricted Dice on the seg path -----
    def _edge_dice_stage(
        self,
        seg_logits: torch.Tensor,
        targets_stage: torch.Tensor,
        edge: torch.Tensor,
    ) -> torch.Tensor:
        """Mean over WT/TC/ET of 1 − Dice on edge-masked region probs."""
        wt_p, tc_p, et_p = _softmax_region_probs(seg_logits)
        wt_t, tc_t, et_t = _label_region_masks(targets_stage)
        edge_b = edge.squeeze(1)  # (B, D, H, W)
        loss = 0.0
        for p, t in [(wt_p, wt_t), (tc_p, tc_t), (et_p, et_t)]:
            p_e = p * edge_b
            t_e = t * edge_b
            inter = (p_e * t_e).sum(dim=(1, 2, 3))
            denom = p_e.sum(dim=(1, 2, 3)) + t_e.sum(dim=(1, 2, 3))
            dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
            loss = loss + (1.0 - dice.mean())
        return loss / 3.0

    def forward(
        self,
        inputs_list,
        targets: torch.Tensor,
        variance: Optional[torch.Tensor] = None,
        boundary: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Base seg loss (with the variance regulariser already baked in when
        # `base_loss` is an UncertaintyAwareLoss).
        if isinstance(self.base_loss, UncertaintyAwareLoss):
            loss = self.base_loss(inputs_list, targets, variance=variance)
        else:
            loss = self.base_loss(inputs_list, targets)

        if boundary is None or self.lambda_boundary <= 0.0:
            return loss

        # Edge GT at full res, then max-pool downsampled for the two deep-sup
        # levels. Computed in float (one-hot dilation is autocast-friendly).
        edge_full = _morph_edge_gt(targets, num_classes=self.num_classes)
        edge_ds1 = _downsample_edge(edge_full, scale=2)
        edge_ds2 = _downsample_edge(edge_full, scale=4)

        # Same nearest-downsampled targets the seg deep-sup branch sees.
        t_full = targets
        t_ds1 = F.interpolate(
            targets.unsqueeze(1).float(), scale_factor=0.5, mode="nearest"
        ).squeeze(1).long()
        t_ds2 = F.interpolate(
            targets.unsqueeze(1).float(), scale_factor=0.25, mode="nearest"
        ).squeeze(1).long()

        seg_full, seg_ds1, seg_ds2 = inputs_list[0], inputs_list[1], inputs_list[2]
        b1, b2, b3 = boundary

        stages = [
            (b1, edge_full, seg_full, t_full, 1.0),
            (b2, edge_ds1, seg_ds1, t_ds1, 0.5),
            (b3, edge_ds2, seg_ds2, t_ds2, 0.25),
        ]

        bce_total = 0.0
        edge_dice_total = 0.0
        for blogit, edge_s, seg_s, t_s, w in stages:
            # BCE-with-logits handles autocast cleanly internally.
            bce = F.binary_cross_entropy_with_logits(
                blogit.float(), edge_s, reduction="mean"
            )
            ed = self._edge_dice_stage(seg_s, t_s, edge_s)
            bce_total = bce_total + w * bce
            edge_dice_total = edge_dice_total + w * ed

        boundary_term = (
            self.bce_weight * bce_total + self.edge_dice_weight * edge_dice_total
        )
        return loss + self.lambda_boundary * boundary_term
