"""Variant-aware trainer.

Subsumes train.py (CNN) and train_transformer.py (transformer with EMA / GPU
aug / gradient clip / dice-based best-model). Picks the right knobs per
variant via TRAINING_PRESETS below.

Usage:
  python src/training/train_variant.py --variant base_cnn
  python src/training/train_variant.py --variant current_transformer
  python src/training/train_variant.py --variant cross_modal --exp-name ablation_v1
  python src/training/train_variant.py --variant base_cnn --epochs 150 --lr 5e-5

Outputs:
  logs/<variant>_<exp_name>_<timestamp>/
    best_model.pth      (state_dict only — no optimizer)
    training_log.csv
    tensorboard/        (via SummaryWriter)
"""
from __future__ import annotations

import argparse
import datetime
import gc
import os
import random
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

CURR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(CURR)
ROOT = os.path.dirname(SRC)
if SRC not in sys.path:
    sys.path.append(SRC)

from configs import config
from model.registry import build_variant, get_output_mode
from preprocessing.dataset import BratsDataset
from training.losses import (
    RegionWiseDiceFocalLoss,
    RegionWiseDiceFocalSigmoidLoss,
    UncertaintyAwareLoss,
    BoundaryAwareLoss,
)


# ---------------------------------------------------------------------------
# Per-variant training presets
# ---------------------------------------------------------------------------
@dataclass
class TrainingPreset:
    """All the knobs that differ between training recipes. Defaults match the
    transformer recipe (the more featureful one); base_cnn overrides them."""
    use_ema: bool = True
    ema_decay: float = 0.999
    use_gpu_aug: bool = True
    use_grad_clip: bool = True
    grad_clip_max_norm: float = 1.0
    amp_dtype: str = "fp16"           # "fp16" or "bf16". fp16 → use GradScaler.
    early_stop: bool = True
    persistent_workers: bool = True
    # Best-model selection: "val_loss" (lower better) or "val_dice" (higher better)
    best_metric: str = "val_dice"
    # Loss class weights (4-class softmax) — applied to per-class CE auxiliary
    class_weights: tuple = (0.1, 2.0, 1.0, 1.0)
    ce_weight: float = 0.3
    # Dataset sampling
    ncr_sample_prob: float = 0.25
    tumor_sample_prob: float = 0.5
    # Param groups: if model exposes parameter_groups(), use the split optimizer
    use_param_groups: bool = True
    # Tag suffix for log dir
    log_suffix: str = ""
    # Phase 4 — uncertainty-aware loss wrapper around the base seg loss
    use_uncertainty_loss: bool = False
    lambda_unc: float = 0.05
    target_unc_at_high_dice: float = 0.0
    # Phase 5 — boundary-aware loss wrapper (composes on top of the uncertainty
    # wrapper for the `full` / `boundary` variants).
    use_boundary_loss: bool = False
    lambda_boundary: float = 0.3
    lambda_boundary_start: float = 0.1
    lambda_boundary_ramp_epochs: int = 50
    boundary_bce_weight: float = 0.3
    boundary_edge_dice_weight: float = 0.2
    # Phase 6 — top-K snapshot saving for the snapshot-ensemble eval recipe.
    # top_k_snapshots=1 (default) preserves the legacy single-best behaviour.
    # snapshot_min_gap enforces a minimum epoch distance between saved
    # snapshots so the ensemble members aren't degenerate near-copies.
    top_k_snapshots: int = 1
    snapshot_min_gap: int = 10


def base_cnn_preset() -> TrainingPreset:
    return TrainingPreset(
        use_ema=False,
        use_gpu_aug=False,
        use_grad_clip=False,
        early_stop=False,
        persistent_workers=False,
        best_metric="val_loss",
        class_weights=None,
        ce_weight=0.0,
        ncr_sample_prob=0.0,
        tumor_sample_prob=0.5,
        use_param_groups=False,
        log_suffix="DiceFocal",
    )


def transformer_preset() -> TrainingPreset:
    return TrainingPreset(log_suffix="DiceFocalTrans")


def spectral_swin_preset() -> TrainingPreset:
    """bf16 mandatory — fp16 overflows windowed-attention softmax at 16^3."""
    return TrainingPreset(amp_dtype="bf16", log_suffix="DiceFocalSwin")


def uncertainty_preset() -> TrainingPreset:
    """Phase 4 — spectral_swin recipe + uncertainty-aware loss wrapper."""
    return TrainingPreset(
        amp_dtype="bf16",
        log_suffix="DiceFocalUnc",
        use_uncertainty_loss=True,
        lambda_unc=0.05,
        target_unc_at_high_dice=0.0,
    )


def boundary_preset() -> TrainingPreset:
    """Phase 5 — uncertainty recipe + boundary-aware loss wrapper.

    BoundaryAwareLoss wraps UncertaintyAwareLoss(RegionWiseDiceFocalLoss). λ_b
    ramps linearly 0.1 → 0.3 over the first 50 epochs to keep the seg path
    stable while the boundary heads learn (the plan's main Phase-5 risk was
    boundary loss dominating Dice early in training).
    """
    return TrainingPreset(
        amp_dtype="bf16",
        log_suffix="DiceFocalBnd",
        use_uncertainty_loss=True,
        lambda_unc=0.05,
        target_unc_at_high_dice=0.0,
        use_boundary_loss=True,
        lambda_boundary=0.3,
        lambda_boundary_start=0.1,
        lambda_boundary_ramp_epochs=50,
        boundary_bce_weight=0.3,
        boundary_edge_dice_weight=0.2,
    )


def full_preset() -> TrainingPreset:
    """Phase 6 v2 — boundary recipe tuned for the bigger fusion-head model.

    v1 (200 ep, dropout=0.10, lambda_boundary 0.1->0.3 over 50 ep) underfit
    boundary by 0.018 mean Dice with a -0.066 ET regression. v2 corrections:
      - decoder_dropout_final 0.10 -> 0.05 (registry kwargs)
      - lambda_boundary ramp 0.05 -> 0.25 over 100 epochs (was 0.1 -> 0.3
        over 50). v1's crossover with boundary happened exactly at ep 50-75
        which is when lambda_b hit 0.3; a slower lower ramp lets seg loss
        dominate longer.
      - snapshot_min_gap 10 -> 15 (better diversity over the 300-ep run)
      - alpha-gated fusion head (architecture-side, see multiscale_fusion.py)
      - epochs/warmup: phase6.sh passes 300/10 via CLI.
    """
    p = boundary_preset()
    p.log_suffix = "DiceFocalFull"
    p.top_k_snapshots = 5
    p.snapshot_min_gap = 15
    p.lambda_boundary = 0.25
    p.lambda_boundary_start = 0.05
    p.lambda_boundary_ramp_epochs = 100
    return p


# Map variant name -> preset factory. New variants register here.
TRAINING_PRESETS = {
    "base_cnn":            base_cnn_preset,
    "current_transformer": transformer_preset,
    "spectral_swin":       spectral_swin_preset,
    "uncertainty":         uncertainty_preset,
    "boundary":            boundary_preset,
    "full":                full_preset,
    # Phase 1+ variants will be added as they're implemented. Default for any
    # missing variant is the transformer preset (the more featureful one).
}


def get_preset(variant: str) -> TrainingPreset:
    return TRAINING_PRESETS.get(variant, transformer_preset)()


# ---------------------------------------------------------------------------
# Phase 6 — Top-K snapshot manager for snapshot-ensemble eval.
# ---------------------------------------------------------------------------
class TopKSnapshotSaver:
    """Maintain the K highest-val-Dice snapshots with a min-epoch-gap constraint.

    Invariants:
      - At most ``k`` snapshots are retained on disk as ``snapshot_top{1..K}.pth``.
      - Any two retained snapshots are at least ``min_gap`` epochs apart.
      - Ranks are by descending ``val_dice`` (rank 1 == best, equivalent to
        ``best_model.pth``).
      - Snapshot weights are whatever the caller passes in (EMA-weighted when
        EMA is on, matching the legacy ``best_model.pth`` semantics).

    Tie-breaking inside ±``min_gap``: a new candidate displaces the existing
    nearby snapshot only if it is strictly better. This keeps the saved set
    diverse while still tracking improving val_dice.
    """

    def __init__(self, out_dir: str, k: int, min_gap: int):
        self.out_dir = out_dir
        self.k = max(1, int(k))
        self.min_gap = max(0, int(min_gap))
        # entries: list of dicts {"val_dice", "epoch", "path"} (sorted by val_dice desc)
        self.entries: list[dict] = []

    def consider(self, epoch: int, val_dice: float, state_dict) -> bool:
        """Returns True iff a snapshot file was written/replaced."""
        if self.k <= 0:
            return False
        # Find any retained entry within ±min_gap epochs of this candidate.
        nearby = [e for e in self.entries if abs(e["epoch"] - epoch) < self.min_gap]
        if nearby:
            # Only displace if strictly better than the BEST nearby.
            best_nearby = max(e["val_dice"] for e in nearby)
            if val_dice <= best_nearby:
                return False
            # Drop all nearby entries to make room for this superior one.
            for e in nearby:
                self.entries.remove(e)
                try:
                    os.remove(e["path"])
                except OSError:
                    pass

        # If we already have K and the candidate isn't better than the worst,
        # skip. (After the nearby-displacement step above, len could be < K.)
        if len(self.entries) >= self.k:
            worst = min(e["val_dice"] for e in self.entries)
            if val_dice <= worst:
                return False
            # Drop the worst to make room.
            worst_e = min(self.entries, key=lambda e: e["val_dice"])
            self.entries.remove(worst_e)
            try:
                os.remove(worst_e["path"])
            except OSError:
                pass

        # Sort by val_dice desc and place the candidate.
        self.entries.append({"val_dice": float(val_dice), "epoch": int(epoch), "path": None})
        self.entries.sort(key=lambda e: -e["val_dice"])

        # Rewrite snapshot files in rank order. We save fresh tensors for the
        # new entry; existing entries are re-pointed to their new rank file.
        # Simplest correct approach: load existing tensors back into a temp
        # dict, rewrite all files. To avoid the load round-trip we instead
        # write the new candidate first, then rename others in-place.
        new_paths = [os.path.join(self.out_dir, f"snapshot_top{rank}.pth")
                     for rank in range(1, len(self.entries) + 1)]
        # Stage 1: free target filenames not currently in use.
        used = {e["path"] for e in self.entries if e["path"] is not None}
        for p in new_paths:
            if p not in used and os.path.exists(p):
                os.remove(p)
        # Stage 2: rename existing files into a tmp namespace to avoid clobber.
        tmp = {}
        for e in self.entries:
            if e["path"] is not None and os.path.exists(e["path"]):
                t = e["path"] + ".tmp"
                os.replace(e["path"], t)
                tmp[id(e)] = t
        # Stage 3: place files in rank order. Write the new candidate; rename
        # the staged tmp files into their new rank slot.
        for rank, e in enumerate(self.entries, start=1):
            target = os.path.join(self.out_dir, f"snapshot_top{rank}.pth")
            if id(e) in tmp:
                os.replace(tmp[id(e)], target)
            else:
                # The new candidate.
                torch.save(state_dict, target)
            e["path"] = target

        self._write_meta()
        return True

    def _write_meta(self):
        import json
        meta = {
            "k": self.k,
            "min_gap": self.min_gap,
            "entries": [
                {"rank": i + 1, "val_dice": e["val_dice"], "epoch": e["epoch"],
                 "path": os.path.basename(e["path"])}
                for i, e in enumerate(self.entries)
            ],
        }
        with open(os.path.join(self.out_dir, "snapshots.json"), "w") as f:
            json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# EMA (copy of the one in train_transformer.py — kept here for self-containment)
# ---------------------------------------------------------------------------
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.module = deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])


# ---------------------------------------------------------------------------
# Dice helpers (matches train_transformer.py val loop)
# ---------------------------------------------------------------------------
def _split_model_output(out):
    """Normalize a model forward output into (seg, variance, boundary).

    `seg` is the original tuple/tensor the seg loss expects. `variance` and
    `boundary` are auxiliary tensors (or None) emitted only when the model
    has the matching aux head enabled. Existing variants (base_cnn,
    cross_modal, frequency, spectral_swin, current_transformer) return
    tuples/tensors → variance/boundary are None and the loss path is
    unchanged.
    """
    if isinstance(out, dict):
        return out.get("seg"), out.get("variance"), out.get("boundary")
    return out, None, None


def _dice_per_region(pred_labels, target_labels, smooth=1e-5):
    out = {}
    for name, fn in [
        ("ET", lambda x: x == 3),
        ("TC", lambda x: (x == 1) | (x == 3)),
        ("WT", lambda x: x > 0),
    ]:
        p = fn(pred_labels).float()
        t = fn(target_labels).float()
        inter = (p * t).sum(dim=(1, 2, 3))
        union = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
        out[name] = ((2.0 * inter + smooth) / (union + smooth)).mean().item()
    return out


# ---------------------------------------------------------------------------
# Train / Validate loops (variant-aware)
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, criterion, device,
                    accum_steps, preset: TrainingPreset, ema=None):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    autocast_dtype = torch.float16 if preset.amp_dtype == "fp16" else torch.bfloat16
    use_scaler = (preset.amp_dtype == "fp16")

    if preset.use_gpu_aug:
        from preprocessing.gpu_augment import gpu_augment

    loop = tqdm(loader, desc="Training", leave=False, smoothing=0)
    for i, batch in enumerate(loop):
        data = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True).long()

        if preset.use_gpu_aug:
            data, targets = gpu_augment(data, targets)

        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            outputs = model(data)
            seg_out, variance, boundary = _split_model_output(outputs)
            if isinstance(criterion, BoundaryAwareLoss):
                loss = criterion(
                    seg_out, targets, variance=variance, boundary=boundary
                ) / accum_steps
            elif isinstance(criterion, UncertaintyAwareLoss):
                loss = criterion(seg_out, targets, variance=variance) / accum_steps
            else:
                loss = criterion(seg_out, targets) / accum_steps

        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accum_steps == 0:
            if use_scaler:
                if preset.use_grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   max_norm=preset.grad_clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if preset.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   max_norm=preset.grad_clip_max_norm)
                optimizer.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)

        running_loss += loss.item() * accum_steps
        loop.set_postfix(loss=running_loss / (i + 1))

    return running_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device, preset: TrainingPreset):
    """Returns (val_loss, mean_dice, per_region) — Dice computed on raw argmax."""
    model.eval()
    running_loss = 0.0
    dice_sums = {"ET": 0.0, "TC": 0.0, "WT": 0.0}
    n_batches = 0
    autocast_dtype = torch.float16 if preset.amp_dtype == "fp16" else torch.bfloat16

    with torch.no_grad():
        loop = tqdm(loader, desc="Validation", leave=False, smoothing=0)
        for batch in loop:
            data = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True).long()

            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                predictions = model(data)
                seg_pred, _var, _bnd = _split_model_output(predictions)
                # In eval mode the model returns the final-resolution tensor
                # for the auxiliary-head-on case (dict["seg"] is a tensor).
                # For the legacy tuple case it's also a tensor (model.eval()
                # path). forward_single just needs the per-resolution tensor.
                seg_for_loss = seg_pred[0] if isinstance(seg_pred, (tuple, list)) else seg_pred
                loss = criterion.forward_single(seg_for_loss, targets)

            preds = seg_for_loss
            pred_labels = preds.float().softmax(dim=1).argmax(dim=1)
            dices = _dice_per_region(pred_labels, targets)
            for k in dice_sums:
                dice_sums[k] += dices[k]

            running_loss += loss.item()
            n_batches += 1

    if n_batches == 0:
        return float("nan"), float("nan"), {k: float("nan") for k in dice_sums}
    mean_dice = sum(dice_sums.values()) / (3 * n_batches)
    per_region = {k: v / n_batches for k, v in dice_sums.items()}
    return running_loss / n_batches, mean_dice, per_region


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    help="Variant name from src/model/registry.py")
    ap.add_argument("--exp-name", default="",
                    help="Tag appended to the log directory name")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override config.NUM_EPOCHS")
    ap.add_argument("--warmup", type=int, default=None,
                    help="Override config.WARMUP_EPOCHS")
    ap.add_argument("--lr", type=float, default=None,
                    help="Override config.LR")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override config.BATCH_SIZE")
    ap.add_argument("--accum-steps", type=int, default=None,
                    help="Override config.ACCUM_STEPS")
    ap.add_argument("--seed", type=int, default=None,
                    help="Override config.SEED. Defaults to config.SEED.")
    # Preset overrides — useful for quick experiments
    ap.add_argument("--preset", choices=["auto", "base_cnn", "transformer"], default="auto",
                    help="Force a specific preset regardless of variant. "
                         "'auto' (default) looks up TRAINING_PRESETS by variant name.")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--no-gpu-aug", action="store_true")
    ap.add_argument("--no-grad-clip", action="store_true")
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--amp-dtype", choices=["fp16", "bf16"], default=None)
    ap.add_argument("--top-k", type=int, default=None,
                    help="Override preset.top_k_snapshots. Trainer saves the K "
                         "highest-val-Dice EMA-weighted snapshots (with min-gap "
                         "spacing) as snapshot_top{1..K}.pth for ensemble eval.")
    return ap.parse_args()


def main():
    args = parse_args()

    variant = args.variant
    if args.preset == "base_cnn":
        preset = base_cnn_preset()
    elif args.preset == "transformer":
        preset = transformer_preset()
    else:
        preset = get_preset(variant)

    # CLI overrides on the preset
    if args.no_ema:        preset.use_ema = False
    if args.no_gpu_aug:    preset.use_gpu_aug = False
    if args.no_grad_clip:  preset.use_grad_clip = False
    if args.no_early_stop: preset.early_stop = False
    if args.amp_dtype:     preset.amp_dtype = args.amp_dtype
    if args.top_k is not None:
        preset.top_k_snapshots = max(1, int(args.top_k))

    # Hyperparams (CLI > config)
    DEVICE = config.DEVICE
    NUM_EPOCHS = args.epochs or config.NUM_EPOCHS
    WARMUP_EPOCHS = args.warmup if args.warmup is not None else config.WARMUP_EPOCHS
    LR = args.lr or config.LR
    BATCH_SIZE = args.batch_size or config.BATCH_SIZE
    ACCUM_STEPS = args.accum_steps or config.ACCUM_STEPS
    SEED = args.seed if args.seed is not None else config.SEED

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name_parts = [variant]
    if args.exp_name:
        name_parts.append(args.exp_name)
    if preset.log_suffix:
        name_parts.append(preset.log_suffix)
    name_parts.append(timestamp)
    log_dir = os.path.join(ROOT, "logs", "run_" + "_".join(name_parts))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    print(f"=== train_variant ===")
    print(f"variant={variant}  exp_name={args.exp_name}  log_dir={log_dir}")
    print(f"epochs={NUM_EPOCHS}  warmup={WARMUP_EPOCHS}  lr={LR}  "
          f"batch={BATCH_SIZE}  accum={ACCUM_STEPS}")
    print(f"preset: ema={preset.use_ema} gpu_aug={preset.use_gpu_aug} "
          f"grad_clip={preset.use_grad_clip} amp={preset.amp_dtype} "
          f"best_metric={preset.best_metric} early_stop={preset.early_stop}")

    # ---- Data ----
    train_dataset = BratsDataset(
        phase="train",
        gpu_aug=preset.use_gpu_aug,
        ncr_sample_prob=preset.ncr_sample_prob,
        tumor_sample_prob=preset.tumor_sample_prob,
    )
    val_dataset = BratsDataset(phase="val", gpu_aug=preset.use_gpu_aug)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR,
        persistent_workers=preset.persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR,
        persistent_workers=preset.persistent_workers,
    )

    # ---- Model ----
    model = build_variant(variant).to(DEVICE)
    output_mode = get_output_mode(variant)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}  output_mode={output_mode}")

    # ---- Optimizer (single vs split groups) ----
    # parameter_groups() may exist on the model but return an empty
    # transformer list (e.g. cross_modal / frequency don't have a transformer
    # stage). Skip the split in that case — AdamW dislikes empty param groups.
    use_split = preset.use_param_groups and hasattr(model, "parameter_groups")
    if use_split:
        cnn_params, transformer_params = model.parameter_groups()
        use_split = len(transformer_params) > 0

    if use_split:
        optimizer = optim.AdamW(
            [
                {"params": cnn_params, "lr": LR},
                {"params": transformer_params, "lr": LR},
            ],
            lr=LR, weight_decay=config.WEIGHT_DECAY,
        )
        print(f"[opt] split groups: cnn={sum(p.numel() for p in cnn_params):,}  "
              f"transformer={sum(p.numel() for p in transformer_params):,}")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=LR,
                                weight_decay=config.WEIGHT_DECAY)
        print(f"[opt] single group")

    # ---- Loss ----
    if output_mode == "softmax":
        seg_criterion = RegionWiseDiceFocalLoss(
            gamma=2.0,
            ce_weight=preset.ce_weight,
            class_weights=preset.class_weights,
        )
    else:
        seg_criterion = RegionWiseDiceFocalSigmoidLoss(gamma=2.0)

    if preset.use_uncertainty_loss:
        unc_criterion = UncertaintyAwareLoss(
            seg_loss=seg_criterion,
            lambda_unc=preset.lambda_unc,
            target_unc_at_high_dice=preset.target_unc_at_high_dice,
        )
        print(f"[loss] UncertaintyAwareLoss(lambda_unc={preset.lambda_unc}, "
              f"target={preset.target_unc_at_high_dice})")
    else:
        unc_criterion = seg_criterion

    if preset.use_boundary_loss:
        criterion = BoundaryAwareLoss(
            base_loss=unc_criterion,
            lambda_boundary=preset.lambda_boundary,
            bce_weight=preset.boundary_bce_weight,
            edge_dice_weight=preset.boundary_edge_dice_weight,
            lambda_boundary_start=preset.lambda_boundary_start,
            ramp_epochs=preset.lambda_boundary_ramp_epochs,
        )
        # Start the schedule at epoch 0's value; the per-epoch ramp updates it.
        criterion.set_lambda(criterion.lambda_at_epoch(0))
        print(f"[loss] BoundaryAwareLoss(lambda_boundary={preset.lambda_boundary} "
              f"start={preset.lambda_boundary_start} ramp={preset.lambda_boundary_ramp_epochs}ep "
              f"bce_w={preset.boundary_bce_weight} edge_dice_w={preset.boundary_edge_dice_weight})")
    else:
        criterion = unc_criterion

    scaler = torch.amp.GradScaler("cuda") if preset.amp_dtype == "fp16" else None
    ema = ModelEMA(model, decay=preset.ema_decay) if preset.use_ema else None

    # ---- Scheduler (linear warmup -> cosine annealing) ----
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-4, end_factor=1.0, total_iters=max(WARMUP_EPOCHS, 1)
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(NUM_EPOCHS - WARMUP_EPOCHS, 1), eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[max(WARMUP_EPOCHS, 1)],
    )

    # ---- Training loop ----
    if preset.best_metric == "val_dice":
        best_score = -1.0
    else:
        best_score = float("inf")
    epochs_no_improve = 0

    # Phase 6 — top-K snapshot saver. Only active when preset/CLI set top_k > 1
    # AND best_metric == "val_dice" (saver ranks by val_dice). For val_loss
    # criteria the saver stays disabled (snapshot-ensemble eval logic assumes
    # val_dice ranking).
    snap_saver = None
    if preset.top_k_snapshots > 1 and preset.best_metric == "val_dice":
        snap_saver = TopKSnapshotSaver(
            out_dir=log_dir,
            k=preset.top_k_snapshots,
            min_gap=preset.snapshot_min_gap,
        )
        print(f"[snapshots] top-K saver active: k={snap_saver.k}  min_gap={snap_saver.min_gap}")

    csv_path = os.path.join(log_dir, "training_log.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,lr,train_loss,val_loss,val_dice_mean,"
                "val_dice_ET,val_dice_TC,val_dice_WT,time_s,best_model\n")

    for epoch in range(NUM_EPOCHS):
        current_lr = scheduler.get_last_lr()[0]
        # Phase 5: linear warm-up of λ_b across the first `ramp_epochs` epochs.
        if isinstance(criterion, BoundaryAwareLoss):
            new_lambda = criterion.lambda_at_epoch(epoch)
            criterion.set_lambda(new_lambda)
            print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] | LR: {current_lr:.2e} | "
                  f"λ_b: {new_lambda:.3f}")
        else:
            print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] | LR: {current_lr:.2e}")
        start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, DEVICE,
            ACCUM_STEPS, preset, ema=ema,
        )
        val_model = ema.module if ema is not None else model
        val_loss, val_dice, per_region = validate_one_epoch(
            val_model, val_loader, criterion, DEVICE, preset,
        )
        scheduler.step()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if epoch == 0 and torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"[VRAM] peak after epoch 1: {peak_gb:.2f} GB")

        elapsed = time.time() - start
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Dice {val_dice:.4f} (ET {per_region['ET']:.3f} / "
              f"TC {per_region['TC']:.3f} / WT {per_region['WT']:.3f}) | "
              f"Time: {elapsed:.1f}s")
        writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch)
        writer.add_scalars("Dice", {"mean": val_dice, **per_region}, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        if preset.best_metric == "val_dice":
            is_best = val_dice > best_score
            if is_best:
                best_score = val_dice
        else:
            is_best = val_loss < best_score
            if is_best:
                best_score = val_loss
        if is_best:
            epochs_no_improve = 0
            save_state = (ema.module if ema is not None else model).state_dict()
            torch.save(save_state, os.path.join(log_dir, "best_model.pth"))
            print(">>> Best Model Saved!" + (" (EMA)" if ema is not None else ""))
        else:
            epochs_no_improve += 1

        # Phase 6 — consider this epoch for the top-K snapshot pool. Independent
        # of the is_best gate above: a snapshot can be added even when it's not
        # the all-time best, as long as it improves the worst retained snapshot.
        if snap_saver is not None:
            save_state = (ema.module if ema is not None else model).state_dict()
            if snap_saver.consider(epoch=epoch + 1, val_dice=val_dice,
                                   state_dict=save_state):
                ranks = ", ".join(f"#{i+1}@ep{e['epoch']}({e['val_dice']:.4f})"
                                  for i, e in enumerate(snap_saver.entries))
                print(f">>> Snapshot pool updated: {ranks}")

        with open(csv_path, "a") as f:
            f.write(f"{epoch+1},{current_lr:.6e},{train_loss:.4f},{val_loss:.4f},"
                    f"{val_dice:.4f},{per_region['ET']:.4f},{per_region['TC']:.4f},"
                    f"{per_region['WT']:.4f},{elapsed:.1f},{int(is_best)}\n")

        if preset.early_stop and \
           (epoch + 1) >= config.EARLY_STOP_MIN_EPOCH and \
           epochs_no_improve >= config.EARLY_STOP_PATIENCE:
            print(f"\n>>> Early stop: no {preset.best_metric} improvement for "
                  f"{config.EARLY_STOP_PATIENCE} epochs. Stopping at epoch {epoch+1}.")
            break

    writer.close()
    metric_label = "Dice" if preset.best_metric == "val_dice" else "Loss"
    print(f"\nTraining complete. Best val {metric_label}: {best_score:.4f}")
    print(f"Outputs: {log_dir}")


if __name__ == "__main__":
    main()
