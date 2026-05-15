import sys
import os
import gc
import random
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from copy import deepcopy

from model.model_transformer import ResUnet3DTransformer
from preprocessing.dataset import BratsDataset
from preprocessing.gpu_augment import gpu_augment
from configs import config
from training.train import RegionWiseDiceFocalLoss


# ---------------------------------------------------------------------------
# Exponential moving average of weights — validate against EMA copy, save it.
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
# Train / Validate loops — fp16 with GradScaler. Attention is softmax-stable
# (max-subtracted) inside the model so fp16 doesn't overflow.
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, accum_steps, ema=None):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()


    loop = tqdm(loader, desc="Training", leave=False, smoothing=0)
    for i, batch in enumerate(loop):
        data    = batch['image'].to(device, non_blocking=True)
        targets = batch['mask'].to(device, non_blocking=True).long()

        data, targets = gpu_augment(data, targets)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            outputs = model(data)
            loss    = criterion(outputs, targets) / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)

        running_loss += loss.item() * accum_steps
        loop.set_postfix(loss=running_loss / (i + 1))

    return running_loss / len(loader)


def _dice_per_region(pred_labels, target_labels, smooth=1e-5):
    """Returns dict {ET, TC, WT} of mean Dice across the batch."""
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


def validate_one_epoch(model, loader, criterion, device):
    """Returns (val_loss, mean_dice, per_region) — Dice computed on raw argmax
    labels (no patch-level post-processing) for honest model selection."""
    model.eval()
    running_loss = 0.0
    dice_sums = {"ET": 0.0, "TC": 0.0, "WT": 0.0}
    n_batches = 0

    with torch.no_grad():
        loop = tqdm(loader, desc="Validation", leave=False, smoothing=0)
        for batch in loop:
            data    = batch['image'].to(device, non_blocking=True)
            targets = batch['mask'].to(device, non_blocking=True).long()

            with torch.amp.autocast('cuda', dtype=torch.float16):
                predictions = model(data)
                loss = criterion.forward_single(predictions, targets)

            pred_labels = predictions.float().softmax(dim=1).argmax(dim=1)
            dices = _dice_per_region(pred_labels, targets)
            for k in dice_sums:
                dice_sums[k] += dices[k]

            running_loss += loss.item()
            n_batches += 1

    mean_dice = sum(dice_sums.values()) / (3 * n_batches)
    per_region = {k: v / n_batches for k, v in dice_sums.items()}
    return running_loss / n_batches, mean_dice, per_region


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DEVICE = config.DEVICE
    NUM_EPOCHS = config.NUM_EPOCHS
    WARMUP_EPOCHS = config.WARMUP_EPOCHS

    # Reproducibility: wire config.SEED into all RNGs used by the pipeline.
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)

    # Throughput knobs — all safe, no Dice impact.
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")  # TF32 on Ampere+

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    project_root = os.path.dirname(src_dir)
    log_dir = os.path.join(project_root, "logs", f"run_{timestamp}_DiceFocalTrans")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    print(f"Training on {DEVICE} | Patch {config.PATCH_SIZE} | "
          f"Batch {config.BATCH_SIZE} | Accum {config.ACCUM_STEPS} | "
          f"Epochs {NUM_EPOCHS} | Warmup {WARMUP_EPOCHS} | AMP=bf16 | "
          f"Model=ResUnet3DTransformer")

    # gpu_aug=True: zoom+blur skipped on CPU because gpu_augment runs them on GPU.
    # ncr_sample_prob=0.25: 25% NCR-centered crops, 25% generic tumor-centered,
    # 50% random — counters NCR's voxel-frequency under-representation, the
    # data-side half of the NCR fix (loss-side half is the weighted CE below).
    train_dataset = BratsDataset(
        phase="train", gpu_aug=True,
        ncr_sample_prob=0.25, tumor_sample_prob=0.5,
    )
    val_dataset   = BratsDataset(phase="val", gpu_aug=True)
    train_loader  = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR, persistent_workers=True,
    )

    model = ResUnet3DTransformer(
        in_channels=config.IN_CHANNELS, num_classes=config.NUM_CLASSES,
        windowed_depth=4,
        output_mode="softmax",
        drop_path_max=config.DROP_PATH_MAX,
        attn_drop=config.ATTN_DROP,
        proj_drop=config.PROJ_DROP,
        mlp_drop=config.MLP_DROP,
        token_drop=config.TOKEN_DROP,
        decoder_dropout_inner=0.0,
        decoder_dropout_final=0.05,
    ).to(DEVICE)

    # Both CNN and transformer params train at config.LR. The previous 10x decay
    # on transformer params undertrained attention since neither sub-network is
    # pretrained. Two groups are kept for future per-group weight-decay tuning.
    cnn_params, transformer_params = model.parameter_groups()
    optimizer = optim.AdamW([
        {"params": cnn_params,         "lr": config.LR},
        {"params": transformer_params, "lr": config.LR},
    ], lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    print(f"[opt] cnn params: {sum(p.numel() for p in cnn_params):,}  "
          f"transformer params: {sum(p.numel() for p in transformer_params):,}")

    # ce_weight + class_weights: per-class CE auxiliary that re-introduces the
    # NCR-specific gradient the region loss collapses out. NCR weight 2.0
    # counters the rare-class deficit; bg weight 0.1 keeps easy background
    # voxels from drowning the term.
    criterion = RegionWiseDiceFocalLoss(
        gamma=2.0,
        ce_weight=0.3,
        class_weights=(0.1, 2.0, 1.0, 1.0),
    )
    scaler    = torch.amp.GradScaler('cuda')
    ema = ModelEMA(model, decay=0.999)
    # torch.compile skipped on Windows — Triton/Inductor backend is not
    # installed by default on Windows wheels. Eager mode still gets the
    # cudnn.benchmark + TF32 + persistent_workers wins.

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-4, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS]
    )

    best_val_dice = -1.0
    epochs_no_improve = 0
    csv_path = os.path.join(log_dir, "training_log.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,lr,train_loss,val_loss,val_dice_mean,val_dice_ET,val_dice_TC,val_dice_WT,time_s,best_model\n")

    for epoch in range(NUM_EPOCHS):
        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] | LR: {current_lr:.2e}")
        start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, DEVICE, config.ACCUM_STEPS, ema=ema
        )
        val_loss, val_dice, per_region = validate_one_epoch(
            ema.module, val_loader, criterion, DEVICE
        )
        scheduler.step()

        # End-of-epoch cleanup — keeps per-epoch time stable on long runs.
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

        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            epochs_no_improve = 0
            torch.save(ema.module.state_dict(), os.path.join(log_dir, "best_model.pth"))
            print(">>> Best Model Saved (EMA)!")
        else:
            epochs_no_improve += 1

        with open(csv_path, "a") as f:
            f.write(f"{epoch+1},{current_lr:.6e},{train_loss:.4f},{val_loss:.4f},"
                    f"{val_dice:.4f},{per_region['ET']:.4f},{per_region['TC']:.4f},"
                    f"{per_region['WT']:.4f},{elapsed:.1f},{int(is_best)}\n")

        if (epoch + 1) >= config.EARLY_STOP_MIN_EPOCH and \
           epochs_no_improve >= config.EARLY_STOP_PATIENCE:
            print(f"\n>>> Early stop: no val-Dice improvement for "
                  f"{config.EARLY_STOP_PATIENCE} epochs. Stopping at epoch {epoch+1}.")
            break

    writer.close()
    print(f"\nTraining complete. Best val Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
