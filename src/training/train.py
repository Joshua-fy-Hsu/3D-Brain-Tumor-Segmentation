import sys
import os
import time
import datetime
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

from model.model import ResUnet3D
from preprocessing.dataset import BratsDataset
from configs import config

# Losses live in `losses.py` as of Phase 4. Re-export them here so any caller
# that still imports `from training.train import RegionWiseDiceFocalLoss`
# (e.g. legacy scripts) keeps working unchanged.
from training.losses import (  # noqa: F401
    RegionWiseDiceFocalLoss,
    RegionWiseDiceFocalSigmoidLoss,
    regions_to_labels,
)


# ---------------------------------------------------------------------------
# Train / Validate loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, accum_steps):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    loop = tqdm(loader, desc="Training", leave=False, smoothing=0)
    for i, batch in enumerate(loop):
        data    = batch['image'].to(device, non_blocking=True)
        targets = batch['mask'].to(device, non_blocking=True).long()

        with torch.amp.autocast('cuda', dtype=torch.float16):
            outputs = model(data)
            loss    = criterion(outputs, targets) / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps
        loop.set_postfix(loss=running_loss / (i + 1))

    return running_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        loop = tqdm(loader, desc="Validation", leave=False, smoothing=0)
        for batch in loop:
            data    = batch['image'].to(device, non_blocking=True)
            targets = batch['mask'].to(device, non_blocking=True).long()

            with torch.amp.autocast('cuda', dtype=torch.float16):
                predictions = model(data)
                loss = criterion.forward_single(predictions, targets)

            running_loss += loss.item()

    return running_loss / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DEVICE = config.DEVICE
    NUM_EPOCHS = config.NUM_EPOCHS
    WARMUP_EPOCHS = config.WARMUP_EPOCHS

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    project_root = os.path.dirname(src_dir)
    log_dir = os.path.join(project_root, "logs", f"run_{timestamp}_DiceFocal")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    print(f"Training on {DEVICE} | Patch {config.PATCH_SIZE} | "
          f"Batch {config.BATCH_SIZE} | Accum {config.ACCUM_STEPS} | "
          f"Epochs {NUM_EPOCHS} | Warmup {WARMUP_EPOCHS}")

    # Data
    train_dataset = BratsDataset(phase="train")
    val_dataset   = BratsDataset(phase="val")
    train_loader  = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR, persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR, persistent_workers=False,
    )

    # Model — 5 input channels (4 modalities + foreground mask)
    model     = ResUnet3D(in_channels=config.IN_CHANNELS, num_classes=config.NUM_CLASSES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    criterion = RegionWiseDiceFocalLoss(gamma=2.0)
    scaler    = torch.amp.GradScaler('cuda')

    # Scheduler: linear warmup → cosine annealing
    warmup_scheduler  = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-4, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS]
    )

    best_val_loss = float('inf')
    csv_path = os.path.join(log_dir, "training_log.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,lr,train_loss,val_loss,time_s,best_model\n")

    for epoch in range(NUM_EPOCHS):
        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] | LR: {current_lr:.2e}")
        start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, DEVICE, config.ACCUM_STEPS
        )
        val_loss = validate_one_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()

        elapsed = time.time() - start
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")
        writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
            print(">>> Best Model Saved!")

        with open(csv_path, "a") as f:
            f.write(f"{epoch+1},{current_lr:.6e},{train_loss:.4f},{val_loss:.4f},{elapsed:.1f},{int(is_best)}\n")

    writer.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
