#!/usr/bin/env bash
# Phase 6 - `full` SOTA-push: architectural upgrades + snapshot ensemble +
# extended TTA + tuned V_min.
#
# `full` differentiates from `boundary` in two ways:
#   - Architecture: spectral_blocks_per_stage=4 (vs 2), encoder_extra_depth at
#     16^3, alpha-gated multi-scale fusion seg head. Param count ~37M
#     (vs 24M boundary).
#   - Recipe: trainer saves top-K=5 EMA snapshots (>=15 epoch gap); eval
#     averages them via EnsemblePredictor + 32-view extended TTA + per-region
#     V_min sweep + sliding-window overlap 0.625.
#
# Phase-6 v1 -> v2 corrections (v1 underfit boundary by 0.018 mean Dice with
# a -0.066 ET regression; v1 was still climbing at ep 190):
#   - decoder_dropout_final 0.10 -> 0.05 (was over-regularising d1)
#   - fusion head: now alpha-gated residual refinement of final_conv(d1),
#     init alpha=0 so the head starts identical to boundary's
#   - lambda_boundary ramp: 0.05 -> 0.25 over 100 epochs (was 0.1 -> 0.3
#     over 50). v1 crossover with boundary happened exactly when lambda_b
#     hit its peak; slower ramp lets seg loss dominate longer.
#   - epochs 200 -> 300, warmup 5 -> 10
#   - snapshot_min_gap 10 -> 15 (better diversity over the longer run)
#
# Phase-5 reference (TTA+post, mean Dice / mean HD95):
#   boundary       0.8252 / 7.21
#   spectral_swin  0.8279 / 7.26
# Phase-6 gate: mean Dice >= 0.84 AND HD95 ET/WT significant vs boundary
# (paired Wilcoxon, p<0.05).
#
# Pipeline:
#   1. Sanity (CPU+CUDA): forward + backward at 128^3; dict shape; param_groups
#      split routes enc4b/fusion_head into CNN group and the deeper Swin blocks
#      into transformer; TopKSnapshotSaver gap-constraint + ranking; ensemble
#      forward shape; extended TTA probs sum to 1.
#   2. Complexity profile for `full` (records params/FLOPs/latency).
#   3. Train `full` (full_preset -> bf16, BoundaryAwareLoss, top-K=5).
#   4. Eval `full` with ensemble + extended TTA + V_min sweep + overlap 0.625.
#   5. Paired Wilcoxon: `full` tta_post vs `boundary` tta_post (primary gate);
#      `full` tta_post vs `spectral_swin` tta_post (champion killer).
#
# Auto-launches into a tmux session named "phase6". Detach: Ctrl-b d.
# Reattach: tmux attach -t phase6. Kill: tmux kill-session -t phase6.
#
# Usage:
#   bash scripts/phase6.sh                        # full pipeline (300 ep)
#   SKIP_FULL=1 bash scripts/phase6.sh            # skip train+eval if done
#   EPOCHS=20 WARMUP=2 bash scripts/phase6.sh     # short dev cycle
#   EPOCHS=200 EXP_NAME=phase6_short bash scripts/phase6.sh
#
# Skip flags (set =1):
#   SKIP_SANITY, SKIP_COMPLEXITY, SKIP_FULL (train+eval), SKIP_COMPARE
#
# Pre-reqs on the pod:
#   - PyTorch + project requirements installed
#   - BraTS preprocessed; src/configs/config.py:TRAIN_DATA_PATH set
#   - Phase 5 already completed -> results/boundary/eval_*/per_case_metrics.csv
#   - Phase 3 already completed -> results/spectral_swin/eval_*/per_case_metrics.csv
#   - One CUDA GPU visible (bf16-capable: Ampere+ / 4090 / A100)
#   - tmux available (script installs it on Debian/Ubuntu pods if missing)

set -euo pipefail

SESSION="${SESSION:-phase6}"

# ---------------------------------------------------------------------------
# Re-exec inside tmux if not already in a session.
# ---------------------------------------------------------------------------
if [ -z "${TMUX:-}" ]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "[info] tmux not installed; attempting apt-get install"
    apt-get update -qq && apt-get install -y -qq tmux \
      || { echo "[error] could not install tmux. Install it manually and re-run."; exit 1; }
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[info] tmux session '$SESSION' already exists. Attaching."
    echo "       (kill with: tmux kill-session -t $SESSION)"
    exec tmux attach -t "$SESSION"
  fi

  SCRIPT_ABS="$(readlink -f "$0")"
  REPO_ABS="$(cd "$(dirname "$SCRIPT_ABS")/.." && pwd)"

  echo "[info] launching Phase 6 in tmux session '$SESSION'."
  echo "       detach: Ctrl-b then d.   reattach: tmux attach -t $SESSION"

  tmux new-session -d -s "$SESSION" -c "$REPO_ABS" \
       -e "EPOCHS=${EPOCHS:-300}" \
       -e "WARMUP=${WARMUP:-10}" \
       -e "EXP_NAME=${EXP_NAME:-phase6}" \
       -e "SKIP_SANITY=${SKIP_SANITY:-0}" \
       -e "SKIP_COMPLEXITY=${SKIP_COMPLEXITY:-0}" \
       -e "SKIP_FULL=${SKIP_FULL:-0}" \
       -e "SKIP_COMPARE=${SKIP_COMPARE:-0}" \
       "bash '$SCRIPT_ABS' $*; ec=\$?; echo; echo \"*** phase6.sh exited with code \$ec. Detach: Ctrl-b d. Close: Ctrl-d.\"; exec bash"

  exec tmux attach -t "$SESSION"
fi

# ---------------------------------------------------------------------------
# Inside tmux from here on.
# ---------------------------------------------------------------------------
EPOCHS="${EPOCHS:-300}"
WARMUP="${WARMUP:-10}"
EXP_NAME="${EXP_NAME:-phase6}"
SKIP_SANITY="${SKIP_SANITY:-0}"
SKIP_COMPLEXITY="${SKIP_COMPLEXITY:-0}"
SKIP_FULL="${SKIP_FULL:-0}"
SKIP_COMPARE="${SKIP_COMPARE:-0}"

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "================================================================"
echo "Phase 6 - full SOTA push"
echo "  epochs   : $EPOCHS"
echo "  warmup   : $WARMUP"
echo "  exp name : $EXP_NAME"
echo "  skip     : sanity=$SKIP_SANITY complexity=$SKIP_COMPLEXITY"
echo "             full=$SKIP_FULL compare=$SKIP_COMPARE"
echo "================================================================"

pip install -q fvcore >/dev/null 2>&1 || echo "[warn] fvcore unavailable; FLOPs will be NaN"

# ---------------------------------------------------------------------------
# 1. Sanity check.
# ---------------------------------------------------------------------------
if [ "$SKIP_SANITY" = "0" ]; then
  echo
  echo "--- [1/5] sanity: forward + backward on cuda + saver + ensemble + extended TTA ---"
  python - <<'PY'
import os, tempfile, json
import torch
from model.registry import build_variant
from model.trans_resunet import TransResUNet3D
from training.losses import (
    RegionWiseDiceFocalLoss, UncertaintyAwareLoss, BoundaryAwareLoss,
)
from training.train_variant import TopKSnapshotSaver
from evaluation._core import DictToSegAdapter
from evaluation.ensemble import EnsemblePredictor
from evaluation import uncertainty as U

assert torch.cuda.is_available(), "CUDA not available - abort."

# ---- (a) Build `full`, verify three new flags are active. ----
m = build_variant("full").cuda()
n_total_p = sum(p.numel() for p in m.parameters())
print(f"full params: {n_total_p:,}")
assert m.enc4b is not None, "encoder_extra_depth=True must give an enc4b ResidualBlock"
assert m.fusion_head is not None, "use_multiscale_fusion_head=True must give a fusion_head"
assert m.final_conv is None, "fusion_head replaces final_conv when the flag is on"
assert len(m.spectral_swin_stage.stage1) == 4, \
    f"spectral_blocks_per_stage=4 -> stage1 must have 4 blocks, got {len(m.spectral_swin_stage.stage1)}"
assert len(m.spectral_swin_stage.stage2) == 4, \
    f"spectral_blocks_per_stage=4 -> stage2 must have 4 blocks, got {len(m.spectral_swin_stage.stage2)}"
print(f"flags ok: enc4b/fusion_head live; swin blocks = "
      f"{len(m.spectral_swin_stage.stage1)} + {len(m.spectral_swin_stage.stage2)}")

# ---- (b) parameter_groups split: enc4b/fusion_head -> CNN, new Swin blocks -> transformer ----
cnn_p, tr_p = m.parameter_groups()
cnn_set = set()
tr_set = set()
for nm, p in m.named_parameters():
    if not p.requires_grad: continue
    if nm.startswith("spectral_swin_stage."):
        tr_set.add(nm)
    else:
        cnn_set.add(nm)
assert any("enc4b" in nm for nm in cnn_set), "enc4b params must be in CNN group"
assert any("fusion_head" in nm for nm in cnn_set), "fusion_head params must be in CNN group"
assert any("spectral_swin_stage.stage1.2" in nm for nm in tr_set), \
    "new Swin block stage1[2] params must be in transformer group"
assert any("spectral_swin_stage.stage1.3" in nm for nm in tr_set), \
    "new Swin block stage1[3] params must be in transformer group"
covered = sum(p.numel() for p in cnn_p) + sum(p.numel() for p in tr_p)
print(f"param_groups split: cnn={sum(p.numel() for p in cnn_p):,}  "
      f"tr={sum(p.numel() for p in tr_p):,}  covered={covered:,}/{n_total_p:,}")
assert covered == sum(p.numel() for p in m.parameters() if p.requires_grad), "leak"

# ---- (c) fp32 forward+backward, dict shape, finite grads on new modules ----
x = torch.randn(1, 5, 128, 128, 128, device="cuda")
target = torch.randint(0, 4, (1, 128, 128, 128), dtype=torch.long, device="cuda")

m.train()
out = m(x)
assert isinstance(out, dict) and sorted(out.keys()) == ["boundary", "seg", "variance"]
seg, var, bnd = out["seg"], out["variance"], out["boundary"]
assert isinstance(seg, tuple) and len(seg) == 3
assert isinstance(bnd, tuple) and len(bnd) == 3
assert tuple(seg[0].shape) == (1, 4, 128, 128, 128)
assert tuple(bnd[0].shape) == (1, 1, 128, 128, 128)
assert (var >= 0).all(), "variance must be non-negative"

seg_loss = RegionWiseDiceFocalLoss(gamma=2.0, ce_weight=0.3,
                                   class_weights=(0.1, 2.0, 1.0, 1.0))
unc = UncertaintyAwareLoss(seg_loss, lambda_unc=0.05, target_unc_at_high_dice=0.0)
crit = BoundaryAwareLoss(base_loss=unc, lambda_boundary=0.3,
                         bce_weight=0.3, edge_dice_weight=0.2,
                         lambda_boundary_start=0.1, ramp_epochs=50)
crit.set_lambda(0.3)
loss = crit(seg, target, variance=var, boundary=bnd)
loss.backward()

bad = [nm for nm, p in m.named_parameters()
       if p.grad is None or not torch.isfinite(p.grad).all()]
assert not bad, f"missing/non-finite grads on {len(bad)} params: {bad[:5]}"
print(f"fp32 backward: all {sum(1 for _ in m.parameters())} params have finite grads.")

# Check specific new modules' grads exist + non-trivial.
for nm in ("enc4b", "fusion_head", "spectral_swin_stage.stage1.2",
           "spectral_swin_stage.stage1.3", "spectral_swin_stage.stage2.2",
           "spectral_swin_stage.stage2.3"):
    sub = m
    for k in nm.split("."):
        sub = sub[int(k)] if k.isdigit() else getattr(sub, k)
    gnorms = [p.grad.abs().mean().item() for p in sub.parameters() if p.grad is not None]
    print(f"  {nm}: {len(gnorms)} grad tensors, mean|grad| avg={sum(gnorms)/max(len(gnorms),1):.2e}")

for p in m.parameters():
    p.grad = None

# ---- (d) bf16 autocast eval forward survives the new blocks ----
torch.cuda.reset_peak_memory_stats()
m.eval()
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out_e = m(x)
seg_e, var_e, bnd_e = out_e["seg"], out_e["variance"], out_e["boundary"]
assert isinstance(seg_e, torch.Tensor) and tuple(seg_e.shape) == (1, 4, 128, 128, 128)
assert isinstance(bnd_e, torch.Tensor) and tuple(bnd_e.shape) == (1, 1, 128, 128, 128)
assert torch.isfinite(seg_e).all() and torch.isfinite(var_e).all() and torch.isfinite(bnd_e).all()
print(f"bf16 eval ok: peak VRAM = {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# ---- (e) TopKSnapshotSaver: gap constraint + ranking ----
with tempfile.TemporaryDirectory() as td:
    saver = TopKSnapshotSaver(out_dir=td, k=3, min_gap=2)
    dummy = {"w": torch.zeros(2)}
    series = [(5, 0.80), (7, 0.81), (9, 0.83), (10, 0.82),
              (15, 0.85), (30, 0.84), (31, 0.86)]
    for ep, vd in series:
        saver.consider(epoch=ep, val_dice=vd, state_dict=dummy)
    files = sorted(f for f in os.listdir(td) if f.endswith(".pth"))
    assert files == ["snapshot_top1.pth", "snapshot_top2.pth", "snapshot_top3.pth"], files
    epochs = [e["epoch"] for e in saver.entries]
    for i in range(len(epochs)):
        for j in range(i + 1, len(epochs)):
            assert abs(epochs[i] - epochs[j]) >= saver.min_gap, \
                f"gap violated: {epochs[i]} vs {epochs[j]}"
    with open(os.path.join(td, "snapshots.json")) as f:
        meta = json.load(f)
    print(f"snapshot saver ok: ranks {[(e['rank'], e['epoch'], e['val_dice']) for e in meta['entries']]}")

# ---- (f) EnsemblePredictor forward shape ----
m1 = DictToSegAdapter(build_variant("full")).cuda().eval()
m2 = DictToSegAdapter(build_variant("full")).cuda().eval()
ens = EnsemblePredictor([m1, m2]).cuda().eval()
with torch.no_grad():
    y_ens = ens(x)
assert tuple(y_ens.shape) == (1, 4, 128, 128, 128)
assert torch.isfinite(y_ens).all()
print(f"ensemble forward ok: shape {tuple(y_ens.shape)}")

# ---- (g) extended TTA shape + probs ----
class _Tiny(torch.nn.Module):
    def __init__(self): super().__init__(); self.c = torch.nn.Conv3d(5, 4, 1)
    def forward(self, x): return self.c(x)
tiny = _Tiny().cuda().eval()
with torch.no_grad():
    mp, ent, var = U.tta_predict_extended(tiny, x[:, :, :64, :64, :64], roi=(64, 64, 64),
                                          overlap=0.5)
assert tuple(mp.shape) == (1, 4, 64, 64, 64)
assert (mp >= 0).all() and (mp <= 1).all()
ch_sum = mp.sum(dim=1)
assert (ch_sum - 1.0).abs().max() < 1e-4, "extended TTA probs must sum to 1 along channel"
print(f"extended TTA ok: 32 views averaged; channel sum max err = "
      f"{(ch_sum - 1.0).abs().max().item():.2e}")

print("\n=== sanity green ===")
PY
fi

# ---------------------------------------------------------------------------
# 2. Complexity profile.
# ---------------------------------------------------------------------------
if [ "$SKIP_COMPLEXITY" = "0" ]; then
  echo
  echo "--- [2/5] complexity profile ---"
  mkdir -p results
  python -m evaluation.complexity --variant full --device cuda \
      --out results/complexity.csv
fi

# ---------------------------------------------------------------------------
# 3-4. full train + eval
#     - Training preset for `full` is bf16 + BoundaryAwareLoss wrapping
#       UncertaintyAwareLoss(RegionWiseDiceFocalLoss). lambda_b ramps 0.1->0.3
#       over the first 50 epochs. Trainer also writes top-K=5 EMA snapshots.
#     - Eval uses the ensemble + extended TTA + V_min sweep + overlap 0.625.
# ---------------------------------------------------------------------------
if [ "$SKIP_FULL" = "0" ]; then
  echo
  echo "--- [3/5] train full ($EPOCHS epochs, warmup=$WARMUP, bf16 preset, top-K=5 snapshots) ---"
  python src/training/train_variant.py --variant full \
      --epochs "$EPOCHS" --warmup "$WARMUP" --exp-name "$EXP_NAME"
  echo
  echo "--- [4/5] eval full (ensemble + extended TTA + V_min sweep + overlap 0.625) ---"
  RUN_DIR=$(ls -1d logs/run_full_${EXP_NAME}_* 2>/dev/null | sort | tail -1)
  if [ -z "$RUN_DIR" ]; then
    echo "[error] could not find logs/run_full_${EXP_NAME}_* directory"
    exit 1
  fi
  echo "[info] using snapshots from $RUN_DIR"
  python src/evaluation/evaluate_variant.py --variant full \
      --ensemble-ckpts "${RUN_DIR}/snapshot_top*.pth" \
      --tta-extended --vmin-sweep --overlap 0.625 \
      --run-name "eval_${EXP_NAME}"

  FULL_DIR="results/full/eval_${EXP_NAME}"
  if [ -f "$FULL_DIR/summary.csv" ]; then
    echo
    echo "--- summary.csv (Dice + HD95 + NSD focus) ---"
    python - <<PY
import pandas as pd
df = pd.read_csv("$FULL_DIR/summary.csv")
cols = ["mode"] + [c for c in df.columns if c.lower().startswith(("dice", "hd95", "nsd"))]
print(df[cols].to_string(index=False))
PY
  fi
fi

# ---------------------------------------------------------------------------
# 5. Paired Wilcoxon: full vs boundary and full vs spectral_swin.
# ---------------------------------------------------------------------------
if [ "$SKIP_COMPARE" = "0" ]; then
  echo
  echo "--- [5/5] paired Wilcoxon (TTA + postprocess) ---"
  FD=$(ls -1d results/full/eval_*          2>/dev/null | sort | tail -1)
  BD=$(ls -1d results/boundary/eval_*      2>/dev/null | sort | tail -1)
  SD=$(ls -1d results/spectral_swin/eval_* 2>/dev/null | sort | tail -1)
  if [ -z "$FD" ] || [ -z "$BD" ]; then
    echo "[error] missing eval folder. full=$FD  boundary=$BD"
    echo "[hint]  Phase 5 must have been run before Phase 6 compare."
    exit 1
  fi
  echo "full          eval : $FD"
  echo "boundary      eval : $BD"
  echo "spectral_swin eval : $SD"
  echo
  echo ">>> full vs boundary (primary Phase-6 gate) <<<"
  python -m evaluation.stats compare \
      "$BD/per_case_metrics.csv" \
      "$FD/per_case_metrics.csv" \
      --mode tta_post --label-a boundary --label-b full
  if [ -n "$SD" ]; then
    echo
    echo ">>> full vs spectral_swin (champion killer) <<<"
    python -m evaluation.stats compare \
        "$SD/per_case_metrics.csv" \
        "$FD/per_case_metrics.csv" \
        --mode tta_post --label-a spectral_swin --label-b full
  fi
fi

echo
echo "================================================================"
echo "Phase 6 done."
echo "  full logs   : logs/run_full_${EXP_NAME}_*/"
echo "  full eval   : results/full/eval_*/"
echo "  complexity  : results/complexity.csv"
echo
echo "  Phase-6 gate: mean Dice >= 0.84 AND HD95 ET/WT significant vs boundary"
echo "  (paired Wilcoxon, p<0.05). The win comes from three architectural"
echo "  upgrades (deeper Swin / extra encoder depth / multi-scale fusion head)"
echo "  compounded by snapshot ensemble + extended TTA + tuned V_min."
echo "================================================================"
echo
echo "Detach now with Ctrl-b d, or just close the terminal - tmux keeps it."
