#!/usr/bin/env bash
# Phase 5 — boundary-aware decoder vs Phase-4 uncertainty baseline.
#
# Pipeline:
#   1. Sanity: cuda forward+backward at 128^3 under bf16 autocast.
#      (a) fp32 forward+backward — dict has seg/variance/boundary keys; the
#          3-tuple boundary output has the expected (128^3 / 64^3 / 32^3)
#          shapes; BoundaryAwareLoss composes on top of
#          UncertaintyAwareLoss(RegionWiseDiceFocalLoss); every boundary-head
#          parameter gets a finite grad; Phase-3 spectral-swin α gates and
#          Phase-4 uncertainty α gate still see finite grads.
#      (b) λ_b ramp endpoints check: lambda_at_epoch(0) == 0.1 and
#          lambda_at_epoch(50) == 0.3 (steady).
#      (c) bf16 autocast eval forward — dict path survives autocast, eval-mode
#          boundary collapses to a single full-res tensor.
#   2. Complexity profile for `boundary` and `uncertainty` (boundary heads
#      should add <0.5M params — three tiny Conv stacks).
#   3. Train boundary (boundary preset → bf16, BoundaryAwareLoss wrapping
#      UncertaintyAwareLoss(RegionWiseDiceFocalLoss) with λ_b ramp 0.1→0.3).
#   4. Eval boundary (auto-picks bf16 because arch_family=transformer).
#      Watch HD95 / NSD columns in summary.csv — Phase-5 headline metrics.
#   5. Paired Wilcoxon: boundary vs uncertainty at TTA+post. Inspect hd95_*
#      and nsd_* p-values + TC (the region most sensitive to boundary cues —
#      Phase-4 had a small TC regression we hope to recover).
#
# Phase-4 reference numbers (TTA+post):
#   uncertainty mean Dice 0.8202  (small TC regression vs spectral_swin)
#   AURC TTA  0.2023 → 0.1950  | Spearman(unc,err) baseline 0.0742 → 0.0774
# Phase-5 success criterion: HD95 / NSD IMPROVE over uncertainty. Dice may
# move sideways; the win is sharper boundaries, and ideally a TC recovery.
#
# Auto-launches into a tmux session named "phase5". Detach with Ctrl-b d.
# Reattach with `tmux attach -t phase5`. Kill with `tmux kill-session -t phase5`.
#
# Usage:
#   bash scripts/phase5.sh                       # full pipeline
#   SKIP_BOUNDARY=1 bash scripts/phase5.sh       # skip train+eval if done
#   EPOCHS=50       bash scripts/phase5.sh       # quick A/B
#
# Skip flags (set =1 to skip):
#   SKIP_SANITY, SKIP_COMPLEXITY, SKIP_BOUNDARY (train+eval), SKIP_COMPARE
#
# Pre-reqs on the pod:
#   - PyTorch + project requirements installed
#   - BraTS preprocessed; src/configs/config.py:TRAIN_DATA_PATH set
#   - Phase 4 already completed → results/uncertainty/eval_*/per_case_metrics.csv
#   - One CUDA GPU visible (bf16-capable: Ampere+ / 4090 / A100)
#   - tmux available (script installs it on Debian/Ubuntu pods if missing)

set -euo pipefail

SESSION="${SESSION:-phase5}"

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

  echo "[info] launching Phase 5 in tmux session '$SESSION'."
  echo "       detach: Ctrl-b then d.   reattach: tmux attach -t $SESSION"

  tmux new-session -d -s "$SESSION" -c "$REPO_ABS" \
       -e "EPOCHS=${EPOCHS:-200}" \
       -e "EXP_NAME=${EXP_NAME:-phase5}" \
       -e "SKIP_SANITY=${SKIP_SANITY:-0}" \
       -e "SKIP_COMPLEXITY=${SKIP_COMPLEXITY:-0}" \
       -e "SKIP_BOUNDARY=${SKIP_BOUNDARY:-0}" \
       -e "SKIP_COMPARE=${SKIP_COMPARE:-0}" \
       "bash '$SCRIPT_ABS' $*; ec=\$?; echo; echo \"*** phase5.sh exited with code \$ec. Detach: Ctrl-b d. Close: Ctrl-d.\"; exec bash"

  exec tmux attach -t "$SESSION"
fi

# ---------------------------------------------------------------------------
# Inside tmux from here on.
# ---------------------------------------------------------------------------
EPOCHS="${EPOCHS:-200}"
EXP_NAME="${EXP_NAME:-phase5}"
SKIP_SANITY="${SKIP_SANITY:-0}"
SKIP_COMPLEXITY="${SKIP_COMPLEXITY:-0}"
SKIP_BOUNDARY="${SKIP_BOUNDARY:-0}"
SKIP_COMPARE="${SKIP_COMPARE:-0}"

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "================================================================"
echo "Phase 5 — boundary vs uncertainty"
echo "  epochs   : $EPOCHS"
echo "  exp name : $EXP_NAME"
echo "  skip     : sanity=$SKIP_SANITY complexity=$SKIP_COMPLEXITY"
echo "             boundary=$SKIP_BOUNDARY compare=$SKIP_COMPARE"
echo "================================================================"

pip install -q fvcore >/dev/null 2>&1 || echo "[warn] fvcore unavailable; FLOPs will be NaN"

# ---------------------------------------------------------------------------
# 1. Sanity check.
# ---------------------------------------------------------------------------
if [ "$SKIP_SANITY" = "0" ]; then
  echo
  echo "--- [1/5] sanity: forward + backward on cuda ---"
  python - <<'PY'
import torch
from model.registry import build_variant
from training.losses import (
    RegionWiseDiceFocalLoss, UncertaintyAwareLoss, BoundaryAwareLoss,
)
from training.train_variant import _split_model_output

assert torch.cuda.is_available(), "CUDA not available — abort."
m = build_variant("boundary").cuda()
n_total_p = sum(p.numel() for p in m.parameters())
print(f"params: {n_total_p:,}")

cnn_p, tr_p = m.parameter_groups()
print(f"  cnn group:         {sum(p.numel() for p in cnn_p):,}")
print(f"  transformer group: {sum(p.numel() for p in tr_p):,}")

# Boundary heads must live in the cnn group (they're plain Conv3d stacks).
bh_params = []
for hname in ("boundary_head_1", "boundary_head_2", "boundary_head_3"):
    head = getattr(m, hname)
    assert head is not None, f"{hname} should exist when use_boundary=True"
    bh_params.extend(list(head.parameters()))
n_bh = sum(p.numel() for p in bh_params)
print(f"  boundary heads:    {n_bh:,} params total (3 heads)")

# Sanity: alpha gates from Phase 3 and 4 still init to 0.
ub = m.uncertainty_block
assert float(ub.alpha) == 0.0, f"uncertainty alpha must init to 0, got {float(ub.alpha)}"
for stage_name in ("stage1", "stage2"):
    blks = getattr(m.spectral_swin_stage, stage_name)
    for i, blk in enumerate(blks):
        assert float(blk.alpha) == 0.0, \
            f"spectral_swin {stage_name}[{i}].alpha must init to 0"
print("Phase-3 + Phase-4 α gates init to 0.0 (boundary heads carry the invariant)")

x = torch.randn(1, 5, 128, 128, 128, device="cuda")
target = torch.randint(0, 4, (1, 128, 128, 128), dtype=torch.long, device="cuda")

# (a) fp32 forward+backward — dict-shaped output with boundary tuple.
m.train()
out = m(x)
assert isinstance(out, dict), f"expected dict output for boundary, got {type(out)}"
assert sorted(out.keys()) == ["boundary", "seg", "variance"], \
    f"unexpected dict keys {list(out.keys())}"
seg = out["seg"]
var = out["variance"]
bnd = out["boundary"]
assert isinstance(seg, tuple) and len(seg) == 3, "train: seg must be (final, ds1, ds2)"
assert isinstance(bnd, tuple) and len(bnd) == 3, "train: boundary must be (b1, b2, b3)"
print("train seg shapes:", [tuple(t.shape) for t in seg])
print("train bnd shapes:", [tuple(t.shape) for t in bnd])
print("variance shape:", tuple(var.shape))
assert tuple(bnd[0].shape) == (1, 1, 128, 128, 128), f"b1 wrong shape: {tuple(bnd[0].shape)}"
assert tuple(bnd[1].shape) == (1, 1,  64,  64,  64), f"b2 wrong shape: {tuple(bnd[1].shape)}"
assert tuple(bnd[2].shape) == (1, 1,  32,  32,  32), f"b3 wrong shape: {tuple(bnd[2].shape)}"
assert (var >= 0).all(), "variance must be non-negative (Softplus head)"

# BoundaryAwareLoss wrapping UncertaintyAwareLoss wrapping RegionWiseDiceFocalLoss.
seg_loss = RegionWiseDiceFocalLoss(gamma=2.0, ce_weight=0.3,
                                   class_weights=(0.1, 2.0, 1.0, 1.0))
unc = UncertaintyAwareLoss(seg_loss, lambda_unc=0.05, target_unc_at_high_dice=0.0)
crit = BoundaryAwareLoss(
    base_loss=unc,
    lambda_boundary=0.3,
    bce_weight=0.3,
    edge_dice_weight=0.2,
    lambda_boundary_start=0.1,
    ramp_epochs=50,
)
# (b) λ_b ramp endpoints.
assert abs(crit.lambda_at_epoch(0) - 0.1) < 1e-6, \
    f"lambda_at_epoch(0) should be 0.1, got {crit.lambda_at_epoch(0)}"
assert abs(crit.lambda_at_epoch(50) - 0.3) < 1e-6, \
    f"lambda_at_epoch(50) should be 0.3, got {crit.lambda_at_epoch(50)}"
assert abs(crit.lambda_at_epoch(25) - 0.2) < 1e-6, \
    f"lambda_at_epoch(25) should be 0.2, got {crit.lambda_at_epoch(25)}"
assert abs(crit.lambda_at_epoch(999) - 0.3) < 1e-6, "ramp should clamp at steady value"
print(f"λ_b ramp endpoints OK: ep0={crit.lambda_at_epoch(0):.3f}  "
      f"ep25={crit.lambda_at_epoch(25):.3f}  ep50={crit.lambda_at_epoch(50):.3f}  "
      f"ep999={crit.lambda_at_epoch(999):.3f}")

# Run loss at the steady value so the boundary term actually fires.
crit.set_lambda(0.3)
loss = crit(seg, target, variance=var, boundary=bnd)
loss_seg_only = seg_loss(seg, target)
loss_seg_unc  = unc(seg, target, variance=var)
print(f"BoundaryAwareLoss: {loss.item():.4f}  "
      f"(seg-only: {loss_seg_only.item():.4f},  seg+unc: {loss_seg_unc.item():.4f})")
assert loss.item() > loss_seg_unc.item() - 1e-4, \
    "Boundary term should add to (not subtract from) the seg+unc loss"
loss.backward()

n_total = sum(1 for _ in m.parameters())
n_grad  = sum(1 for p in m.parameters() if p.grad is not None)
n_fin   = sum(1 for p in m.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
assert n_grad == n_total, f"missing grads on {n_total - n_grad} tensors"
assert n_fin  == n_total, f"non-finite grads on {n_total - n_fin} tensors"

# Boundary heads must all have finite grads.
for hname in ("boundary_head_1", "boundary_head_2", "boundary_head_3"):
    head = getattr(m, hname)
    for pn, p in head.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), \
            f"{hname}.{pn} grad missing/NaN"
    # Mean abs grad sanity (should be non-trivial because the BCE term fires).
    gnorm = sum(p.grad.abs().mean().item() for p in head.parameters()) / sum(
        1 for _ in head.parameters())
    print(f"{hname}: all params finite, mean|grad|≈{gnorm:.2e}")

# uncertainty α + var_head invariants carried.
ag = ub.alpha.grad
assert ag is not None and torch.isfinite(ag).all(), "uncertainty alpha grad missing/NaN"
for pn, p in ub.var_head.named_parameters():
    assert p.grad is not None and torch.isfinite(p.grad).all(), \
        f"var_head.{pn} grad missing/NaN"

# spectral-swin α gates still get finite grads.
for stage_name in ("stage1", "stage2"):
    blks = getattr(m.spectral_swin_stage, stage_name)
    for i, blk in enumerate(blks):
        ssg = blk.alpha.grad
        assert ssg is not None and torch.isfinite(ssg).all(), \
            f"{stage_name}[{i}].alpha grad missing/NaN"
print(f"fp32 backward: grads finite on {n_fin}/{n_total} tensors. "
      f"Boundary heads + Phase-3/4 α gates all green.")

for p in m.parameters():
    p.grad = None

# (c) bf16 autocast eval forward — verifies dict path / boundary survives autocast,
#     eval-mode boundary collapses to a single full-res tensor.
torch.cuda.reset_peak_memory_stats()
m.eval()
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out_e = m(x)
assert isinstance(out_e, dict)
seg_e = out_e["seg"]
var_e = out_e["variance"]
bnd_e = out_e["boundary"]
assert isinstance(seg_e, torch.Tensor), "eval: seg must be a tensor"
assert isinstance(bnd_e, torch.Tensor), "eval: boundary must be a single tensor (b1 only)"
assert tuple(bnd_e.shape) == (1, 1, 128, 128, 128), \
    f"eval boundary shape wrong: {tuple(bnd_e.shape)}"
assert torch.isfinite(seg_e).all(), "bf16 autocast eval produced non-finite seg"
assert torch.isfinite(var_e).all(), "bf16 autocast eval produced non-finite variance"
assert torch.isfinite(bnd_e).all(), "bf16 autocast eval produced non-finite boundary"
print(f"bf16 autocast eval OK: seg.dtype={seg_e.dtype}, var.dtype={var_e.dtype}, "
      f"bnd.dtype={bnd_e.dtype}, peak VRAM={torch.cuda.max_memory_allocated()/1e9:.2f} GB")
PY
fi

# ---------------------------------------------------------------------------
# 2. Complexity profile.
# ---------------------------------------------------------------------------
if [ "$SKIP_COMPLEXITY" = "0" ]; then
  echo
  echo "--- [2/5] complexity profile ---"
  mkdir -p results
  python -m evaluation.complexity --variant boundary    --device cuda \
      --out results/complexity.csv
  python -m evaluation.complexity --variant uncertainty --device cuda \
      --out results/complexity.csv
fi

# ---------------------------------------------------------------------------
# 3-4. boundary train + eval
#     - Training preset for `boundary` is bf16 + BoundaryAwareLoss wrapping
#       UncertaintyAwareLoss(RegionWiseDiceFocalLoss). λ_b ramps 0.1→0.3 over
#       the first 50 epochs (printed at the head of each epoch).
#     - Eval uses bf16 default because arch_family=transformer.
#     - The dict output is unwrapped at the eval boundary (DictToSegAdapter
#       in evaluation/_core.py). Boundary is a training-only signal — metric
#       pipeline ignores it; HD95/NSD are the targets to watch.
# ---------------------------------------------------------------------------
if [ "$SKIP_BOUNDARY" = "0" ]; then
  echo
  echo "--- [3/5] train boundary ($EPOCHS epochs, bf16 preset, λ_b 0.1→0.3 over 50ep) ---"
  python src/training/train_variant.py --variant boundary \
      --epochs "$EPOCHS" --exp-name "$EXP_NAME"
  echo
  echo "--- [4/5] eval boundary ---"
  python src/evaluation/evaluate_variant.py --variant boundary

  # Headline Phase-5 metrics live in summary.csv: HD95 + NSD per region.
  BND_DIR=$(ls -1d results/boundary/eval_* 2>/dev/null | sort | tail -1)
  if [ -n "$BND_DIR" ] && [ -f "$BND_DIR/summary.csv" ]; then
    echo
    echo "--- summary.csv (HD95 / NSD focus) ---"
    python - <<PY
import pandas as pd
df = pd.read_csv("$BND_DIR/summary.csv")
cols = ["mode"] + [c for c in df.columns if c.lower().startswith(("hd95", "nsd", "dice"))]
print(df[cols].to_string(index=False))
PY
  fi
fi

# ---------------------------------------------------------------------------
# 5. Paired Wilcoxon: boundary vs uncertainty.
# ---------------------------------------------------------------------------
if [ "$SKIP_COMPARE" = "0" ]; then
  echo
  echo "--- [5/5] paired Wilcoxon (TTA + postprocess) ---"
  BD=$(ls -1d results/boundary/eval_*    2>/dev/null | sort | tail -1)
  UC=$(ls -1d results/uncertainty/eval_* 2>/dev/null | sort | tail -1)
  if [ -z "$BD" ] || [ -z "$UC" ]; then
    echo "[error] missing eval folder. boundary=$BD  uncertainty=$UC"
    echo "[hint]  Phase 4 must have been run before Phase 5 compare."
    exit 1
  fi
  echo "boundary    eval : $BD"
  echo "uncertainty eval : $UC"
  python -m evaluation.stats compare \
      "$UC/per_case_metrics.csv" \
      "$BD/per_case_metrics.csv" \
      --mode tta_post --label-a uncertainty --label-b boundary
fi

echo
echo "================================================================"
echo "Phase 5 done."
echo "  boundary logs : logs/run_boundary_${EXP_NAME}_*/"
echo "  boundary eval : results/boundary/eval_*/"
echo "  complexity    : results/complexity.csv"
echo
echo "  Phase-5 headline metrics are HD95 and NSD (surface distance), not Dice."
echo "  Dice may move sideways; the win is sharper boundaries."
echo "  Pay special attention to TC — Phase 4 had a small regression we hope"
echo "  to recover here (boundary cues drive TC accuracy the most)."
echo "================================================================"
echo
echo "Detach now with Ctrl-b d, or just close the terminal — tmux keeps it."
