#!/usr/bin/env bash
# hybrid - clean-slate CNN+Transformer built solely to beat the `unet3d`
# lower-bound baseline on the same harness.
#
# `hybrid` (src/model/hybrid.py, registry key "hybrid") is a TransBTS-style
# residual-CNN encoder + 4-layer transformer at the 8^3 bottleneck + CNN
# decoder, with 3-level deep supervision and 2x Dropout3d (eval MC-Dropout).
# 4-class softmax head -> standard argmax decode (no sigmoid TC-leak).
#
# Recipe = `hybrid_preset()` in train_variant.py: an explicit clone of the
# recipe that made unet3d win (default TrainingPreset: fp16, EMA, gpu_aug,
# grad_clip, val_dice selection, ncr_sample_prob=0.25, class_weights
# (0.1,2,1,1), ce_weight 0.3, single AdamW group) + top-K=5 snapshot
# ensemble. Effective batch 32 (BATCH*ACCUM) = unet3d parity.
#
# Pipeline:
#   1. Sanity (CUDA): 128^3 forward+backward; train 3-tuple DS shapes; eval
#      bare tensor; Dropout3d present; preset matches unet3d's recipe.
#   2. Complexity profile (params/FLOPs/latency) -> results/complexity.csv.
#   3. Train `hybrid` (hybrid_preset; writes best_model.pth + snapshot_top*.pth).
#   4a. Eval single best_model            -> results/hybrid/eval_single/
#   4b. Eval snapshot-ensemble + ext-TTA  -> results/hybrid/eval_ensemble/
#   5. Paired Wilcoxon + bootstrap CI vs the latest results/unet3d/eval_*,
#      both tta_post and baseline_post, for single AND ensemble.
#
# Auto-launches into a tmux session named "hybrid". Detach: Ctrl-b d.
# Reattach: tmux attach -t hybrid. Kill: tmux kill-session -t hybrid.
#
# Usage:
#   bash scripts/hybrid.sh                          # full pipeline (300 ep)
#   EPOCHS=2 bash scripts/hybrid.sh                 # short smoke
#   AMP=bf16 bash scripts/hybrid.sh                 # fp16-NaN escape hatch
#   SKIP_TRAIN=1 bash scripts/hybrid.sh             # eval+compare only
#   BATCH=4 ACCUM=8 bash scripts/hybrid.sh          # alt eff-batch-32 split
#   UNET3D_DIR=results/unet3d/eval_X bash scripts/hybrid.sh   # pin baseline
#
# Skip flags (set =1): SKIP_SANITY SKIP_COMPLEXITY SKIP_TRAIN SKIP_EVAL SKIP_COMPARE
#
# Pre-reqs: PyTorch + project reqs installed; BraTS preprocessed and
# src/configs/config.py:TRAIN_DATA_PATH set; one CUDA GPU (5090/32GB target);
# results/unet3d/eval_*/per_case_metrics.csv already exists (for step 5).

set -euo pipefail

SESSION="${SESSION:-hybrid}"

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

  echo "[info] launching hybrid pipeline in tmux session '$SESSION'."
  echo "       detach: Ctrl-b then d.   reattach: tmux attach -t $SESSION"

  tmux new-session -d -s "$SESSION" -c "$REPO_ABS" \
       -e "EPOCHS=${EPOCHS:-300}" \
       -e "WARMUP=${WARMUP:-}" \
       -e "BATCH=${BATCH:-8}" \
       -e "ACCUM=${ACCUM:-4}" \
       -e "AMP=${AMP:-fp16}" \
       -e "EXP_NAME=${EXP_NAME:-}" \
       -e "UNET3D_DIR=${UNET3D_DIR:-}" \
       -e "SKIP_SANITY=${SKIP_SANITY:-0}" \
       -e "SKIP_COMPLEXITY=${SKIP_COMPLEXITY:-0}" \
       -e "SKIP_TRAIN=${SKIP_TRAIN:-0}" \
       -e "SKIP_EVAL=${SKIP_EVAL:-0}" \
       -e "SKIP_COMPARE=${SKIP_COMPARE:-0}" \
       "bash '$SCRIPT_ABS' $*; ec=\$?; echo; echo \"*** hybrid.sh exited with code \$ec. Detach: Ctrl-b d. Close: Ctrl-d.\"; exec bash"

  exec tmux attach -t "$SESSION"
fi

# ---------------------------------------------------------------------------
# Inside tmux from here on.
# ---------------------------------------------------------------------------
EPOCHS="${EPOCHS:-300}"
WARMUP="${WARMUP:-}"          # empty -> let hybrid_preset decide (20 epochs)
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-4}"
AMP="${AMP:-fp16}"
EXP_NAME="${EXP_NAME:-}"
UNET3D_DIR="${UNET3D_DIR:-}"
SKIP_SANITY="${SKIP_SANITY:-0}"
SKIP_COMPLEXITY="${SKIP_COMPLEXITY:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_COMPARE="${SKIP_COMPARE:-0}"

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

EXP_ARGS=()
[ -n "$EXP_NAME" ] && EXP_ARGS=(--exp-name "$EXP_NAME")

WARMUP_ARGS=()
[ -n "$WARMUP" ] && WARMUP_ARGS=(--warmup "$WARMUP")

echo "================================================================"
echo "hybrid - beat the unet3d baseline"
echo "  epochs   : $EPOCHS   warmup: ${WARMUP:-<preset:20>}"
echo "  batch    : $BATCH x accum $ACCUM  (effective $((BATCH*ACCUM)) = unet3d parity)"
echo "  amp      : $AMP"
echo "  exp name : ${EXP_NAME:-<none>}"
echo "  skip     : sanity=$SKIP_SANITY complexity=$SKIP_COMPLEXITY"
echo "             train=$SKIP_TRAIN eval=$SKIP_EVAL compare=$SKIP_COMPARE"
echo "================================================================"

pip install -q fvcore >/dev/null 2>&1 || echo "[warn] fvcore unavailable; FLOPs will be NaN"

# ---------------------------------------------------------------------------
# 1. Sanity (CUDA): contract + preset parity.
# ---------------------------------------------------------------------------
if [ "$SKIP_SANITY" = "0" ]; then
  echo
  echo "--- [1/5] sanity: 128^3 fwd/bwd + contract + preset parity ---"
  python - <<'PY'
import torch
from model.registry import build_variant, get_output_mode, get_arch_family
from training.train_variant import get_preset

assert torch.cuda.is_available(), "CUDA not available - abort."
m = build_variant("hybrid").cuda()
n = sum(p.numel() for p in m.parameters())
print(f"hybrid params: {n:,}")
assert get_output_mode("hybrid") == "softmax"
assert get_arch_family("hybrid") == "cnn"          # -> eval AMP fp16, like unet3d

x = torch.randn(2, 5, 128, 128, 128, device="cuda")
m.train()
f, d1, d2 = m(x)
assert tuple(f.shape)  == (2, 4, 128, 128, 128), f.shape
assert tuple(d1.shape) == (2, 4, 64, 64, 64),    d1.shape
assert tuple(d2.shape) == (2, 4, 32, 32, 32),    d2.shape
(f.float().mean() + 0.5*d1.float().mean() + 0.25*d2.float().mean()).backward()
bad = [nm for nm, p in m.named_parameters()
       if p.grad is None or not torch.isfinite(p.grad).all()]
assert not bad, f"missing/non-finite grads: {bad[:5]}"
m.eval()
with torch.no_grad():
    o = m(x)
assert torch.is_tensor(o) and tuple(o.shape) == (2, 4, 128, 128, 128), o.shape
assert any(isinstance(mm, torch.nn.Dropout3d) for mm in m.modules()), \
    "no Dropout3d -> eval MC-Dropout would be skipped"

p = get_preset("hybrid")
assert p.amp_dtype == "fp16" and p.use_ema and p.use_gpu_aug and p.use_grad_clip
assert p.best_metric == "val_dice" and p.ncr_sample_prob == 0.25
assert p.class_weights == (0.1, 2.0, 1.0, 1.0) and p.ce_weight == 0.3
assert p.top_k_snapshots == 5 and p.use_param_groups is False
assert p.warmup_epochs == 20      # longer ramp: transformer + small data
print("sanity OK: train 3-tuple DS, eval tensor, Dropout3d present, "
      "preset == unet3d recipe + top-K=5 + warmup=20")
PY
fi

# ---------------------------------------------------------------------------
# 2. Complexity profile.
# ---------------------------------------------------------------------------
if [ "$SKIP_COMPLEXITY" = "0" ]; then
  echo
  echo "--- [2/5] complexity profile -> results/complexity.csv ---"
  python -m evaluation.complexity --variant hybrid --out results/complexity.csv \
    || echo "[warn] complexity profiling failed; continuing"
fi

# ---------------------------------------------------------------------------
# 3. Train.
# ---------------------------------------------------------------------------
if [ "$SKIP_TRAIN" = "0" ]; then
  echo
  echo "--- [3/5] train hybrid ($EPOCHS ep, batch $BATCH x accum $ACCUM, $AMP) ---"
  python src/training/train_variant.py --variant hybrid "${EXP_ARGS[@]}" \
    --epochs "$EPOCHS" "${WARMUP_ARGS[@]}" \
    --batch-size "$BATCH" --accum-steps "$ACCUM" --amp-dtype "$AMP"
fi

# Resolve the run dir (newest run_hybrid_*). Used for explicit checkpoint
# selection so eval never auto-discovers a stale/foreign run.
RUN_DIR="$(ls -dt logs/run_hybrid_*/ 2>/dev/null | head -1 || true)"
RUN_DIR="${RUN_DIR%/}"
if [ -n "$RUN_DIR" ]; then
  echo "[info] hybrid run dir: $RUN_DIR"
else
  echo "[warn] no logs/run_hybrid_*/ found - eval will auto-discover"
fi

# ---------------------------------------------------------------------------
# 4. Evaluate: (a) single best_model, (b) snapshot ensemble + extended TTA.
# ---------------------------------------------------------------------------
if [ "$SKIP_EVAL" = "0" ]; then
  CKPT_ARGS=()
  ENS_GLOB="logs/run_hybrid_*/snapshot_top*.pth"
  if [ -n "$RUN_DIR" ]; then
    CKPT_ARGS=(--checkpoint "$RUN_DIR/best_model.pth")
    ENS_GLOB="$RUN_DIR/snapshot_top*.pth"
  fi

  echo
  echo "--- [4a/5] eval single best_model -> results/hybrid/eval_single/ ---"
  python src/evaluation/evaluate_variant.py --variant hybrid \
    "${CKPT_ARGS[@]}" --vmin-sweep --run-name eval_single

  echo
  echo "--- [4b/5] eval snapshot-ensemble + extended TTA -> results/hybrid/eval_ensemble/ ---"
  python src/evaluation/evaluate_variant.py --variant hybrid \
    --ensemble-ckpts "$ENS_GLOB" --tta-extended \
    --vmin-sweep --run-name eval_ensemble
fi

# ---------------------------------------------------------------------------
# 5. Compare vs unet3d baseline (paired Wilcoxon + bootstrap 95% CI).
# ---------------------------------------------------------------------------
if [ "$SKIP_COMPARE" = "0" ]; then
  if [ -z "$UNET3D_DIR" ]; then
    UNET3D_DIR="$(ls -dt results/unet3d/eval_*/ 2>/dev/null | head -1 || true)"
    UNET3D_DIR="${UNET3D_DIR%/}"
  fi
  BASE_CSV="$UNET3D_DIR/per_case_metrics.csv"
  if [ ! -f "$BASE_CSV" ]; then
    echo "[error] no unet3d baseline metrics at '$BASE_CSV'."
    echo "        Evaluate unet3d first or pass UNET3D_DIR=results/unet3d/eval_X"
    exit 1
  fi
  echo
  echo "--- [5/5] compare vs unet3d: $BASE_CSV ---"
  for tag in single ensemble; do
    HY_CSV="results/hybrid/eval_${tag}/per_case_metrics.csv"
    [ -f "$HY_CSV" ] || { echo "[skip] $HY_CSV missing"; continue; }
    for mode in tta_post baseline_post; do
      echo
      echo "===== hybrid:$tag vs unet3d  [mode=$mode] ====="
      python -m evaluation.stats compare "$BASE_CSV" "$HY_CSV" \
        --mode "$mode" --label-a unet3d --label-b "hybrid_${tag}" || true
    done
  done
  echo
  echo "Win criterion: hybrid:single tta_post mean Dice >= unet3d AND TC not"
  echo "regressed (paired Wilcoxon) = clean same-apparatus claim."
  echo "hybrid:ensemble = decisive headline (ensemble-vs-single, disclose)."
fi

echo
echo "================================================================"
echo "hybrid pipeline done."
echo "================================================================"
