#!/usr/bin/env bash
# Phase 1 — cross-modal attention vs matched-recipe base_cnn.
#
# Pipeline:
#   1. Sanity: cuda forward+backward at 128^3 under fp16 autocast
#   2. Complexity profile for cross_modal and base_cnn
#   3. Train cross_modal (transformer preset)
#   4. Eval  cross_modal
#   5. Train base_cnn under transformer preset (matched baseline)
#   6. Eval  base_cnn
#   7. Paired Wilcoxon: cross_modal vs base_cnn at TTA+post
#
# Auto-launches into a tmux session named "phase1" so an SSH disconnect
# doesn't kill ~12 hours of training. Detach with Ctrl-b d. Reattach with
# `tmux attach -t phase1`. Kill with `tmux kill-session -t phase1`.
#
# Usage:
#   bash scripts/phase1.sh                      # full pipeline, ~12 h on 4090
#   SKIP_CROSS_MODAL=1 bash scripts/phase1.sh   # skip if you already trained it
#   SKIP_BASELINE=1    bash scripts/phase1.sh   # skip if you already trained it
#   EPOCHS=50          bash scripts/phase1.sh   # quick A/B
#
# Skip flags (set =1 to skip):
#   SKIP_SANITY, SKIP_COMPLEXITY,
#   SKIP_CROSS_MODAL  (train + eval),
#   SKIP_BASELINE     (train + eval),
#   SKIP_COMPARE
#
# Pre-reqs on the pod:
#   - PyTorch + project requirements installed
#   - BraTS preprocessed; src/configs/config.py:TRAIN_DATA_PATH set
#   - One CUDA GPU visible
#   - tmux available (script installs it on Debian/Ubuntu pods if missing)

set -euo pipefail

SESSION="${SESSION:-phase1}"

# ---------------------------------------------------------------------------
# Re-exec inside tmux if not already in a session.
# Pattern: create a detached session running this script, then attach.
# After the script finishes the session keeps an interactive shell open
# so failures stay on screen instead of vanishing as "[exited]".
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

  echo "[info] launching Phase 1 in tmux session '$SESSION'."
  echo "       detach: Ctrl-b then d.   reattach: tmux attach -t $SESSION"

  # Create a detached session that runs the script, then drops into an
  # interactive bash so the user can read output even if the script crashed.
  tmux new-session -d -s "$SESSION" -c "$REPO_ABS" \
       -e "EPOCHS=${EPOCHS:-200}" \
       -e "EXP_NAME=${EXP_NAME:-phase1}" \
       -e "SKIP_SANITY=${SKIP_SANITY:-0}" \
       -e "SKIP_COMPLEXITY=${SKIP_COMPLEXITY:-0}" \
       -e "SKIP_CROSS_MODAL=${SKIP_CROSS_MODAL:-0}" \
       -e "SKIP_BASELINE=${SKIP_BASELINE:-0}" \
       -e "SKIP_COMPARE=${SKIP_COMPARE:-0}" \
       "bash '$SCRIPT_ABS' $*; ec=\$?; echo; echo \"*** phase1.sh exited with code \$ec. Detach: Ctrl-b d. Close: Ctrl-d.\"; exec bash"

  exec tmux attach -t "$SESSION"
fi

# ---------------------------------------------------------------------------
# Inside tmux from here on.
# ---------------------------------------------------------------------------
EPOCHS="${EPOCHS:-200}"
EXP_NAME="${EXP_NAME:-phase1}"
SKIP_SANITY="${SKIP_SANITY:-0}"
SKIP_COMPLEXITY="${SKIP_COMPLEXITY:-0}"
SKIP_CROSS_MODAL="${SKIP_CROSS_MODAL:-0}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
SKIP_COMPARE="${SKIP_COMPARE:-0}"

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "================================================================"
echo "Phase 1 — cross_modal vs matched base_cnn"
echo "  epochs   : $EPOCHS"
echo "  exp name : $EXP_NAME"
echo "  skip     : sanity=$SKIP_SANITY complexity=$SKIP_COMPLEXITY"
echo "             cross_modal=$SKIP_CROSS_MODAL baseline=$SKIP_BASELINE"
echo "             compare=$SKIP_COMPARE"
echo "================================================================"

pip install -q fvcore >/dev/null 2>&1 || echo "[warn] fvcore unavailable; FLOPs will be NaN"

# ---------------------------------------------------------------------------
# 1. Sanity check
# ---------------------------------------------------------------------------
if [ "$SKIP_SANITY" = "0" ]; then
  echo
  echo "--- [1/7] sanity: forward + backward on cuda ---"
  python - <<'PY'
import torch
from model.registry import build_variant

assert torch.cuda.is_available(), "CUDA not available — abort."
m = build_variant("cross_modal").cuda()
print(f"params: {sum(p.numel() for p in m.parameters()):,}")
x = torch.randn(1, 5, 128, 128, 128, device="cuda")
m.train()
with torch.amp.autocast("cuda", dtype=torch.float16):
    out = m(x)
assert isinstance(out, tuple) and len(out) == 3, f"unexpected output: {type(out)}"
print("train shapes:", [tuple(t.shape) for t in out])
loss = sum(t.float().sum() for t in out)
loss.backward()
n_grad = sum(1 for p in m.parameters() if p.grad is not None)
n_total = sum(1 for _ in m.parameters())
assert n_grad == n_total, f"missing grads on {n_total - n_grad} tensors"
print(f"grads on {n_grad}/{n_total} param tensors; peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
PY
fi

# ---------------------------------------------------------------------------
# 2. Complexity profile (both variants → results/complexity.csv)
# ---------------------------------------------------------------------------
if [ "$SKIP_COMPLEXITY" = "0" ]; then
  echo
  echo "--- [2/7] complexity profile ---"
  mkdir -p results
  python -m evaluation.complexity --variant cross_modal --device cuda \
      --out results/complexity.csv
  python -m evaluation.complexity --variant base_cnn --device cuda \
      --out results/complexity.csv
fi

# ---------------------------------------------------------------------------
# 3-4. cross_modal train + eval
# ---------------------------------------------------------------------------
if [ "$SKIP_CROSS_MODAL" = "0" ]; then
  echo
  echo "--- [3/7] train cross_modal ($EPOCHS epochs, transformer preset) ---"
  python src/training/train_variant.py --variant cross_modal \
      --preset transformer --epochs "$EPOCHS" --exp-name "$EXP_NAME"
  echo
  echo "--- [4/7] eval cross_modal ---"
  python src/evaluation/evaluate_variant.py --variant cross_modal
fi

# ---------------------------------------------------------------------------
# 5-6. base_cnn matched-recipe train + eval
# ---------------------------------------------------------------------------
if [ "$SKIP_BASELINE" = "0" ]; then
  echo
  echo "--- [5/7] train base_cnn matched ($EPOCHS epochs, transformer preset) ---"
  python src/training/train_variant.py --variant base_cnn \
      --preset transformer --epochs "$EPOCHS" --exp-name "${EXP_NAME}_matched"
  echo
  echo "--- [6/7] eval base_cnn matched ---"
  python src/evaluation/evaluate_variant.py --variant base_cnn
fi

# ---------------------------------------------------------------------------
# 7. Paired Wilcoxon: pick newest eval dir per variant.
# ---------------------------------------------------------------------------
if [ "$SKIP_COMPARE" = "0" ]; then
  echo
  echo "--- [7/7] paired Wilcoxon (TTA + postprocess) ---"
  CM=$(ls -1d results/cross_modal/eval_* 2>/dev/null | sort | tail -1)
  BC=$(ls -1d results/base_cnn/eval_*    2>/dev/null | sort | tail -1)
  if [ -z "$CM" ] || [ -z "$BC" ]; then
    echo "[error] missing eval folder. cross_modal=$CM  base_cnn=$BC"
    exit 1
  fi
  echo "cross_modal eval : $CM"
  echo "base_cnn eval    : $BC"
  python -m evaluation.stats compare \
      "$BC/per_case_metrics.csv" \
      "$CM/per_case_metrics.csv" \
      --mode tta_post --label-a base_cnn_matched --label-b cross_modal
fi

echo
echo "================================================================"
echo "Phase 1 done."
echo "  cross_modal logs   : logs/run_cross_modal_${EXP_NAME}_*/"
echo "  base_cnn   logs   : logs/run_base_cnn_${EXP_NAME}_matched_*/"
echo "  cross_modal eval  : results/cross_modal/eval_*/"
echo "  base_cnn   eval  : results/base_cnn/eval_*/"
echo "  complexity table  : results/complexity.csv"
echo "================================================================"
echo
echo "Detach now with Ctrl-b d, or just close the terminal — tmux keeps it."
