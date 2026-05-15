#!/usr/bin/env bash
# Phase 2 — frequency-aware block vs Phase-1 cross_modal baseline.
#
# Pipeline:
#   1. Sanity: cuda forward+backward at 128^3 under fp16 autocast
#      Confirms FFT path runs in fp32 inside autocast (no autocast crash).
#   2. Complexity profile for `frequency` (also re-profiles `cross_modal`
#      so the two rows in results/complexity.csv are comparable).
#   3. Train frequency  (transformer preset, same recipe as Phase 1)
#   4. Eval  frequency
#   5. Paired Wilcoxon: frequency vs cross_modal at TTA+post
#
# Phase-1 outcome (locked in):
#   base_cnn  TTA+post mean Dice: 0.8253
#   cross_mod TTA+post mean Dice: 0.8242   (statistically tied)
# Phase-2 success criterion: frequency variant Dice not below cross_modal noise
#   floor AND FLOPs increase < 5%. Spectral filter validates only if the
#   per-channel × per-band gain matrix drifts away from 1.0 during training
#   (inspect best_model.pth → freq_block.band_gain after training).
#
# Auto-launches into a tmux session named "phase2" so an SSH disconnect
# doesn't kill ~12 hours of training. Detach with Ctrl-b d. Reattach with
# `tmux attach -t phase2`. Kill with `tmux kill-session -t phase2`.
#
# Usage:
#   bash scripts/phase2.sh                     # full pipeline, ~12 h on 4090
#   SKIP_FREQUENCY=1 bash scripts/phase2.sh    # skip if you already trained it
#   EPOCHS=50        bash scripts/phase2.sh    # quick A/B
#
# Skip flags (set =1 to skip):
#   SKIP_SANITY, SKIP_COMPLEXITY, SKIP_FREQUENCY (train+eval), SKIP_COMPARE
#
# Pre-reqs on the pod:
#   - PyTorch + project requirements installed
#   - BraTS preprocessed; src/configs/config.py:TRAIN_DATA_PATH set
#   - Phase 1 already completed → results/cross_modal/eval_*/per_case_metrics.csv
#   - One CUDA GPU visible
#   - tmux available (script installs it on Debian/Ubuntu pods if missing)

set -euo pipefail

SESSION="${SESSION:-phase2}"

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

  echo "[info] launching Phase 2 in tmux session '$SESSION'."
  echo "       detach: Ctrl-b then d.   reattach: tmux attach -t $SESSION"

  tmux new-session -d -s "$SESSION" -c "$REPO_ABS" \
       -e "EPOCHS=${EPOCHS:-200}" \
       -e "EXP_NAME=${EXP_NAME:-phase2}" \
       -e "SKIP_SANITY=${SKIP_SANITY:-0}" \
       -e "SKIP_COMPLEXITY=${SKIP_COMPLEXITY:-0}" \
       -e "SKIP_FREQUENCY=${SKIP_FREQUENCY:-0}" \
       -e "SKIP_COMPARE=${SKIP_COMPARE:-0}" \
       "bash '$SCRIPT_ABS' $*; ec=\$?; echo; echo \"*** phase2.sh exited with code \$ec. Detach: Ctrl-b d. Close: Ctrl-d.\"; exec bash"

  exec tmux attach -t "$SESSION"
fi

# ---------------------------------------------------------------------------
# Inside tmux from here on.
# ---------------------------------------------------------------------------
EPOCHS="${EPOCHS:-200}"
EXP_NAME="${EXP_NAME:-phase2}"
SKIP_SANITY="${SKIP_SANITY:-0}"
SKIP_COMPLEXITY="${SKIP_COMPLEXITY:-0}"
SKIP_FREQUENCY="${SKIP_FREQUENCY:-0}"
SKIP_COMPARE="${SKIP_COMPARE:-0}"

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "================================================================"
echo "Phase 2 — frequency vs cross_modal"
echo "  epochs   : $EPOCHS"
echo "  exp name : $EXP_NAME"
echo "  skip     : sanity=$SKIP_SANITY complexity=$SKIP_COMPLEXITY"
echo "             frequency=$SKIP_FREQUENCY compare=$SKIP_COMPARE"
echo "================================================================"

pip install -q fvcore >/dev/null 2>&1 || echo "[warn] fvcore unavailable; FLOPs will be NaN"

# ---------------------------------------------------------------------------
# 1. Sanity check: two-stage so the fp16-overflow risk of `.sum()` over 8M
#    voxels doesn't masquerade as an FFT-autocast bug.
#      (a) fp32 forward + backward — verifies grads flow on every parameter,
#          including `freq_block.band_gain` (this is the only test that
#          *actually* exercises the FFT path's gradient).
#      (b) fp16 autocast forward    — verifies the FFT path doesn't crash
#          under autocast. The block wraps FFT in autocast(enabled=False)
#          and casts in/out to fp16; this confirms that contract.
# ---------------------------------------------------------------------------
if [ "$SKIP_SANITY" = "0" ]; then
  echo
  echo "--- [1/5] sanity: forward + backward on cuda ---"
  python - <<'PY'
import torch
from model.registry import build_variant

assert torch.cuda.is_available(), "CUDA not available — abort."
m = build_variant("frequency").cuda()
print(f"params: {sum(p.numel() for p in m.parameters()):,}")
x = torch.randn(1, 5, 128, 128, 128, device="cuda")

# (a) fp32 forward + backward.
m.train()
out = m(x)
assert isinstance(out, tuple) and len(out) == 3, f"unexpected output: {type(out)}"
print("train shapes:", [tuple(t.shape) for t in out])
# Scale-down on the fp32 sum keeps loss/grad magnitudes well inside fp32 range
# regardless of patch size; we only need grads to flow, not be physically meaningful.
loss = sum(t.float().sum() for t in out) * 1e-6
loss.backward()
n_total = sum(1 for _ in m.parameters())
n_grad  = sum(1 for p in m.parameters() if p.grad is not None)
n_fin   = sum(1 for p in m.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
assert n_grad == n_total, f"missing grads on {n_total - n_grad} tensors"
assert n_fin  == n_total, f"non-finite grads on {n_total - n_fin} tensors"
g = m.freq_block.band_gain.grad
assert g is not None and torch.isfinite(g).all(), "band_gain grad missing/NaN"
print(f"fp32 backward: grads finite on {n_fin}/{n_total} tensors; "
      f"band_gain.grad norm = {float(g.norm()):.3e}")

# Reset grads before the autocast forward test.
for p in m.parameters():
    p.grad = None

# (b) fp16 autocast forward — exercises FFT-under-autocast contract.
torch.cuda.reset_peak_memory_stats()
m.eval()
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
    e = m(x)
assert torch.isfinite(e).all(), "fp16 autocast forward produced non-finite output"
print(f"fp16 autocast forward OK: dtype={e.dtype}, peak VRAM="
      f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB")
PY
fi

# ---------------------------------------------------------------------------
# 2. Complexity profile (frequency + cross_modal so they're side by side).
# ---------------------------------------------------------------------------
if [ "$SKIP_COMPLEXITY" = "0" ]; then
  echo
  echo "--- [2/5] complexity profile ---"
  mkdir -p results
  python -m evaluation.complexity --variant frequency   --device cuda \
      --out results/complexity.csv
  python -m evaluation.complexity --variant cross_modal --device cuda \
      --out results/complexity.csv
fi

# ---------------------------------------------------------------------------
# 3-4. frequency train + eval
# ---------------------------------------------------------------------------
if [ "$SKIP_FREQUENCY" = "0" ]; then
  echo
  echo "--- [3/5] train frequency ($EPOCHS epochs, transformer preset) ---"
  python src/training/train_variant.py --variant frequency \
      --preset transformer --epochs "$EPOCHS" --exp-name "$EXP_NAME"
  echo
  echo "--- [4/5] eval frequency ---"
  python src/evaluation/evaluate_variant.py --variant frequency
fi

# ---------------------------------------------------------------------------
# 5. Paired Wilcoxon: frequency vs cross_modal (newest eval folder per variant).
# ---------------------------------------------------------------------------
if [ "$SKIP_COMPARE" = "0" ]; then
  echo
  echo "--- [5/5] paired Wilcoxon (TTA + postprocess) ---"
  FQ=$(ls -1d results/frequency/eval_*   2>/dev/null | sort | tail -1)
  CM=$(ls -1d results/cross_modal/eval_* 2>/dev/null | sort | tail -1)
  if [ -z "$FQ" ] || [ -z "$CM" ]; then
    echo "[error] missing eval folder. frequency=$FQ  cross_modal=$CM"
    echo "[hint]  Phase 1 must have been run before Phase 2 compare."
    exit 1
  fi
  echo "frequency   eval : $FQ"
  echo "cross_modal eval : $CM"
  python -m evaluation.stats compare \
      "$CM/per_case_metrics.csv" \
      "$FQ/per_case_metrics.csv" \
      --mode tta_post --label-a cross_modal --label-b frequency
fi

echo
echo "================================================================"
echo "Phase 2 done."
echo "  frequency logs   : logs/run_frequency_${EXP_NAME}_*/"
echo "  frequency eval   : results/frequency/eval_*/"
echo "  complexity table : results/complexity.csv"
echo
echo "  Inspect learned spectral gains:"
echo "    python -c \"import torch; sd = torch.load(sorted(__import__('glob').glob('logs/run_frequency_${EXP_NAME}_*/best_model.pth'))[-1], map_location='cpu'); g = sd.get('freq_block.band_gain', sd.get('model', sd).get('freq_block.band_gain')); print(g)\""
echo "================================================================"
echo
echo "Detach now with Ctrl-b d, or just close the terminal — tmux keeps it."
