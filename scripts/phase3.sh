#!/usr/bin/env bash
# Phase 3 — hierarchical spectral Swin vs Phase-2 frequency baseline.
#
# Pipeline:
#   1. Sanity: cuda forward+backward at 128^3 under bf16 autocast.
#      Two-stage check (matches phase2.sh):
#        (a) fp32 forward+backward — confirms grads land on every parameter,
#            including each spectral block's `alpha` (FrequencyAwareBlock's
#            `band_gain` will have grad 0 at init because α=0 zeros the
#            spectral contribution; this is by design).
#        (b) bf16 autocast forward — confirms windowed attention softmax is
#            stable in bf16 (fp16 overflows here).
#   2. Complexity profile for `spectral_swin` and `frequency` (FLOPs delta is
#      a key Phase-3 success criterion — plan calls for +20-30% over freq).
#   3. Train spectral_swin (spectral_swin preset → bf16, no GradScaler).
#   4. Eval  spectral_swin (auto-picks bf16 because arch_family=transformer).
#   5. Paired Wilcoxon: spectral_swin vs frequency at TTA+post.
#
# Phase-2 reference numbers (TTA+post):
#   base_cnn   mean Dice 0.8253
#   cross_mod  mean Dice 0.8242
#   frequency  mean Dice 0.8207  (TC regression vs cross_modal, ET +1.8 pts)
# Phase-3 success criterion: spectral_swin mean Dice ≥ frequency, with the
# strong target being recovery of TC. The plan also notes Param count should
# be in the 22-40M range and FLOPs ~+20-30% over frequency.
#
# Auto-launches into a tmux session named "phase3". Detach with Ctrl-b d.
# Reattach with `tmux attach -t phase3`. Kill with `tmux kill-session -t phase3`.
#
# Usage:
#   bash scripts/phase3.sh                          # full pipeline
#   SKIP_SPECTRAL=1 bash scripts/phase3.sh          # skip train+eval if done
#   EPOCHS=50       bash scripts/phase3.sh          # quick A/B
#
# Skip flags (set =1 to skip):
#   SKIP_SANITY, SKIP_COMPLEXITY, SKIP_SPECTRAL (train+eval), SKIP_COMPARE
#
# Pre-reqs on the pod:
#   - PyTorch + project requirements installed
#   - BraTS preprocessed; src/configs/config.py:TRAIN_DATA_PATH set
#   - Phase 2 already completed → results/frequency/eval_*/per_case_metrics.csv
#   - One CUDA GPU visible (bf16-capable: Ampere+ / 4090 / A100)
#   - tmux available (script installs it on Debian/Ubuntu pods if missing)

set -euo pipefail

SESSION="${SESSION:-phase3}"

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

  echo "[info] launching Phase 3 in tmux session '$SESSION'."
  echo "       detach: Ctrl-b then d.   reattach: tmux attach -t $SESSION"

  tmux new-session -d -s "$SESSION" -c "$REPO_ABS" \
       -e "EPOCHS=${EPOCHS:-200}" \
       -e "EXP_NAME=${EXP_NAME:-phase3}" \
       -e "SKIP_SANITY=${SKIP_SANITY:-0}" \
       -e "SKIP_COMPLEXITY=${SKIP_COMPLEXITY:-0}" \
       -e "SKIP_SPECTRAL=${SKIP_SPECTRAL:-0}" \
       -e "SKIP_COMPARE=${SKIP_COMPARE:-0}" \
       "bash '$SCRIPT_ABS' $*; ec=\$?; echo; echo \"*** phase3.sh exited with code \$ec. Detach: Ctrl-b d. Close: Ctrl-d.\"; exec bash"

  exec tmux attach -t "$SESSION"
fi

# ---------------------------------------------------------------------------
# Inside tmux from here on.
# ---------------------------------------------------------------------------
EPOCHS="${EPOCHS:-200}"
EXP_NAME="${EXP_NAME:-phase3}"
SKIP_SANITY="${SKIP_SANITY:-0}"
SKIP_COMPLEXITY="${SKIP_COMPLEXITY:-0}"
SKIP_SPECTRAL="${SKIP_SPECTRAL:-0}"
SKIP_COMPARE="${SKIP_COMPARE:-0}"

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "================================================================"
echo "Phase 3 — spectral_swin vs frequency"
echo "  epochs   : $EPOCHS"
echo "  exp name : $EXP_NAME"
echo "  skip     : sanity=$SKIP_SANITY complexity=$SKIP_COMPLEXITY"
echo "             spectral=$SKIP_SPECTRAL compare=$SKIP_COMPARE"
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

assert torch.cuda.is_available(), "CUDA not available — abort."
m = build_variant("spectral_swin").cuda()
n_total_p = sum(p.numel() for p in m.parameters())
print(f"params: {n_total_p:,}")

cnn_p, tr_p = m.parameter_groups()
print(f"  cnn group:         {sum(p.numel() for p in cnn_p):,}")
print(f"  transformer group: {sum(p.numel() for p in tr_p):,}")

x = torch.randn(1, 5, 128, 128, 128, device="cuda")

# (a) fp32 forward+backward.
m.train()
out = m(x)
assert isinstance(out, tuple) and len(out) == 3
print("train shapes:", [tuple(t.shape) for t in out])
loss = sum(t.float().sum() for t in out) * 1e-6
loss.backward()
n_total = sum(1 for _ in m.parameters())
n_grad  = sum(1 for p in m.parameters() if p.grad is not None)
n_fin   = sum(1 for p in m.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
assert n_grad == n_total, f"missing grads on {n_total - n_grad} tensors"
assert n_fin  == n_total, f"non-finite grads on {n_total - n_fin} tensors"

# Spectral-swin specific: alpha (gates the spectral branch) must have grad;
# band_gain.grad is *expected* to be 0 at init (alpha=0 zeros the contribution).
ss = m.spectral_swin_stage
for stage_name in ("stage1", "stage2"):
    blks = getattr(ss, stage_name)
    for i, blk in enumerate(blks):
        ag = blk.alpha.grad
        assert ag is not None and torch.isfinite(ag).all(), \
            f"{stage_name}[{i}].alpha grad missing/NaN"
        assert float(blk.alpha) == 0.0, "alpha must init to 0"
print(f"fp32 backward: grads finite on {n_fin}/{n_total} tensors. "
      f"alpha grad finite on all 4 spectral blocks. "
      f"alpha init=0 confirmed.")

for p in m.parameters():
    p.grad = None

# (b) bf16 autocast forward — verifies windowed-attention softmax stable in bf16.
torch.cuda.reset_peak_memory_stats()
m.eval()
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    e = m(x)
assert torch.isfinite(e).all(), "bf16 autocast forward produced non-finite output"
print(f"bf16 autocast forward OK: dtype={e.dtype}, peak VRAM="
      f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB")
PY
fi

# ---------------------------------------------------------------------------
# 2. Complexity profile.
# ---------------------------------------------------------------------------
if [ "$SKIP_COMPLEXITY" = "0" ]; then
  echo
  echo "--- [2/5] complexity profile ---"
  mkdir -p results
  python -m evaluation.complexity --variant spectral_swin --device cuda \
      --out results/complexity.csv
  python -m evaluation.complexity --variant frequency     --device cuda \
      --out results/complexity.csv
fi

# ---------------------------------------------------------------------------
# 3-4. spectral_swin train + eval
#     - Training preset for spectral_swin is bf16 (fp16 overflows attention).
#     - Eval uses bf16 default because arch_family=transformer.
# ---------------------------------------------------------------------------
if [ "$SKIP_SPECTRAL" = "0" ]; then
  echo
  echo "--- [3/5] train spectral_swin ($EPOCHS epochs, bf16 preset) ---"
  python src/training/train_variant.py --variant spectral_swin \
      --epochs "$EPOCHS" --exp-name "$EXP_NAME"
  echo
  echo "--- [4/5] eval spectral_swin ---"
  python src/evaluation/evaluate_variant.py --variant spectral_swin
fi

# ---------------------------------------------------------------------------
# 5. Paired Wilcoxon: spectral_swin vs frequency.
# ---------------------------------------------------------------------------
if [ "$SKIP_COMPARE" = "0" ]; then
  echo
  echo "--- [5/5] paired Wilcoxon (TTA + postprocess) ---"
  SS=$(ls -1d results/spectral_swin/eval_* 2>/dev/null | sort | tail -1)
  FQ=$(ls -1d results/frequency/eval_*     2>/dev/null | sort | tail -1)
  if [ -z "$SS" ] || [ -z "$FQ" ]; then
    echo "[error] missing eval folder. spectral_swin=$SS  frequency=$FQ"
    echo "[hint]  Phase 2 must have been run before Phase 3 compare."
    exit 1
  fi
  echo "spectral_swin eval : $SS"
  echo "frequency     eval : $FQ"
  python -m evaluation.stats compare \
      "$FQ/per_case_metrics.csv" \
      "$SS/per_case_metrics.csv" \
      --mode tta_post --label-a frequency --label-b spectral_swin
fi

echo
echo "================================================================"
echo "Phase 3 done."
echo "  spectral_swin logs : logs/run_spectral_swin_${EXP_NAME}_*/"
echo "  spectral_swin eval : results/spectral_swin/eval_*/"
echo "  complexity table   : results/complexity.csv"
echo
echo "  Inspect learned spectral-swin gating (per block alpha):"
echo "    python -c \"import torch, glob; sd = torch.load(sorted(glob.glob('logs/run_spectral_swin_${EXP_NAME}_*/best_model.pth'))[-1], map_location='cpu'); m = sd if 'spectral_swin_stage.stage1.0.alpha' in sd else sd.get('model', sd); [print(k, float(m[k])) for k in m if k.endswith('.alpha') and 'spectral_swin_stage' in k]\""
echo "================================================================"
echo
echo "Detach now with Ctrl-b d, or just close the terminal — tmux keeps it."
