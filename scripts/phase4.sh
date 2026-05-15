#!/usr/bin/env bash
# Phase 4 — uncertainty-guided bottleneck vs Phase-3 spectral_swin baseline.
#
# Pipeline:
#   1. Sanity: cuda forward+backward at 128^3 under bf16 autocast.
#      (a) fp32 forward+backward — confirms grads land on every parameter,
#          including the uncertainty block's `alpha` (variance gate scalar)
#          and the variance head conv params. alpha must init to 0.0 (the
#          block is identity at start of training).
#      (b) bf16 autocast forward — confirms downstream still matches the
#          spectral_swin path (no NaN from the new variance regulariser path).
#   2. Complexity profile for `uncertainty` and `spectral_swin` (FLOPs delta
#      should be a small bump — variance head is tiny vs the Swin stage).
#   3. Train uncertainty (uncertainty preset → bf16, UncertaintyAwareLoss
#      wrapper around RegionWiseDiceFocalLoss with λ_unc = 0.05).
#   4. Eval  uncertainty (auto-picks bf16 because arch_family=transformer).
#      Verify AURC + spearman_unc_error populate in evaluation_meta.json.
#   5. Paired Wilcoxon: uncertainty vs spectral_swin at TTA+post.
#
# Phase-3 reference numbers (TTA+post):
#   spectral_swin mean Dice 0.8279  (champion)
#   frequency     mean Dice 0.8207
#   base_cnn      mean Dice 0.8253
# Phase-4 success criterion: AURC and Spearman(unc, error) IMPROVE over
# spectral_swin. Dice may not move materially — the win is calibration.
#
# Auto-launches into a tmux session named "phase4". Detach with Ctrl-b d.
# Reattach with `tmux attach -t phase4`. Kill with `tmux kill-session -t phase4`.
#
# Usage:
#   bash scripts/phase4.sh                          # full pipeline
#   SKIP_UNCERTAINTY=1 bash scripts/phase4.sh       # skip train+eval if done
#   EPOCHS=50          bash scripts/phase4.sh       # quick A/B
#
# Skip flags (set =1 to skip):
#   SKIP_SANITY, SKIP_COMPLEXITY, SKIP_UNCERTAINTY (train+eval), SKIP_COMPARE
#
# Pre-reqs on the pod:
#   - PyTorch + project requirements installed
#   - BraTS preprocessed; src/configs/config.py:TRAIN_DATA_PATH set
#   - Phase 3 already completed → results/spectral_swin/eval_*/per_case_metrics.csv
#   - One CUDA GPU visible (bf16-capable: Ampere+ / 4090 / A100)
#   - tmux available (script installs it on Debian/Ubuntu pods if missing)

set -euo pipefail

SESSION="${SESSION:-phase4}"

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

  echo "[info] launching Phase 4 in tmux session '$SESSION'."
  echo "       detach: Ctrl-b then d.   reattach: tmux attach -t $SESSION"

  tmux new-session -d -s "$SESSION" -c "$REPO_ABS" \
       -e "EPOCHS=${EPOCHS:-200}" \
       -e "EXP_NAME=${EXP_NAME:-phase4}" \
       -e "SKIP_SANITY=${SKIP_SANITY:-0}" \
       -e "SKIP_COMPLEXITY=${SKIP_COMPLEXITY:-0}" \
       -e "SKIP_UNCERTAINTY=${SKIP_UNCERTAINTY:-0}" \
       -e "SKIP_COMPARE=${SKIP_COMPARE:-0}" \
       "bash '$SCRIPT_ABS' $*; ec=\$?; echo; echo \"*** phase4.sh exited with code \$ec. Detach: Ctrl-b d. Close: Ctrl-d.\"; exec bash"

  exec tmux attach -t "$SESSION"
fi

# ---------------------------------------------------------------------------
# Inside tmux from here on.
# ---------------------------------------------------------------------------
EPOCHS="${EPOCHS:-200}"
EXP_NAME="${EXP_NAME:-phase4}"
SKIP_SANITY="${SKIP_SANITY:-0}"
SKIP_COMPLEXITY="${SKIP_COMPLEXITY:-0}"
SKIP_UNCERTAINTY="${SKIP_UNCERTAINTY:-0}"
SKIP_COMPARE="${SKIP_COMPARE:-0}"

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "================================================================"
echo "Phase 4 — uncertainty vs spectral_swin"
echo "  epochs   : $EPOCHS"
echo "  exp name : $EXP_NAME"
echo "  skip     : sanity=$SKIP_SANITY complexity=$SKIP_COMPLEXITY"
echo "             uncertainty=$SKIP_UNCERTAINTY compare=$SKIP_COMPARE"
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
from training.losses import RegionWiseDiceFocalLoss, UncertaintyAwareLoss
from training.train_variant import _split_model_output

assert torch.cuda.is_available(), "CUDA not available — abort."
m = build_variant("uncertainty").cuda()
n_total_p = sum(p.numel() for p in m.parameters())
print(f"params: {n_total_p:,}")

cnn_p, tr_p = m.parameter_groups()
print(f"  cnn group:         {sum(p.numel() for p in cnn_p):,}")
print(f"  transformer group: {sum(p.numel() for p in tr_p):,}")

# alpha must init to 0
ub = m.uncertainty_block
assert float(ub.alpha) == 0.0, f"alpha must init to 0, got {float(ub.alpha)}"
print(f"uncertainty_block.alpha init: {float(ub.alpha):.6f} (identity at init)")

x = torch.randn(1, 5, 128, 128, 128, device="cuda")
target = torch.randint(0, 4, (1, 128, 128, 128), dtype=torch.long, device="cuda")

# (a) fp32 forward+backward — dict-shaped output, finite grads on every param.
m.train()
out = m(x)
assert isinstance(out, dict), f"expected dict output for uncertainty, got {type(out)}"
assert sorted(out.keys()) == ["boundary", "seg", "variance"], \
    f"unexpected dict keys {list(out.keys())}"
seg = out["seg"]
var = out["variance"]
assert isinstance(seg, tuple) and len(seg) == 3, "train: seg must be (final, ds1, ds2)"
print("train shapes:", [tuple(t.shape) for t in seg], "variance:", tuple(var.shape))
assert (var >= 0).all(), "variance must be non-negative (Softplus head)"
assert var.shape == (1, 1, 128, 128, 128), f"variance shape wrong: {tuple(var.shape)}"

seg_loss = RegionWiseDiceFocalLoss(gamma=2.0, ce_weight=0.3,
                                   class_weights=(0.1, 2.0, 1.0, 1.0))
crit = UncertaintyAwareLoss(seg_loss, lambda_unc=0.05, target_unc_at_high_dice=0.0)
loss = crit(seg, target, variance=var)
print(f"composite loss: {loss.item():.4f}  "
      f"(seg-only: {seg_loss(seg, target).item():.4f})")
loss.backward()

n_total = sum(1 for _ in m.parameters())
n_grad  = sum(1 for p in m.parameters() if p.grad is not None)
n_fin   = sum(1 for p in m.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
assert n_grad == n_total, f"missing grads on {n_total - n_grad} tensors"
assert n_fin  == n_total, f"non-finite grads on {n_total - n_fin} tensors"

# uncertainty-specific: alpha + var_head must have finite grads.
ag = ub.alpha.grad
assert ag is not None and torch.isfinite(ag).all(), "alpha grad missing/NaN"
print(f"alpha.grad: {float(ag):.4e}  (finite)")
for n_, p in ub.var_head.named_parameters():
    assert p.grad is not None and torch.isfinite(p.grad).all(), \
        f"var_head.{n_} grad missing/NaN"
print(f"var_head: all parameters got finite grads")

# Spectral-swin alpha (4 blocks) still works — Phase-3 invariant carried.
ss = m.spectral_swin_stage
for stage_name in ("stage1", "stage2"):
    blks = getattr(ss, stage_name)
    for i, blk in enumerate(blks):
        ssg = blk.alpha.grad
        assert ssg is not None and torch.isfinite(ssg).all(), \
            f"{stage_name}[{i}].alpha grad missing/NaN"
        assert float(blk.alpha) == 0.0, "spectral_swin alpha must init to 0"
print(f"fp32 backward: grads finite on {n_fin}/{n_total} tensors. "
      f"alpha grads OK on uncertainty + 4 spectral blocks. "
      f"alpha init=0 confirmed.")

for p in m.parameters():
    p.grad = None

# (b) bf16 autocast forward — verifies dict path / variance survives autocast.
torch.cuda.reset_peak_memory_stats()
m.eval()
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out_e = m(x)
assert isinstance(out_e, dict)
seg_e = out_e["seg"]
var_e = out_e["variance"]
assert isinstance(seg_e, torch.Tensor), "eval: seg must be a tensor"
assert torch.isfinite(seg_e).all(), "bf16 autocast eval produced non-finite seg"
assert torch.isfinite(var_e).all(), "bf16 autocast eval produced non-finite variance"
print(f"bf16 autocast forward OK: seg.dtype={seg_e.dtype}, var.dtype={var_e.dtype}, "
      f"peak VRAM={torch.cuda.max_memory_allocated()/1e9:.2f} GB")
PY
fi

# ---------------------------------------------------------------------------
# 2. Complexity profile.
# ---------------------------------------------------------------------------
if [ "$SKIP_COMPLEXITY" = "0" ]; then
  echo
  echo "--- [2/5] complexity profile ---"
  mkdir -p results
  python -m evaluation.complexity --variant uncertainty   --device cuda \
      --out results/complexity.csv
  python -m evaluation.complexity --variant spectral_swin --device cuda \
      --out results/complexity.csv
fi

# ---------------------------------------------------------------------------
# 3-4. uncertainty train + eval
#     - Training preset for `uncertainty` is bf16 + UncertaintyAwareLoss.
#     - Eval uses bf16 default because arch_family=transformer.
#     - The dict output is unwrapped at the eval boundary (DictToSegAdapter
#       in evaluation/_core.py) so sliding-window / TTA / MC dropout see a
#       plain seg tensor. Variance is dropped at eval — the AURC / Spearman
#       diagnostics come from predictive entropy + TTA-variance + MC-dropout
#       as in earlier phases.
# ---------------------------------------------------------------------------
if [ "$SKIP_UNCERTAINTY" = "0" ]; then
  echo
  echo "--- [3/5] train uncertainty ($EPOCHS epochs, bf16 preset, λ_unc=0.05) ---"
  python src/training/train_variant.py --variant uncertainty \
      --epochs "$EPOCHS" --exp-name "$EXP_NAME"
  echo
  echo "--- [4/5] eval uncertainty ---"
  python src/evaluation/evaluate_variant.py --variant uncertainty

  # Verify AURC + Spearman populated in evaluation_meta.json.
  UNC_DIR=$(ls -1d results/uncertainty/eval_* 2>/dev/null | sort | tail -1)
  if [ -n "$UNC_DIR" ] && [ -f "$UNC_DIR/evaluation_meta.json" ]; then
    echo
    echo "--- evaluation_meta.json: AURC + Spearman(unc, error) ---"
    python - <<PY
import json
with open("$UNC_DIR/evaluation_meta.json") as f:
    m = json.load(f)
aurc = m.get("aurc") or {}
spear = m.get("spearman_unc_error") or {}
print("AURC by mode (lower = better risk-coverage):")
for k, v in aurc.items():
    print(f"  {k:>12s} : {v}")
print("\nSpearman(unc, error) by mode (higher = uncertainty tracks error):")
for k, v in spear.items():
    print(f"  {k:>12s} : {v}")
if not aurc:
    print("[warn] AURC empty — check uncertainty.py mc_dropout / tta paths.")
if not spear:
    print("[warn] Spearman empty — same diagnosis.")
PY
  fi
fi

# ---------------------------------------------------------------------------
# 5. Paired Wilcoxon: uncertainty vs spectral_swin.
# ---------------------------------------------------------------------------
if [ "$SKIP_COMPARE" = "0" ]; then
  echo
  echo "--- [5/5] paired Wilcoxon (TTA + postprocess) ---"
  UC=$(ls -1d results/uncertainty/eval_*  2>/dev/null | sort | tail -1)
  SS=$(ls -1d results/spectral_swin/eval_* 2>/dev/null | sort | tail -1)
  if [ -z "$UC" ] || [ -z "$SS" ]; then
    echo "[error] missing eval folder. uncertainty=$UC  spectral_swin=$SS"
    echo "[hint]  Phase 3 must have been run before Phase 4 compare."
    exit 1
  fi
  echo "uncertainty   eval : $UC"
  echo "spectral_swin eval : $SS"
  python -m evaluation.stats compare \
      "$SS/per_case_metrics.csv" \
      "$UC/per_case_metrics.csv" \
      --mode tta_post --label-a spectral_swin --label-b uncertainty
fi

echo
echo "================================================================"
echo "Phase 4 done."
echo "  uncertainty logs : logs/run_uncertainty_${EXP_NAME}_*/"
echo "  uncertainty eval : results/uncertainty/eval_*/"
echo "  complexity table : results/complexity.csv"
echo
echo "  Inspect learned uncertainty gate (alpha):"
echo "    python -c \"import torch, glob; sd = torch.load(sorted(glob.glob('logs/run_uncertainty_${EXP_NAME}_*/best_model.pth'))[-1], map_location='cpu'); print('alpha =', float(sd['uncertainty_block.alpha']))\""
echo
echo "  Phase-4 headline metric is AURC + Spearman(unc, error), not Dice."
echo "  Dice may not move materially vs spectral_swin — the win is calibration."
echo "================================================================"
echo
echo "Detach now with Ctrl-b d, or just close the terminal — tmux keeps it."
