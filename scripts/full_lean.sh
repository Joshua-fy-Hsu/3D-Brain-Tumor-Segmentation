#!/usr/bin/env bash
# full_lean - single-run debloat of `full` to beat the unet3d baseline.
#
# WHY: the whole AURAS ablation chain never beats plain base_cnn on TC; the
# entire deficit vs unet3d is TC (tta_post: full 0.781 vs unet3d 0.838).
# freq/uncertainty are TC-poison, boundary only patches what uncertainty
# broke, and all of full's mean gain is ET-only. full_lean keeps only the
# proven winners (spectral_swin + Phase-6 arch) + the monotonic prerequisite
# (modality_stems + cross_modal), drops freq/uncertainty/boundary, adds a
# dedicated gated TC-Refine pathway + TC-weighted loss (TC region x1.8), and
# trains 250 ep with heavier spatial aug. Uncertainty novelty is preserved
# training-free via MC-Dropout (decoder_dropout_final=0.05) + TTA + ECE.
#
# Pipeline:
#   1. Sanity (CUDA): forward+backward at 128^3; train->dict has 'tc';
#      eval->plain tensor (legacy contract); gate.grad finite; tc params
#      routed into an optimizer group; region_weights=(1,1,1) is a no-op.
#   2. (optional) Complexity profile -> results/complexity.csv.
#   3. Train full_lean (full_lean_preset: bf16, TCRefineLoss, TC x1.8,
#      gate warmup, single best model - NO ensemble).
#   4. Eval full_lean with the joint postprocess sweep (tau x et_vmin x
#      tc_vmin).
#   5. Eval unet3d with the SAME joint sweep (fairness - each model gets its
#      own val-tuned best). Skipped if results/unet3d/eval_lean_cmp exists.
#   6. summarize_vmin_sweep for both (shows each model's best operating
#      point) + paired Wilcoxon full_lean vs unet3d on tta_post.
#
# GOAL GATE: full_lean mean Dice > unet3d (each at its own val-tuned best),
# ideally TC >= ~0.82, Wilcoxon not significantly worse on any region.
#
# Auto-launches into a tmux session "full_lean". Detach: Ctrl-b d.
# Reattach: tmux attach -t full_lean. Kill: tmux kill-session -t full_lean.
#
# Usage:
#   SMOKE=1 bash scripts/full_lean.sh        # sanity + 5-ep smoke, then STOP
#   bash scripts/full_lean.sh                # full pipeline (250 ep + evals)
#   EPOCHS=200 EXP_NAME=lean2 bash scripts/full_lean.sh
#
# Skip flags (set =1):
#   SKIP_SANITY, SKIP_COMPLEXITY, SKIP_TRAIN, SKIP_EVAL,
#   SKIP_BASELINE_EVAL (reuse existing unet3d sweep eval),
#   SKIP_COMPARE
# Other env:
#   EPOCHS (250)  WARMUP (5)  TC_WARMUP (20)  EXP_NAME (lean)
#   SMOKE=1 -> sanity + 5-epoch dry run only (the ~20-min pre-check)
#   FORCE_BASELINE_EVAL=1 -> re-run unet3d sweep even if it exists
#
# Pre-reqs: PyTorch + project reqs; BraTS preprocessed; config.py
# TRAIN_DATA_PATH set; one bf16-capable CUDA GPU; unet3d checkpoint present
# under logs/run_unet3d_*/best_model.pth; tmux (auto-installed on Debian).

set -euo pipefail

SESSION="${SESSION:-full_lean}"

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

  echo "[info] launching full_lean in tmux session '$SESSION'."
  echo "       detach: Ctrl-b then d.   reattach: tmux attach -t $SESSION"

  tmux new-session -d -s "$SESSION" -c "$REPO_ABS" \
       -e "EPOCHS=${EPOCHS:-250}" \
       -e "WARMUP=${WARMUP:-5}" \
       -e "TC_WARMUP=${TC_WARMUP:-20}" \
       -e "EXP_NAME=${EXP_NAME:-lean}" \
       -e "SMOKE=${SMOKE:-0}" \
       -e "SKIP_SANITY=${SKIP_SANITY:-0}" \
       -e "SKIP_COMPLEXITY=${SKIP_COMPLEXITY:-0}" \
       -e "SKIP_TRAIN=${SKIP_TRAIN:-0}" \
       -e "SKIP_EVAL=${SKIP_EVAL:-0}" \
       -e "SKIP_BASELINE_EVAL=${SKIP_BASELINE_EVAL:-0}" \
       -e "FORCE_BASELINE_EVAL=${FORCE_BASELINE_EVAL:-0}" \
       -e "SKIP_COMPARE=${SKIP_COMPARE:-0}" \
       "bash '$SCRIPT_ABS' $*; ec=\$?; echo; echo \"*** full_lean.sh exited with code \$ec. Detach: Ctrl-b d. Close: Ctrl-d.\"; exec bash"

  exec tmux attach -t "$SESSION"
fi

# ---------------------------------------------------------------------------
# Inside tmux from here on.
# ---------------------------------------------------------------------------
EPOCHS="${EPOCHS:-250}"
WARMUP="${WARMUP:-5}"
TC_WARMUP="${TC_WARMUP:-20}"
EXP_NAME="${EXP_NAME:-lean}"
SMOKE="${SMOKE:-0}"
SKIP_SANITY="${SKIP_SANITY:-0}"
SKIP_COMPLEXITY="${SKIP_COMPLEXITY:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_BASELINE_EVAL="${SKIP_BASELINE_EVAL:-0}"
FORCE_BASELINE_EVAL="${FORCE_BASELINE_EVAL:-0}"
SKIP_COMPARE="${SKIP_COMPARE:-0}"

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "================================================================"
echo "full_lean - single-run debloat to beat unet3d"
echo "  epochs    : $EPOCHS   warmup: $WARMUP   tc-warmup: $TC_WARMUP"
echo "  exp name  : $EXP_NAME"
echo "  smoke     : $SMOKE  (1 = sanity + 5ep dry-run then stop)"
echo "  skip      : sanity=$SKIP_SANITY complexity=$SKIP_COMPLEXITY"
echo "              train=$SKIP_TRAIN eval=$SKIP_EVAL"
echo "              baseline_eval=$SKIP_BASELINE_EVAL compare=$SKIP_COMPARE"
echo "================================================================"

# ---------------------------------------------------------------------------
# 1. Sanity check (CUDA).
# ---------------------------------------------------------------------------
if [ "$SKIP_SANITY" = "0" ]; then
  echo
  echo "--- [1] sanity: forward+backward 128^3 + contracts + no-op checks ---"
  python - <<'PY'
import torch
from model.registry import build_variant
from training.losses import RegionWiseDiceFocalLoss, TCRefineLoss

assert torch.cuda.is_available(), "CUDA not available - abort."

m = build_variant("full_lean").cuda()
n = sum(p.numel() for p in m.parameters())
print(f"full_lean params: {n:,}")
assert m.tc_refine is not None, "use_tc_refine must build a TCRefineHead"
assert m.fusion_head is not None and m.final_conv is None, "fusion head must be live"
assert not m.use_uncertainty and not m.use_boundary, "uncertainty/boundary must be OFF"
assert m.freq_block is None, "freq must be OFF"
assert len(m.spectral_swin_stage.stage1) == 4, "spectral_blocks_per_stage=4 expected"

# region_weights=(1,1,1) is a byte-identical no-op vs no arg.
torch.manual_seed(0)
lg = [torch.randn(2,4,8,8,8), torch.randn(2,4,4,4,4), torch.randn(2,4,2,2,2)]
tg = torch.randint(0,4,(2,8,8,8))
a = RegionWiseDiceFocalLoss(ce_weight=0.3, class_weights=(0.1,2,1,1))
b = RegionWiseDiceFocalLoss(ce_weight=0.3, class_weights=(0.1,2,1,1),
                            region_weights=(1,1,1))
assert torch.allclose(a(lg,tg), b(lg,tg)), "region_weights=(1,1,1) must be a no-op"
print("region_weights (1,1,1) no-op: OK")

# Train: dict with 'tc'; backward; gate.grad finite.
x = torch.randn(1,5,128,128,128, device="cuda")
y = torch.randint(0,4,(1,128,128,128), device="cuda")
m.train()
out = m(x)
assert isinstance(out, dict) and out.get("tc") is not None, "train must surface tc"
assert out["variance"] is None and out["boundary"] is None
seg = out["seg"]; tc1, tc2 = out["tc"]
assert tuple(seg[0].shape) == (1,4,128,128,128)
assert tuple(tc1.shape) == (1,1,128,128,128) and tuple(tc2.shape) == (1,1,64,64,64)
crit = TCRefineLoss(RegionWiseDiceFocalLoss(gamma=2.0, ce_weight=0.3,
                    class_weights=(0.1,2,1,1), region_weights=(1.0,1.8,1.0)),
                    ds_weights=(1.0,0.5))
loss = crit(seg, y, tc=(tc1, tc2))
loss.backward()
g = m.tc_refine.gate.grad
assert g is not None and torch.isfinite(g).all(), "gate.grad must be finite"
assert torch.isfinite(loss).item()
print(f"train dict+TCRefineLoss OK | loss={float(loss):.4f} gate.grad={float(g):.4f}")

# Eval: plain (1,4,128^3) tensor - legacy contract, _core needs no change.
for p in m.parameters(): p.grad = None
m.eval()
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    ev = m(x)
assert torch.is_tensor(ev) and tuple(ev.shape) == (1,4,128,128,128), \
    f"eval must return a plain (1,4,128^3) tensor, got {type(ev)}"
assert torch.isfinite(ev).all()
print("eval plain-tensor contract + bf16: OK")
print("\n=== sanity green ===")
PY
fi

# ---------------------------------------------------------------------------
# SMOKE: sanity + 5-epoch dry-run, then STOP for inspection.
# ---------------------------------------------------------------------------
if [ "$SMOKE" = "1" ]; then
  echo
  echo "--- SMOKE: 5-epoch dry run (exp-name=${EXP_NAME}_smoke) ---"
  echo "    check: bf16 no NaN, [tc-warmup] log prints, loss decreases."
  python src/training/train_variant.py --variant full_lean \
      --epochs 5 --warmup 2 --tc-warmup 5 --exp-name "${EXP_NAME}_smoke"
  echo
  echo "================================================================"
  echo "SMOKE done. Inspect logs/run_full_lean_${EXP_NAME}_smoke_*/training_log.csv"
  echo "If healthy, run the full pipeline:  bash scripts/full_lean.sh"
  echo "================================================================"
  exit 0
fi

# ---------------------------------------------------------------------------
# 2. Complexity profile (cheap; supports the 'lighter model' narrative).
# ---------------------------------------------------------------------------
if [ "$SKIP_COMPLEXITY" = "0" ]; then
  echo
  echo "--- [2] complexity profile ---"
  mkdir -p results
  python -m evaluation.complexity --variant full_lean --device cuda \
      --out results/complexity.csv || echo "[warn] complexity profile failed; continuing"
fi

# ---------------------------------------------------------------------------
# 3. Train full_lean (single best model - no ensemble).
# ---------------------------------------------------------------------------
if [ "$SKIP_TRAIN" = "0" ]; then
  echo
  echo "--- [3] train full_lean ($EPOCHS ep, warmup=$WARMUP, tc-warmup=$TC_WARMUP) ---"
  python src/training/train_variant.py --variant full_lean \
      --epochs "$EPOCHS" --warmup "$WARMUP" --tc-warmup "$TC_WARMUP" \
      --exp-name "$EXP_NAME"
fi

# ---------------------------------------------------------------------------
# 4. Eval full_lean with the joint postprocess sweep.
# ---------------------------------------------------------------------------
if [ "$SKIP_EVAL" = "0" ]; then
  echo
  echo "--- [4] eval full_lean (joint tau x et_vmin x tc_vmin sweep) ---"
  python src/evaluation/evaluate_variant.py --variant full_lean \
      --vmin-sweep --run-name "eval_${EXP_NAME}"
fi

# ---------------------------------------------------------------------------
# 5. Eval unet3d with the SAME joint sweep (fairness).
#    Reused if results/unet3d/eval_lean_cmp already has the joint sweep.
# ---------------------------------------------------------------------------
UNET_CMP_DIR="results/unet3d/eval_lean_cmp"
if [ "$SKIP_BASELINE_EVAL" = "0" ]; then
  if [ -f "$UNET_CMP_DIR/vmin_sweep.csv" ] && [ "$FORCE_BASELINE_EVAL" = "0" ]; then
    echo
    echo "--- [5] unet3d joint-sweep eval already exists ($UNET_CMP_DIR) - reuse ---"
    echo "        (FORCE_BASELINE_EVAL=1 to re-run)"
  else
    echo
    echo "--- [5] eval unet3d (SAME joint sweep, fair head-to-head) ---"
    python src/evaluation/evaluate_variant.py --variant unet3d \
        --vmin-sweep --run-name "eval_lean_cmp"
  fi
fi

# ---------------------------------------------------------------------------
# 6. Per-model best operating point + paired Wilcoxon (the goal gate).
# ---------------------------------------------------------------------------
if [ "$SKIP_COMPARE" = "0" ]; then
  echo
  echo "--- [6] per-model val-tuned best + paired Wilcoxon ---"
  LEAN_DIR=$(ls -1d results/full_lean/eval_* 2>/dev/null | sort | tail -1)
  UNET_DIR=$(ls -1d results/unet3d/eval_lean_cmp 2>/dev/null | sort | tail -1)
  [ -z "$UNET_DIR" ] && UNET_DIR=$(ls -1d results/unet3d/eval_* 2>/dev/null | sort | tail -1)
  if [ -z "$LEAN_DIR" ] || [ -z "$UNET_DIR" ]; then
    echo "[error] missing eval dir. full_lean=$LEAN_DIR  unet3d=$UNET_DIR"
    exit 1
  fi
  echo "full_lean eval : $LEAN_DIR"
  echo "unet3d    eval : $UNET_DIR"
  echo
  echo ">>> full_lean val-tuned best (tau, et_vmin, tc_vmin) per mode <<<"
  python src/evaluation/summarize_vmin_sweep.py "$LEAN_DIR" || true
  echo
  echo ">>> unet3d val-tuned best (tau, et_vmin, tc_vmin) per mode <<<"
  python src/evaluation/summarize_vmin_sweep.py "$UNET_DIR" || true
  echo
  echo ">>> paired Wilcoxon: full_lean vs unet3d (tta_post, default postprocess) <<<"
  echo "    (conservative honest number; the val-tuned best per model is the"
  echo "     summaries above - disclose as val-tuned, no separate test set.)"
  python -m evaluation.stats compare \
      "$UNET_DIR/per_case_metrics.csv" \
      "$LEAN_DIR/per_case_metrics.csv" \
      --mode tta_post --label-a unet3d --label-b full_lean
fi

echo
echo "================================================================"
echo "full_lean done."
echo "  logs        : logs/run_full_lean_${EXP_NAME}_*/"
echo "  eval        : results/full_lean/eval_${EXP_NAME}/"
echo "  unet3d cmp  : $UNET_CMP_DIR/"
echo "  complexity  : results/complexity.csv"
echo
echo "  GOAL: full_lean mean Dice > unet3d (each at its own val-tuned best,"
echo "  see the two summarize_vmin_sweep blocks), ideally TC >= ~0.82, and"
echo "  Wilcoxon not significantly worse on any region."
echo "  Fallback if missed: segresnet/unet3d win raw Dice; AURAS pitch ="
echo "  comparable Dice + training-free uncertainty (MC-Dropout/TTA/ECE) +"
echo "  a lighter model (see results/complexity.csv)."
echo "================================================================"
echo
echo "Detach now with Ctrl-b d, or just close the terminal - tmux keeps it."
