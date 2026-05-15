#!/usr/bin/env bash
# Phase 9 - TRUE leave-one-out ablation from `full`.
#
# The registry chain base_cnn -> ... -> full is a *cumulative build-up* study:
# each variant adds one component on top of the previous, so every step's gain
# is confounded with its position in the chain. Phase 9 answers the question a
# thesis examiner actually asks: "in the FINISHED model, how much does each
# component contribute?" — by removing exactly ONE component from `full` and
# measuring the drop, with the training recipe held identical to `full`.
#
# Variants (registry entries; all trained with a full_preset-derived recipe):
#   full_no_cross_modal   - stems kept, cross-modal attention removed
#   full_no_freq          - frequency branch removed
#   full_no_uncertainty   - variance head + uncertainty-loss term removed
#   full_no_boundary      - boundary head + boundary-loss term removed
#   full_no_arch          - Phase-6 arch trio removed (deeper Swin / extra
#                           encoder depth / multi-scale fusion head)
#
# NOT ablated here (report via chain delta + footnote, not a fake clean number):
#   spectral_swin / modality_stems — the monotonic guards in trans_resunet.py
#   couple them to dependent heads (uncertainty/boundary need spectral_swin;
#   cross_modal needs modality_stems), and the aux heads structurally consume
#   spectral-swin features. They cannot be cleanly leave-one-out ablated.
#
# Pipeline (per variant):
#   1. Sanity (CUDA): build, assert the removed flag is off & dependents on,
#      assert the conditional forward dict has exactly the expected keys, then
#      run the EXACT trainer loss path (get_preset -> criterion composition ->
#      _split_model_output dispatch) forward+backward; assert finite grads.
#   2. Complexity profile -> results/complexity.csv.
#   3. Train with full's recipe (TRAINING_PRESETS resolves full_no_* -> a
#      full_preset-derived factory; top-K=5 EMA snapshots, bf16, 300 ep).
#   4. Eval with `full`'s exact protocol: snapshot ensemble + 32-view extended
#      TTA + per-region V_min sweep + sliding-window overlap 0.625.
#   5. Paired Wilcoxon vs `full` (tta_post). Each-vs-full is a paired test on
#      the same val cases. Bonferroni n = number of variants run.
#
# Requires `full` already trained AND evaluated:
#   results/full/eval_*/per_case_metrics.csv  (run scripts/phase6.sh first)
#
# Auto-launches into a tmux session "phase9". Detach: Ctrl-b d.
# Reattach: tmux attach -t phase9. Kill: tmux kill-session -t phase9.
#
# Usage:
#   bash scripts/phase9_ablation.sh                       # all 5, 300 ep
#   EPOCHS=20 WARMUP=2 bash scripts/phase9_ablation.sh    # short dev cycle
#   VARIANTS="full_no_freq full_no_arch" bash scripts/phase9_ablation.sh
#   SKIP_TRAIN=1 SKIP_EVAL=1 bash scripts/phase9_ablation.sh   # compare only
#
# Skip flags (=1): SKIP_SANITY, SKIP_COMPLEXITY, SKIP_TRAIN, SKIP_EVAL,
#                  SKIP_COMPARE

set -euo pipefail

SESSION="${SESSION:-phase9}"

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

  echo "[info] launching Phase 9 in tmux session '$SESSION'."
  echo "       detach: Ctrl-b then d.   reattach: tmux attach -t $SESSION"

  tmux new-session -d -s "$SESSION" -c "$REPO_ABS" \
       -e "EPOCHS=${EPOCHS:-300}" \
       -e "WARMUP=${WARMUP:-10}" \
       -e "EXP_NAME=${EXP_NAME:-phase9}" \
       -e "VARIANTS=${VARIANTS:-}" \
       -e "SKIP_SANITY=${SKIP_SANITY:-0}" \
       -e "SKIP_COMPLEXITY=${SKIP_COMPLEXITY:-0}" \
       -e "SKIP_TRAIN=${SKIP_TRAIN:-0}" \
       -e "SKIP_EVAL=${SKIP_EVAL:-0}" \
       -e "SKIP_COMPARE=${SKIP_COMPARE:-0}" \
       "bash '$SCRIPT_ABS' $*; ec=\$?; echo; echo \"*** phase9_ablation.sh exited with code \$ec. Detach: Ctrl-b d. Close: Ctrl-d.\"; exec bash"

  exec tmux attach -t "$SESSION"
fi

# ---------------------------------------------------------------------------
# Inside tmux from here on.
# ---------------------------------------------------------------------------
EPOCHS="${EPOCHS:-300}"
WARMUP="${WARMUP:-10}"
EXP_NAME="${EXP_NAME:-phase9}"
SKIP_SANITY="${SKIP_SANITY:-0}"
SKIP_COMPLEXITY="${SKIP_COMPLEXITY:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_COMPARE="${SKIP_COMPARE:-0}"

# Default: all 5 leave-one-out variants. Override with VARIANTS="a b".
DEFAULT_VARIANTS="full_no_cross_modal full_no_freq full_no_uncertainty full_no_boundary full_no_arch"
VARIANTS="${VARIANTS:-$DEFAULT_VARIANTS}"
read -r -a VARIANT_ARR <<< "$VARIANTS"
N_VARIANTS=${#VARIANT_ARR[@]}

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "================================================================"
echo "Phase 9 - TRUE leave-one-out ablation from \`full\`"
echo "  variants : ${VARIANT_ARR[*]}  (n=$N_VARIANTS)"
echo "  epochs   : $EPOCHS   warmup: $WARMUP   exp: $EXP_NAME"
echo "  skip     : sanity=$SKIP_SANITY complexity=$SKIP_COMPLEXITY"
echo "             train=$SKIP_TRAIN eval=$SKIP_EVAL compare=$SKIP_COMPARE"
echo "================================================================"

# `full` baseline must already exist for the comparison step.
FULL_EVAL=$(ls -1d results/full/eval_* 2>/dev/null | sort | tail -1 || true)
if [ "$SKIP_COMPARE" = "0" ] && [ -z "$FULL_EVAL" ]; then
  echo "[error] no results/full/eval_*/ found. Run scripts/phase6.sh first —"
  echo "        Phase 9 compares every leave-one-out variant against \`full\`."
  exit 1
fi
[ -n "$FULL_EVAL" ] && echo "[info] \`full\` baseline eval: $FULL_EVAL"

pip install -q fvcore >/dev/null 2>&1 || echo "[warn] fvcore unavailable; FLOPs will be NaN"

# ---------------------------------------------------------------------------
# 1. Sanity: per variant, exercise the EXACT trainer loss path.
# ---------------------------------------------------------------------------
if [ "$SKIP_SANITY" = "0" ]; then
  echo
  echo "--- [1/5] sanity: build + conditional-dict shape + trainer loss path ---"
  VARIANTS="$VARIANTS" python - <<'PY'
import os
import torch
from model.registry import build_variant, get_output_mode
from training.losses import (
    RegionWiseDiceFocalLoss, UncertaintyAwareLoss, BoundaryAwareLoss,
)
from training.train_variant import get_preset, _split_model_output

assert torch.cuda.is_available(), "CUDA not available - abort."

# Expected model flags + forward-dict keys per variant. The forward dict has a
# key only for an *active* aux head: variance<-use_uncertainty, boundary<-use_boundary.
EXPECT = {
    "full_no_cross_modal": dict(off="use_cross_modal",
                                keys=["boundary", "seg", "variance"]),
    "full_no_freq":        dict(off="use_freq",
                                keys=["boundary", "seg", "variance"]),
    "full_no_uncertainty": dict(off="use_uncertainty",
                                keys=["boundary", "seg"]),
    "full_no_boundary":    dict(off="use_boundary",
                                keys=["seg", "variance"]),
    "full_no_arch":        dict(off=None,
                                keys=["boundary", "seg", "variance"]),
}

variants = os.environ["VARIANTS"].split()
x = torch.randn(1, 5, 128, 128, 128, device="cuda")
target = torch.randint(0, 4, (1, 128, 128, 128), dtype=torch.long, device="cuda")

for v in variants:
    spec = EXPECT[v]
    m = build_variant(v).cuda()
    n_p = sum(p.numel() for p in m.parameters())

    # (a) the removed flag is OFF; spectral_swin (the dependency anchor) still ON.
    if spec["off"] is not None:
        assert getattr(m, spec["off"]) is False, \
            f"{v}: {spec['off']} must be False"
    assert m.use_spectral_swin is True, f"{v}: use_spectral_swin must stay True"
    if v == "full_no_arch":
        assert m.spectral_blocks_per_stage == 2 and not m.encoder_extra_depth \
            and not m.use_multiscale_fusion_head, f"{v}: arch trio must be off"
    else:
        assert m.spectral_blocks_per_stage == 4 and m.encoder_extra_depth \
            and m.use_multiscale_fusion_head, f"{v}: arch trio must stay on"

    # (b) conditional forward dict has EXACTLY the expected keys.
    m.train()
    out = m(x)
    assert isinstance(out, dict), f"{v}: aux head on -> forward must return dict"
    assert sorted(out.keys()) == spec["keys"], \
        f"{v}: dict keys {sorted(out.keys())} != expected {spec['keys']}"

    # (c) reconstruct the criterion EXACTLY as train_variant.main() does, then
    #     run the exact dispatch from train_one_epoch. This is the real test:
    #     full_no_uncertainty (no variance) and full_no_boundary (no boundary)
    #     are the only new loss combos and must not crash.
    preset = get_preset(v)
    seg_loss = RegionWiseDiceFocalLoss(
        gamma=2.0, ce_weight=preset.ce_weight, class_weights=preset.class_weights)
    base = seg_loss
    if preset.use_uncertainty_loss:
        base = UncertaintyAwareLoss(
            seg_loss=base, lambda_unc=preset.lambda_unc,
            target_unc_at_high_dice=preset.target_unc_at_high_dice)
    if preset.use_boundary_loss:
        crit = BoundaryAwareLoss(
            base_loss=base, lambda_boundary=preset.lambda_boundary,
            bce_weight=preset.boundary_bce_weight,
            edge_dice_weight=preset.boundary_edge_dice_weight,
            lambda_boundary_start=preset.lambda_boundary_start,
            ramp_epochs=preset.lambda_boundary_ramp_epochs)
        crit.set_lambda(crit.lambda_at_epoch(0))
    else:
        crit = base

    seg_out, variance, boundary = _split_model_output(out)
    if isinstance(crit, BoundaryAwareLoss):
        loss = crit(seg_out, target, variance=variance, boundary=boundary)
    elif isinstance(crit, UncertaintyAwareLoss):
        loss = crit(seg_out, target, variance=variance)
    else:
        loss = crit(seg_out, target)
    assert torch.isfinite(loss), f"{v}: non-finite loss"
    loss.backward()
    bad = [nm for nm, p in m.named_parameters()
           if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not bad, f"{v}: {len(bad)} params missing/non-finite grad: {bad[:5]}"

    print(f"  {v:<22} ok  params={n_p:,}  preset_loss="
          f"{type(crit).__name__}({type(getattr(crit,'base_loss',crit)).__name__})"
          f"  keys={sorted(out.keys())}  loss={loss.item():.4f}")
    del m, out, loss
    torch.cuda.empty_cache()

print("\n=== sanity green: all variants build, shape-match, and train ===")
PY
fi

# ---------------------------------------------------------------------------
# 2-4. Per variant: complexity -> train -> eval (full's exact protocol).
# ---------------------------------------------------------------------------
mkdir -p results
for V in "${VARIANT_ARR[@]}"; do
  echo
  echo "================================================================"
  echo " variant: $V"
  echo "================================================================"

  if [ "$SKIP_COMPLEXITY" = "0" ]; then
    echo "--- [2/5] $V complexity profile ---"
    python -m evaluation.complexity --variant "$V" --device cuda \
        --out results/complexity.csv
  fi

  if [ "$SKIP_TRAIN" = "0" ]; then
    echo "--- [3/5] train $V ($EPOCHS ep, warmup=$WARMUP, full recipe, top-K=5) ---"
    python src/training/train_variant.py --variant "$V" \
        --epochs "$EPOCHS" --warmup "$WARMUP" --exp-name "$EXP_NAME"
  fi

  if [ "$SKIP_EVAL" = "0" ]; then
    echo "--- [4/5] eval $V (snapshot ensemble + extended TTA + V_min + ov0.625) ---"
    RUN_DIR=$(ls -1d logs/run_${V}_${EXP_NAME}_* 2>/dev/null | sort | tail -1 || true)
    if [ -z "$RUN_DIR" ]; then
      echo "[error] no logs/run_${V}_${EXP_NAME}_* — did training run?"
      exit 1
    fi
    echo "[info] $V snapshots from $RUN_DIR"
    python src/evaluation/evaluate_variant.py --variant "$V" \
        --ensemble-ckpts "${RUN_DIR}/snapshot_top*.pth" \
        --tta-extended --vmin-sweep --overlap 0.625 \
        --run-name "eval_${EXP_NAME}"
  fi
done

# ---------------------------------------------------------------------------
# 5. Paired Wilcoxon: each leave-one-out variant vs `full` (tta_post).
# ---------------------------------------------------------------------------
if [ "$SKIP_COMPARE" = "0" ]; then
  echo
  echo "--- [5/5] paired Wilcoxon: <variant> vs full (tta_post) ---"
  FD=$(ls -1d results/full/eval_* 2>/dev/null | sort | tail -1)
  echo "full baseline eval : $FD"
  echo "Bonferroni: $N_VARIANTS comparisons -> use alpha/$N_VARIANTS (or"
  echo "  \`python -m evaluation.aggregate_final\`, which Bonferroni-corrects)."
  for V in "${VARIANT_ARR[@]}"; do
    VD=$(ls -1d results/${V}/eval_* 2>/dev/null | sort | tail -1 || true)
    if [ -z "$VD" ]; then
      echo "[warn] no results/${V}/eval_* — skipping compare for $V"
      continue
    fi
    echo
    echo ">>> full  vs  $V   (negative delta => removing the component HURTS) <<<"
    python -m evaluation.stats compare \
        "$FD/per_case_metrics.csv" \
        "$VD/per_case_metrics.csv" \
        --mode tta_post --label-a full --label-b "$V"
  done
fi

echo
echo "================================================================"
echo "Phase 9 done."
echo "  variant logs : logs/run_<variant>_${EXP_NAME}_*/"
echo "  variant eval : results/<variant>/eval_${EXP_NAME}/"
echo "  complexity   : results/complexity.csv"
echo
echo "  Each row's Dice/HD95 DROP vs \`full\` is that component's marginal"
echo "  contribution in the finished model. A significant drop (paired"
echo "  Wilcoxon, Bonferroni-corrected over $N_VARIANTS tests) is the thesis"
echo "  evidence that the component matters. spectral_swin / modality_stems"
echo "  are reported via the cumulative chain delta + the coupling footnote."
echo "================================================================"
echo
echo "Detach now with Ctrl-b d, or just close the terminal - tmux keeps it."
