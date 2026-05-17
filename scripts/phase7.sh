#!/usr/bin/env bash
# Phase 7 - external baselines: SwinUNETR / SegResNet / 3D-UNet, run in their
# standard published config, compared against AURAS (`full`).
#
# Unlike Phases 1-6 (each adds one ablation flag and must *beat* the previous
# champion to pass its gate), Phase 7 is the reverse: these are well-known
# reference models. The "gate" is that AURAS (`full`) **significantly beats**
# each of them on tta_post mean Dice (paired Wilcoxon, p<0.05), and that our
# `base_cnn` is no worse than the vanilla `unet3d` floor (a recipe sanity:
# same family, so any gap is data pipeline / training recipe, not architecture).
#
# Baselines (registry keys -> standard config, wrappers in
# src/model/baselines/monai_baselines.py):
#   swinunetr : Swin UNETR, feature_size=48   (transformer, BraTS SOTA)   bf16
#   segresnet : SegResNet, init_filters=32    (CNN, BraTS-2018 winner)    fp16
#   unet3d    : MONAI BasicUNet, 32..512      (vanilla 3D U-Net, floor)   fp16
#
# Forward contract is identical to the AURAS family (train -> 1-tuple so the
# deep-supervision loss consumes it with no DS term; eval -> bare tensor),
# so the generic train_variant.py / evaluate_variant.py drive them unchanged.
#
# Training preset per baseline (recipe-fair, not "best possible"):
#   swinunetr -> transformer preset + FORCED bf16 (transformer_preset itself
#                defaults to fp16; Swin windowed-attention softmax overflows
#                in fp16, and AURAS/full trained in bf16 - so we align it).
#   segresnet -> base_cnn preset  (fp16 + GradScaler, val_loss criterion) -
#                this is the EXACT recipe `base_cnn` was trained with.
#   unet3d    -> base_cnn preset
# Eval AMP dtype is auto from each variant's arch_family (transformer->bf16,
# cnn->fp16) - no override needed.
#
# RECIPE ALIGNMENT (read before quoting any number):
#   The comparison vs AURAS is reported TWO ways, because results/full/eval_*
#   from Phase 6 is ensemble-of-5 + 32-view TTA - not a like-for-like eval:
#     - [system]            AURAS full recipe + ensemble + extended TTA
#                           vs baseline standard config. The "is the deployed
#                           system better" claim. Standard in the literature.
#     - [recipe-controlled] AURAS evaluated with a SINGLE best_model.pth +
#                           standard 8-way TTA (this script runs that eval)
#                           vs baseline standard config. Removes the
#                           ensemble/extended-TTA confound. The residual gap
#                           is loss (Boundary/Uncertainty heads) + deep
#                           supervision, both architecture-intrinsic to AURAS
#                           and unavailable to a plain baseline - state that.
#
# Pipeline (looped over each baseline):
#   1. Sanity (CUDA): build from registry; fp32 forward+backward at 128^3;
#      train-mode output is a 1-tuple of 4-ch logits with finite grads;
#      eval-mode output is a bare (1,4,128,128,128) tensor; autocast eval
#      forward (per arch_family) is finite.
#   2. Complexity profile -> appended to results/complexity.csv.
#   3. Train the baseline (its preset, $EPOCHS epochs; swinunetr forced bf16).
#   4. Eval the baseline (standard config: default modes + V_min sweep, no
#      ensemble / extended TTA). Then eval `full` ONCE single-checkpoint +
#      standard TTA -> results/full/eval_${EXP_NAME}_single (the matched arm).
#   5. Paired Wilcoxon, TWO comparisons per baseline: [system] vs the newest
#      ensemble results/full/eval_*, and [recipe-controlled] vs the single-
#      checkpoint full eval. Plus `base_cnn` vs `unet3d` once (recipe floor).
#
# Auto-launches into a tmux session named "phase7". Detach: Ctrl-b d.
# Reattach: tmux attach -t phase7. Kill: tmux kill-session -t phase7.
#
# Usage:
#   bash scripts/phase7.sh                                   # all 3, 300 ep
#   BASELINES="segresnet unet3d" bash scripts/phase7.sh      # subset
#   EPOCHS=20 WARMUP=2 bash scripts/phase7.sh                # short dev cycle
#   SKIP_TRAIN=1 SKIP_EVAL=1 bash scripts/phase7.sh          # just re-compare
#
# Skip flags (set =1):
#   SKIP_SANITY, SKIP_COMPLEXITY, SKIP_TRAIN, SKIP_EVAL, SKIP_COMPARE
#
# Pre-reqs on the pod:
#   - PyTorch + project requirements installed; MONAI present (this script
#     attempts `pip install monai` if it is missing - baselines need it).
#   - BraTS preprocessed; src/configs/config.py:TRAIN_DATA_PATH set
#   - Phase 6 already completed -> results/full/eval_*/per_case_metrics.csv
#     (the AURAS champion these baselines are measured against)
#   - Phase 0 base_cnn evaluated -> results/base_cnn/eval_*/ (floor check)
#   - One CUDA GPU visible (bf16-capable for swinunetr: Ampere+ / 4090 / A100)
#   - tmux available (script installs it on Debian/Ubuntu pods if missing)

set -euo pipefail

SESSION="${SESSION:-phase7}"

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

  echo "[info] launching Phase 7 in tmux session '$SESSION'."
  echo "       detach: Ctrl-b then d.   reattach: tmux attach -t $SESSION"

  tmux new-session -d -s "$SESSION" -c "$REPO_ABS" \
       -e "BASELINES=${BASELINES:-swinunetr segresnet unet3d}" \
       -e "EPOCHS=${EPOCHS:-300}" \
       -e "WARMUP=${WARMUP:-5}" \
       -e "EXP_NAME=${EXP_NAME:-phase7}" \
       -e "SKIP_SANITY=${SKIP_SANITY:-0}" \
       -e "SKIP_COMPLEXITY=${SKIP_COMPLEXITY:-0}" \
       -e "SKIP_TRAIN=${SKIP_TRAIN:-0}" \
       -e "SKIP_EVAL=${SKIP_EVAL:-0}" \
       -e "SKIP_COMPARE=${SKIP_COMPARE:-0}" \
       "bash '$SCRIPT_ABS' $*; ec=\$?; echo; echo \"*** phase7.sh exited with code \$ec. Detach: Ctrl-b d. Close: Ctrl-d.\"; exec bash"

  exec tmux attach -t "$SESSION"
fi

# ---------------------------------------------------------------------------
# Inside tmux from here on.
# ---------------------------------------------------------------------------
BASELINES="${BASELINES:-swinunetr segresnet unet3d}"
EPOCHS="${EPOCHS:-300}"
WARMUP="${WARMUP:-5}"
EXP_NAME="${EXP_NAME:-phase7}"
SKIP_SANITY="${SKIP_SANITY:-0}"
SKIP_COMPLEXITY="${SKIP_COMPLEXITY:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_COMPARE="${SKIP_COMPARE:-0}"

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# Recipe-fair training preset per baseline (see header).
preset_for() {
  case "$1" in
    swinunetr) echo "transformer" ;;
    segresnet|unet3d) echo "base_cnn" ;;
    *) echo "auto" ;;
  esac
}

# AMP override per baseline. transformer_preset defaults to fp16, but Swin's
# windowed-attention softmax overflows in fp16 and AURAS/full trained in bf16
# - so swinunetr is forced to bf16 to stay aligned. Others use preset default.
amp_for() {
  case "$1" in
    swinunetr) echo "--amp-dtype bf16" ;;
    *) echo "" ;;
  esac
}

echo "================================================================"
echo "Phase 7 - external baselines vs AURAS"
echo "  baselines : $BASELINES"
echo "  epochs    : $EPOCHS   warmup : $WARMUP"
echo "  exp name  : $EXP_NAME"
echo "  skip      : sanity=$SKIP_SANITY complexity=$SKIP_COMPLEXITY"
echo "              train=$SKIP_TRAIN eval=$SKIP_EVAL compare=$SKIP_COMPARE"
echo "================================================================"

pip install -q fvcore >/dev/null 2>&1 || echo "[warn] fvcore unavailable; FLOPs will be NaN"
# MONAI is a hard dep of the baselines (lazy-imported inside the factory).
if ! python -c "import monai" >/dev/null 2>&1; then
  echo "[info] MONAI not found; attempting pip install monai"
  pip install -q monai >/dev/null 2>&1 \
    || { echo "[error] could not install MONAI. Baselines need it. Abort."; exit 1; }
fi
python -c "import monai; print('[info] MONAI', monai.__version__)"

# ---------------------------------------------------------------------------
# 1. Sanity check - per baseline: registry build + contract + grads.
# ---------------------------------------------------------------------------
if [ "$SKIP_SANITY" = "0" ]; then
  echo
  echo "--- [1/5] sanity: build + fwd/bwd at 128^3 + forward contract ---"
  for b in $BASELINES; do
    echo
    echo ">>> sanity: $b <<<"
    BVAR="$b" python - <<'PY'
import os
import torch
from model.registry import build_variant, get_arch_family, get_output_mode
from training.losses import RegionWiseDiceFocalLoss

b = os.environ["BVAR"]
assert torch.cuda.is_available(), "CUDA not available - abort."
assert get_output_mode(b) == "softmax", f"{b} must be a softmax-head baseline"
fam = get_arch_family(b)

m = build_variant(b).cuda()
n_p = sum(p.numel() for p in m.parameters())
print(f"{b}: {n_p:,} params, arch_family={fam}")

x = torch.randn(1, 5, 128, 128, 128, device="cuda")
target = torch.randint(0, 4, (1, 128, 128, 128), dtype=torch.long, device="cuda")

# ---- train-mode contract: 1-tuple of 4-ch logits, finite grads ----
m.train()
out = m(x)
assert isinstance(out, tuple) and len(out) == 1, \
    f"train output must be a 1-tuple, got {type(out)}"
assert tuple(out[0].shape) == (1, 4, 128, 128, 128), \
    f"logits must be (1,4,128,128,128), got {tuple(out[0].shape)}"
crit = RegionWiseDiceFocalLoss(gamma=2.0, ce_weight=0.3,
                               class_weights=(0.1, 2.0, 1.0, 1.0))
loss = crit(out, target)            # exact trainer path: criterion(seg_out, t)
loss.backward()
bad = [nm for nm, p in m.named_parameters()
       if p.grad is None or not torch.isfinite(p.grad).all()]
assert not bad, f"missing/non-finite grads on {len(bad)} params: {bad[:5]}"
print(f"  train: 1-tuple OK, loss={loss.item():.4f}, all grads finite")
for p in m.parameters():
    p.grad = None

# ---- eval-mode contract: bare tensor; autocast (per arch) is finite ----
m.eval()
with torch.no_grad():
    out_e = m(x)
assert torch.is_tensor(out_e) and tuple(out_e.shape) == (1, 4, 128, 128, 128), \
    f"eval output must be a (1,4,128,128,128) tensor, got {type(out_e)}"
amp_dt = torch.bfloat16 if fam == "transformer" else torch.float16
torch.cuda.reset_peak_memory_stats()
with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dt):
    out_a = m(x)
assert torch.isfinite(out_a).all(), f"{b}: non-finite logits under {amp_dt}"
print(f"  eval: tensor OK; {str(amp_dt).split('.')[-1]} autocast finite; "
      f"peak VRAM = {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print(f"=== {b} sanity green ===")
PY
  done
fi

# ---------------------------------------------------------------------------
# 2. Complexity profile (params / FLOPs / latency) - appended to one CSV.
# ---------------------------------------------------------------------------
if [ "$SKIP_COMPLEXITY" = "0" ]; then
  echo
  echo "--- [2/5] complexity profile ---"
  mkdir -p results
  for b in $BASELINES; do
    echo ">>> complexity: $b <<<"
    python -m evaluation.complexity --variant "$b" --device cuda \
        --out results/complexity.csv
  done
fi

# ---------------------------------------------------------------------------
# 3. Train each baseline (recipe-fair preset, standard config).
# ---------------------------------------------------------------------------
if [ "$SKIP_TRAIN" = "0" ]; then
  echo
  echo "--- [3/5] train baselines ($EPOCHS epochs, warmup=$WARMUP) ---"
  for b in $BASELINES; do
    P="$(preset_for "$b")"
    A="$(amp_for "$b")"
    echo
    echo ">>> train: $b (preset=$P ${A:+$A}) <<<"
    python src/training/train_variant.py --variant "$b" \
        --preset "$P" $A --epochs "$EPOCHS" --warmup "$WARMUP" \
        --exp-name "$EXP_NAME"
  done
fi

# ---------------------------------------------------------------------------
# 4. Eval each baseline (standard config: default modes + V_min sweep;
#    AMP dtype auto from arch_family; no ensemble / extended TTA).
# ---------------------------------------------------------------------------
if [ "$SKIP_EVAL" = "0" ]; then
  echo
  echo "--- [4/5] eval baselines ---"
  for b in $BASELINES; do
    echo
    echo ">>> eval: $b <<<"
    RUN_DIR=$(ls -1d logs/run_${b}_${EXP_NAME}_* 2>/dev/null | sort | tail -1)
    if [ -z "$RUN_DIR" ]; then
      echo "[error] could not find logs/run_${b}_${EXP_NAME}_* - did training run?"
      exit 1
    fi
    echo "[info] checkpoint from $RUN_DIR"
    python src/evaluation/evaluate_variant.py --variant "$b" \
        --checkpoint "${RUN_DIR}/best_model.pth" \
        --vmin-sweep --run-name "eval_${EXP_NAME}"

    BDIR="results/${b}/eval_${EXP_NAME}"
    if [ -f "$BDIR/summary.csv" ]; then
      echo
      echo "--- $b summary.csv (Dice / HD95 / NSD) ---"
      # `|| true`: this is a cosmetic summary print - a pandas hiccup here
      # must never abort the eval/compare pipeline (it did once: summary.csv's
      # label column is `Method`, not `mode`, so don't hardcode the name).
      python - <<PY || true
import pandas as pd
df = pd.read_csv("$BDIR/summary.csv")
cols = [df.columns[0]] + [c for c in df.columns if c.lower().startswith(("dice", "hd95", "nsd"))]
print(df[cols].to_string(index=False))
PY
    fi
  done

  # ---- recipe-controlled arm: eval AURAS (`full`) ONCE, single best_model
  #      checkpoint, standard 8-way TTA, NO ensemble / extended TTA - matched
  #      to the baselines' eval recipe. evaluate_variant.py auto-discovers the
  #      newest full-matching best_model.pth (state_dict matched) when
  #      --checkpoint is omitted. This is the apples-to-apples comparison row.
  echo
  echo ">>> eval: full (SINGLE ckpt, no ensemble/extended-TTA - matched arm) <<<"
  if ls logs/run_full_*/best_model.pth >/dev/null 2>&1; then
    python src/evaluation/evaluate_variant.py --variant full \
        --vmin-sweep --run-name "eval_${EXP_NAME}_single"
  else
    echo "[warn] no logs/run_full_*/best_model.pth found - skipping the"
    echo "       recipe-controlled full eval. Phase 6 must have trained \`full\`"
    echo "       first; without it only the [system] comparison can run."
  fi
fi

# ---------------------------------------------------------------------------
# 5. Paired Wilcoxon: AURAS (`full`) must beat each baseline (tta_post).
#    Plus a one-off recipe floor check: base_cnn vs unet3d.
# ---------------------------------------------------------------------------
if [ "$SKIP_COMPARE" = "0" ]; then
  echo
  echo "--- [5/5] paired Wilcoxon (TTA + postprocess) ---"
  # [system] arm: newest ensemble full eval (Phase 6) - exclude the *_single
  # dirs this script writes for the matched arm.
  # `ls -1dt` = newest-by-mtime first (NOT lexicographic `sort` - dir names
  # are now `eval_phaseN`, not timestamps, so string order != chronological;
  # this matches aggregate_final.latest_eval_dir / find_latest_checkpoint).
  # `|| true`: grep exits 1 when nothing matches; without it `set -e` +
  # `pipefail` would abort here before the graceful fallback below.
  FD_SYS=$(ls -1dt results/full/eval_* 2>/dev/null | grep -v '_single$' | head -1 || true)
  # [recipe-controlled] arm: the single-checkpoint full eval written in step 4.
  FD_CTRL="results/full/eval_${EXP_NAME}_single"
  [ -d "$FD_CTRL" ] || FD_CTRL=$(ls -1dt results/full/eval_*_single 2>/dev/null | head -1)
  if [ -z "$FD_SYS" ] && [ -z "${FD_CTRL:-}" ]; then
    echo "[error] no results/full/eval_* - Phase 6 must run before Phase 7 compare."
    exit 1
  fi
  echo "AURAS [system]           eval : ${FD_SYS:-<missing>}"
  echo "AURAS [recipe-controlled] eval : ${FD_CTRL:-<missing, run step 4>}"
  for b in $BASELINES; do
    BD="results/${b}/eval_${EXP_NAME}"
    [ -d "$BD" ] || BD=$(ls -1dt results/${b}/eval_* 2>/dev/null | grep -v '_single$' | head -1)
    if [ -z "${BD:-}" ] || [ ! -f "$BD/per_case_metrics.csv" ]; then
      echo "[warn] no eval folder for $b - skipping its comparison."
      continue
    fi
    if [ -n "$FD_SYS" ] && [ -f "$FD_SYS/per_case_metrics.csv" ]; then
      echo
      echo ">>> [system] AURAS (full, ensemble+32-TTA) vs $b  [gate: AURAS wins, p<0.05] <<<"
      python -m evaluation.stats compare \
          "$BD/per_case_metrics.csv" \
          "$FD_SYS/per_case_metrics.csv" \
          --mode tta_post --label-a "$b" --label-b AURAS
    fi
    if [ -n "${FD_CTRL:-}" ] && [ -f "$FD_CTRL/per_case_metrics.csv" ]; then
      echo
      echo ">>> [recipe-controlled] AURAS (full, single ckpt + 8-way TTA) vs $b <<<"
      echo "    (matched eval recipe; residual gap = aux-head loss + deep"
      echo "     supervision, both AURAS-intrinsic and unavailable to a baseline)"
      python -m evaluation.stats compare \
          "$BD/per_case_metrics.csv" \
          "$FD_CTRL/per_case_metrics.csv" \
          --mode tta_post --label-a "$b" --label-b AURAS-single
    fi
  done

  # Recipe floor: our minimal CNN vs the vanilla U-Net. Same family, so a
  # gap here is the data pipeline / training recipe, not architecture.
  case " $BASELINES " in
    *" unet3d "*)
      UD="results/unet3d/eval_${EXP_NAME}"
      [ -d "$UD" ] || UD=$(ls -1dt results/unet3d/eval_* 2>/dev/null | head -1)
      CD=$(ls -1dt results/base_cnn/eval_* 2>/dev/null | head -1 || true)
      if [ -n "${UD:-}" ] && [ -n "${CD:-}" ] \
         && [ -f "$UD/per_case_metrics.csv" ] && [ -f "$CD/per_case_metrics.csv" ]; then
        echo
        echo ">>> base_cnn vs unet3d  [recipe floor sanity] <<<"
        python -m evaluation.stats compare \
            "$UD/per_case_metrics.csv" \
            "$CD/per_case_metrics.csv" \
            --mode tta_post --label-a unet3d --label-b base_cnn
      else
        echo "[warn] base_cnn or unet3d eval missing - skipping floor check."
      fi
      ;;
  esac
fi

echo
echo "================================================================"
echo "Phase 7 done."
echo "  baseline logs  : logs/run_<baseline>_${EXP_NAME}_*/"
echo "  baseline eval  : results/<baseline>/eval_${EXP_NAME}/"
echo "  matched arm    : results/full/eval_${EXP_NAME}_single/  (single ckpt)"
echo "  complexity     : results/complexity.csv"
echo
echo "  Two comparisons were reported per baseline:"
echo "   [system]            full+ensemble+32-TTA vs baseline - the deployed"
echo "                       system claim (standard in the literature)."
echo "   [recipe-controlled] full single-ckpt + 8-way TTA vs baseline - the"
echo "                       eval recipe is matched; quote THIS for any"
echo "                       'architecture is better' claim, and note the"
echo "                       residual gap is the Boundary/Uncertainty loss +"
echo "                       deep supervision (architecture-intrinsic to AURAS)."
echo "  Plus base_cnn vs unet3d: recipe floor (same family, recipe-matched)."
echo "  Phase-7 gate: AURAS beats swinunetr / segresnet / unet3d on tta_post"
echo "  mean Dice (paired Wilcoxon, p<0.05) on BOTH arms; base_cnn >= unet3d."
echo "  Feed these into Phase 8 (evaluation.aggregate_final) for the final"
echo "  cross-model table."
echo "================================================================"
echo
echo "Detach now with Ctrl-b d, or just close the terminal - tmux keeps it."
