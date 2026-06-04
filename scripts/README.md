# scripts/

Helper scripts that sit outside the `src/` package.

| Script | Purpose |
| --- | --- |
| `prepare_webapp_assets.py` | One-time build of the web app's population statistics from the validation split. Optional `--aal-source <atlas>` enables the anatomy card. Run once before serving the web app. |
| `run_complexity.py` | Convenience wrapper around `python -m evaluation.complexity` — profiles params / FLOPs / VRAM / latency / FPS for a variant and appends to a CSV. |

## Training the three models

Training recipes live in the per-variant presets in
[../src/training/train_variant.py](../src/training/train_variant.py); the
commands below just select a variant.

```bash
# baseline — plain 3D Residual U-Net
python src/training/train_variant.py --variant base_cnn

# Complex — all components enabled (bf16, uncertainty + boundary loss, top-5 snapshots)
python src/training/train_variant.py --variant full --epochs 300 --warmup 10

# AURA (hybrid, the deployed model) — CNN encoder + transformer bottleneck + CNN decoder
python src/training/train_variant.py --variant hybrid --epochs 300
```

Checkpoints are written to `logs/run_<variant>_*/` (`best_model.pth`, plus
`snapshot_top*.pth` for variants that save a top-K snapshot ensemble).

## Evaluating

```bash
# single best checkpoint
python src/evaluation/evaluate_variant.py --variant base_cnn --vmin-sweep

# snapshot ensemble + 32-view extended TTA (full / hybrid)
python src/evaluation/evaluate_variant.py --variant full \
    --ensemble-ckpts "logs/run_full_*/snapshot_top*.pth" \
    --tta-extended --vmin-sweep --overlap 0.625 --run-name eval_ensemble
```

Results are written to `results/<variant>/eval_*/`. To compare two runs with a
paired Wilcoxon test + bootstrap 95% CIs:

```bash
python -m evaluation.stats compare \
    results/base_cnn/eval_X/per_case_metrics.csv \
    results/hybrid/eval_Y/per_case_metrics.csv \
    --mode tta_post --label-a baseline --label-b AURA
```
