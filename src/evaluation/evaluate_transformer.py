"""Transformer evaluation pipeline entry point (ResUnet3DTransformer).

Runs over the val split of BraTS2021_Optimized, evaluates inference modes
(baseline, +TS [if softmax head], +MC Dropout, +TTA, +postprocess) and writes
per-case metrics + uncertainty + plots into
`results/transformer/eval_YYYYMMDD-HHMMSS/`.

Defaults to bf16 sliding-window inference to match training precision —
fp16 risks attention-softmax overflow in the transformer.

Usage:
    python src/evaluation/evaluate_transformer.py
    python src/evaluation/evaluate_transformer.py --max-cases 10 --skip-robustness
    python src/evaluation/evaluate_transformer.py --checkpoint logs/run_.../best_model.pth
"""
import argparse
import os
import sys

CURR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(CURR)
ROOT = os.path.dirname(SRC)
if SRC not in sys.path:
    sys.path.append(SRC)

import torch

from configs import config
from model.model_transformer import ResUnet3DTransformer
from evaluation._core import (
    add_common_args,
    detect_output_mode,
    find_latest_checkpoint,
    run_evaluation,
)


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate the ResUnet3D-Transformer checkpoint."
    )
    add_common_args(ap)
    args = ap.parse_args()

    device = torch.device(config.DEVICE)

    def _build(output_mode):
        # Must match the training-time config in train_transformer.py.
        return ResUnet3DTransformer(
            in_channels=config.IN_CHANNELS,
            num_classes=config.NUM_CLASSES,
            windowed_depth=4,
            output_mode=output_mode,
        ).to(device)

    if args.checkpoint is not None:
        ckpt = args.checkpoint
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        output_mode = detect_output_mode(ckpt)
        model = _build(output_mode)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
    else:
        # Probe with sigmoid first (model default) — output_mode does not
        # affect state-dict key set, only head channel count.
        probe = _build(output_mode="sigmoid")
        ckpt = find_latest_checkpoint(os.path.join(ROOT, "logs"),
                                       model=probe, arch_label="transformer")
        if ckpt is None:
            raise FileNotFoundError(
                "No transformer-compatible checkpoint found under logs/run_*/best_model.pth. "
                "Pass --checkpoint to point at one explicitly.")
        output_mode = detect_output_mode(ckpt)
        model = _build(output_mode)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)

    results_base = os.path.join(ROOT, "results", "transformer")
    run_evaluation(args, arch="transformer", model=model, ckpt=ckpt,
                   output_mode=output_mode, results_base_dir=results_base)


if __name__ == "__main__":
    main()
