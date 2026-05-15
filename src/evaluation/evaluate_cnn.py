"""CNN evaluation pipeline entry point (ResUnet3D).

Runs over the val split of BraTS2021_Optimized, evaluates inference modes
(baseline, +TS, +MC Dropout, +TTA, +postprocess) and writes per-case metrics
+ uncertainty + plots into `results/cnn/eval_YYYYMMDD-HHMMSS/`.

Usage:
    python src/evaluation/evaluate_cnn.py
    python src/evaluation/evaluate_cnn.py --max-cases 10 --skip-robustness
    python src/evaluation/evaluate_cnn.py --checkpoint logs/run_.../best_model.pth
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
from model.model import ResUnet3D
from evaluation._core import (
    add_common_args,
    detect_output_mode,
    find_latest_checkpoint,
    run_evaluation,
)


def main():
    ap = argparse.ArgumentParser(description="Evaluate the 3D ResU-Net (CNN) checkpoint.")
    add_common_args(ap)
    args = ap.parse_args()

    device = torch.device(config.DEVICE)

    def _build():
        return ResUnet3D(in_channels=config.IN_CHANNELS,
                         num_classes=config.NUM_CLASSES).to(device)

    if args.checkpoint is not None:
        ckpt = args.checkpoint
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        output_mode = detect_output_mode(ckpt)
        model = _build()
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
    else:
        probe = _build()
        ckpt = find_latest_checkpoint(os.path.join(ROOT, "logs"),
                                       model=probe, arch_label="cnn")
        if ckpt is None:
            raise FileNotFoundError(
                "No CNN-compatible checkpoint found under logs/run_*/best_model.pth. "
                "Pass --checkpoint to point at one explicitly.")
        output_mode = detect_output_mode(ckpt)
        model = probe
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)

    results_base = os.path.join(ROOT, "results", "cnn")
    run_evaluation(args, arch="cnn", model=model, ckpt=ckpt,
                   output_mode=output_mode, results_base_dir=results_base)


if __name__ == "__main__":
    main()
