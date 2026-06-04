"""Variant-aware evaluator.

Reads the variant from the registry, picks the right AMP family (fp16 for CNN,
bf16 for transformer) unless overridden, and writes results to
results/<variant>/eval_*/.

Usage:
  python src/evaluation/evaluate_variant.py --variant base_cnn
  python src/evaluation/evaluate_variant.py --variant full --vmin-sweep
  python src/evaluation/evaluate_variant.py --variant hybrid \\
      --checkpoint logs/run_hybrid_.../best_model.pth
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
from model.registry import build_variant, get_output_mode, get_arch_family
from evaluation._core import (
    add_common_args,
    detect_output_mode,
    find_latest_checkpoint,
    run_evaluation,
    state_dict_matches,
)


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate any registered variant. See src/model/registry.py."
    )
    ap.add_argument("--variant", required=True,
                    help="Variant name from src/model/registry.py")
    add_common_args(ap)
    args = ap.parse_args()

    variant = args.variant
    arch_family = get_arch_family(variant)
    device = torch.device(config.DEVICE)

    # Default AMP per arch family. CLI --amp-dtype still wins.
    if args.amp_dtype is None:
        # _core.run_evaluation already applies the fp16/bf16 default by `arch`,
        # but we set it explicitly here so any per-family override picked up
        # by the registry (e.g., a future "transformer" variant that wants
        # fp16) can be plumbed without touching _core.
        pass

    # Build the model. Some transformer variants accept output_mode at
    # construction time and the head shape depends on it.
    # If a checkpoint is supplied, sniff its head shape first so we instantiate
    # the right variant. Otherwise build with the registry default and then
    # use that to filter checkpoints during auto-discovery.
    registry_output_mode = get_output_mode(variant)

    if args.ensemble_ckpts is not None:
        # Phase 6 — snapshot ensemble. Load N members of the same variant,
        # average logits at forward. The members are already DictToSegAdapter
        # wrapped, so we hand the ensemble straight to run_evaluation as the
        # "model" and bypass wrap_for_eval (which is idempotent anyway).
        from evaluation.ensemble import load_ensemble, resolve_ckpt_glob
        members_paths = resolve_ckpt_glob(args.ensemble_ckpts)
        # Sniff the head shape from the first member to pick output_mode.
        sniffed = detect_output_mode(members_paths[0], fallback=registry_output_mode)
        try:
            ens = load_ensemble(members_paths, variant=variant, device=device,
                                output_mode=sniffed)
            output_mode = sniffed
        except TypeError:
            ens = load_ensemble(members_paths, variant=variant, device=device)
            output_mode = registry_output_mode
        ckpt_path = ";".join(members_paths)  # joined paths for logging/meta
        print(f"[evaluate:{variant}] ensemble: {len(members_paths)} members")
        for i, p in enumerate(members_paths, 1):
            print(f"  [{i}/{len(members_paths)}] {p}")
        model = ens
    elif args.checkpoint is not None:
        ckpt_path = args.checkpoint
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        # Sniff the actual output_mode from the checkpoint head shape.
        sniffed = detect_output_mode(ckpt_path, fallback=registry_output_mode)
        # Try with sniffed mode first; fall back to registry default if the
        # variant doesn't support output_mode kwarg.
        try:
            model = build_variant(variant, output_mode=sniffed).to(device)
            output_mode = sniffed
        except TypeError:
            model = build_variant(variant).to(device)
            output_mode = registry_output_mode
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
    else:
        # Auto-discover: build a probe model and walk logs/ for matching ckpts.
        probe = build_variant(variant).to(device)
        ckpt_path = find_latest_checkpoint(
            os.path.join(ROOT, "logs"), model=probe, arch_label=variant,
        )
        if ckpt_path is None:
            raise FileNotFoundError(
                f"No checkpoint compatible with variant '{variant}' found under "
                f"logs/run_*/best_model.pth. Pass --checkpoint to point at one."
            )
        sniffed = detect_output_mode(ckpt_path, fallback=registry_output_mode)
        try:
            model = build_variant(variant, output_mode=sniffed).to(device)
            output_mode = sniffed
        except TypeError:
            model = probe
            output_mode = registry_output_mode
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)

    # Results dir: results/<variant>/eval_*/  unless caller overrides
    results_base = args.results_dir or os.path.join(ROOT, "results", variant)

    # Pass arch_family as `arch` so _core picks the right AMP default + label
    run_evaluation(args, arch=arch_family, model=model, ckpt=ckpt_path,
                   output_mode=output_mode, results_base_dir=results_base)


if __name__ == "__main__":
    main()
