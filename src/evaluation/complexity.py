"""Computational complexity profiling.

Reports for a single model variant:
  - Parameters (total + trainable)
  - FLOPs at a 1×C×128³ input  (via fvcore if installed; else NaN)
  - Peak GPU memory for one forward pass
  - Latency mean ± std and FPS (warmup 5 → time 20 forwards)

Used to populate the "complexity table" the professor required, AND to back
the Path-C deployment story (efficient variant inference cost vs full model).

CLI:
  python -m evaluation.complexity --variant base_cnn
  python -m evaluation.complexity --variant current_transformer --device cpu
  python -m evaluation.complexity --variant base_cnn --out results/complexity.csv

Notes:
  - fvcore.nn.FlopCountAnalysis is the de-facto standard for FLOPs in
    PyTorch. If unavailable, we still report params/memory/latency.
  - Memory measurement requires CUDA. CPU runs report NaN for memory.
  - We profile the full forward, NOT sliding-window inference. SW inference
    cost depends on overlap and ROI; report it separately if needed.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
import torch

CURR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(CURR)
ROOT = os.path.dirname(SRC)
if SRC not in sys.path:
    sys.path.append(SRC)

from configs import config
from model.registry import build_variant


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_flops(model: torch.nn.Module, x: torch.Tensor) -> Optional[float]:
    """Returns FLOPs (multiply-accumulates counted as 1 op) for one forward,
    or None if fvcore is unavailable.

    Models that return a dict (Phase 4+ with auxiliary heads) are wrapped so
    fvcore's tracer sees a single tensor output — the FLOP count is unchanged
    because the tracer counts ops, not return-value handling.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None
    model_was_training = model.training
    model.eval()

    class _SegOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            o = self.m(x)
            if isinstance(o, dict):
                return o["seg"]
            if isinstance(o, (tuple, list)):
                return o[0]
            return o

    target = _SegOnly(model)
    try:
        analysis = FlopCountAnalysis(target, x)
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        flops = float(analysis.total())
    except Exception as e:
        print(f"[complexity] fvcore failed: {e}")
        flops = None
    if model_was_training:
        model.train()
    return flops


@torch.no_grad()
def measure_memory_and_latency(
    model: torch.nn.Module, x: torch.Tensor,
    warmup: int = 5, iters: int = 20,
) -> dict:
    """Returns dict with peak_mem_mb, latency_mean_ms, latency_std_ms, fps."""
    device = x.device
    model.eval()
    is_cuda = device.type == "cuda"

    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        _ = model(x)
    if is_cuda:
        torch.cuda.synchronize()

    times = []
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    for _ in range(iters):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        if is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    times = np.asarray(times)
    peak_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if is_cuda else float("nan")
    return dict(
        peak_mem_mb=float(peak_mb),
        latency_mean_ms=float(times.mean()),
        latency_std_ms=float(times.std(ddof=1)) if len(times) > 1 else 0.0,
        fps=float(1000.0 / times.mean()) if times.mean() > 0 else float("nan"),
    )


def profile_variant(name: str, device: str = "cuda",
                    in_channels: int = 5, patch: tuple[int, int, int] = (128, 128, 128),
                    warmup: int = 5, iters: int = 20,
                    skip_flops: bool = False) -> dict:
    """Full profile of one variant. Returns dict suitable for a CSV row."""
    if device == "cuda" and not torch.cuda.is_available():
        print("[complexity] CUDA requested but not available — falling back to CPU")
        device = "cpu"
    dev = torch.device(device)

    # Build the model. `current_transformer` uses 5 channels, base_cnn uses 5
    # too (BraTS-2021 4 modalities + foreground mask). Registry decides.
    model = build_variant(name).to(dev)
    total_p, trainable_p = count_params(model)

    x = torch.randn(1, in_channels, *patch, device=dev)

    flops = None if skip_flops else count_flops(model, x)
    timing = measure_memory_and_latency(model, x, warmup=warmup, iters=iters)

    out = dict(
        variant=name,
        device=device,
        input_shape=f"1x{in_channels}x{patch[0]}x{patch[1]}x{patch[2]}",
        params_total=total_p,
        params_trainable=trainable_p,
        flops=flops if flops is not None else float("nan"),
        gflops=(flops / 1e9) if flops is not None else float("nan"),
        **timing,
    )
    return out


def _cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True,
                    help="Variant name from src/model/registry.py")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--patch", type=int, nargs=3, default=(128, 128, 128))
    ap.add_argument("--in-channels", type=int, default=config.IN_CHANNELS)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--skip-flops", action="store_true",
                    help="Skip fvcore FLOP counting (it can be slow / fragile).")
    ap.add_argument("--out", default=None,
                    help="Append to this CSV. If absent, only prints to stdout.")
    args = ap.parse_args()

    row = profile_variant(args.variant, device=args.device,
                          in_channels=args.in_channels, patch=tuple(args.patch),
                          warmup=args.warmup, iters=args.iters,
                          skip_flops=args.skip_flops)

    print(f"\n========== complexity: {args.variant} ==========")
    for k, v in row.items():
        if isinstance(v, float):
            print(f"  {k:>18s}: {v:.4f}" if abs(v) < 1e6 else f"  {k:>18s}: {v:.4e}")
        else:
            print(f"  {k:>18s}: {v}")

    if args.out:
        import pandas as pd
        df_new = pd.DataFrame([row])
        if os.path.exists(args.out):
            df = pd.read_csv(args.out)
            # Replace existing row for this variant if present
            df = df[df["variant"] != args.variant]
            df = pd.concat([df, df_new], ignore_index=True)
        else:
            df = df_new
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    _cli()
