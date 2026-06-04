"""Model variant registry.

Single source of truth for the variants used in the ablation study and the
baseline comparisons. Keeps the trainer and evaluator generic — they take a
`--variant <name>` flag and look up the factory here.

Each entry is `(factory_callable, default_kwargs, output_mode, arch_family)`:
  - `factory_callable(**kwargs)` returns the nn.Module
  - `default_kwargs` is merged with caller overrides (caller wins)
  - `output_mode` is "softmax" (4-class head) or "sigmoid" (3-channel ET/TC/WT)
  - `arch_family` is "cnn" or "transformer". Drives AMP defaults at eval time
    (fp16 for cnn, bf16 for transformer — fp16 risks attention-softmax overflow).

Variants in build order. Ones marked TODO will be added as their architecture
phases land. The registry intentionally raises `NotImplementedError` for
unbuilt variants rather than silently returning None — fail loud at startup.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, Tuple

CURR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(CURR)
if SRC not in sys.path:
    sys.path.append(SRC)

from model.model import ResUnet3D
from model.trans_resunet import TransResUNet3D
# Clean-slate CNN+Transformer (the headline model, AURA). Standalone — does
# NOT import from model.model / model_transformer / trans_resunet.
# See src/model/hybrid.py.
from model.hybrid import HybridUNet3D


# (factory, default_kwargs, output_mode, arch_family)
VariantSpec = Tuple[Callable[..., Any], Dict[str, Any], str, str]


# The public release keeps three models — a plain CNN baseline, a fully-featured
# "complex" model, and the headline AURA model:
#
#   base_cnn (baseline) — 3D Residual U-Net, no transformer/attention.
#   full     (Complex)  — TransResUNet3D with every component enabled
#                         (modality stems + cross-modal attention + frequency
#                         branch + spectral-Swin stage + predictive-variance
#                         head + boundary head + deeper Swin/encoder + multi-
#                         scale fusion head). The "kitchen-sink" comparison.
#   hybrid   (AURA)     — clean-slate TransBTS-style CNN encoder + transformer
#                         bottleneck + CNN decoder, 4-class softmax head,
#                         MC-Dropout uncertainty. The deployed/web model.
#
VARIANTS: Dict[str, VariantSpec] = {
    "base_cnn": (
        ResUnet3D,
        dict(in_channels=5, num_classes=4, base_filters=32),
        "softmax", "cnn",
    ),
    "full": (
        TransResUNet3D,
        dict(in_channels=5, num_classes=4, base_filters=32,
             use_modality_stems=True, use_cross_modal=True,
             use_freq=True,
             use_spectral_swin=True,
             use_uncertainty=True,
             use_boundary=True,
             spectral_blocks_per_stage=4,
             encoder_extra_depth=True,
             use_multiscale_fusion_head=True,
             decoder_dropout_final=0.05,
             output_mode="softmax"),
        "softmax", "transformer",
    ),
    "hybrid": (
        HybridUNet3D,
        dict(in_channels=5, num_classes=4, base_filters=32),
        "softmax", "cnn",
    ),
}


# Product-facing display names. The registry keys above are load-bearing
# (logs/run_<key>_*, results/<key>/, checkpoint matching) and must NEVER
# change. These are the brand names shown in the web UI, report exports, and
# figures. Anything not listed falls back to its raw key.
#
#   base_cnn -> baseline   the plain 3D Residual U-Net comparison point.
#   full     -> Complex    the all-components-on model.
#   hybrid   -> AURA       the headline / deployed model.
DISPLAY_NAMES: Dict[str, str] = {
    "base_cnn": "baseline",
    "full":     "Complex",
    "hybrid":   "AURA",
}


def build_variant(name: str, **overrides: Any):
    """Instantiate a model variant by name. Caller kwargs override defaults."""
    if name not in VARIANTS:
        raise KeyError(f"Unknown variant '{name}'. Known: {sorted(VARIANTS)}")
    factory, defaults, _, _ = VARIANTS[name]
    kwargs = {**defaults, **overrides}
    return factory(**kwargs)


def get_output_mode(name: str) -> str:
    if name not in VARIANTS:
        raise KeyError(f"Unknown variant '{name}'. Known: {sorted(VARIANTS)}")
    return VARIANTS[name][2]


def get_arch_family(name: str) -> str:
    """Returns 'cnn' or 'transformer' — drives AMP defaults at eval time."""
    if name not in VARIANTS:
        raise KeyError(f"Unknown variant '{name}'. Known: {sorted(VARIANTS)}")
    return VARIANTS[name][3]


def get_display_name(name: str) -> str:
    """Product-facing brand name for a variant. Falls back to the raw key.

    Single source of truth for the AURAS branding — the web UI, report
    export, and figures all resolve through here so the name lives in one
    place. The registry key (e.g. 'full') stays the internal identifier.
    """
    if name not in VARIANTS:
        raise KeyError(f"Unknown variant '{name}'. Known: {sorted(VARIANTS)}")
    return DISPLAY_NAMES.get(name, name)


def list_implemented() -> list[str]:
    """Return the subset of variants whose factory is wired up (not stubbed)."""
    out = []
    for name, (factory, _, _, _) in VARIANTS.items():
        if not getattr(factory, "__name__", "").startswith("_raise"):
            out.append(name)
    return out
