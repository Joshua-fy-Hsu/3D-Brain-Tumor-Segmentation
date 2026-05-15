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
from model.model_transformer import ResUnet3DTransformer
from model.trans_resunet import TransResUNet3D
# Phase 7 baseline factories. These do a *lazy* MONAI import inside the
# factory body, so importing them here keeps registry import MONAI-free.
from model.baselines import build_segresnet, build_swinunetr, build_unet3d


# (factory, default_kwargs, output_mode, arch_family)
VariantSpec = Tuple[Callable[..., Any], Dict[str, Any], str, str]


def _not_implemented(name: str):
    def _raise(**_kw):
        raise NotImplementedError(
            f"Variant '{name}' is not implemented yet. See implementation plan "
            f"phases. Available variants: {sorted(VARIANTS)}"
        )
    return _raise


VARIANTS: Dict[str, VariantSpec] = {
    # ---- Baselines / current models (Phase 0) --------------------------------
    "base_cnn": (
        ResUnet3D,
        dict(in_channels=5, num_classes=4, base_filters=32),
        "softmax", "cnn",
    ),
    "current_transformer": (
        ResUnet3DTransformer,
        dict(in_channels=5, num_classes=4, base_filters=32,
             output_mode="softmax",
             decoder_dropout_inner=0.0, decoder_dropout_final=0.05),
        "softmax", "transformer",
    ),

    # ---- Ablation variants (Phases 1–6) --------------------------------------
    # Filled in as each phase lands.
    "cross_modal": (
        TransResUNet3D,
        dict(in_channels=5, num_classes=4, base_filters=32,
             use_modality_stems=True, use_cross_modal=True,
             output_mode="softmax"),
        "softmax", "cnn",
    ),
    "frequency": (
        TransResUNet3D,
        dict(in_channels=5, num_classes=4, base_filters=32,
             use_modality_stems=True, use_cross_modal=True,
             use_freq=True,
             output_mode="softmax"),
        "softmax", "cnn",
    ),
    "spectral_swin": (
        TransResUNet3D,
        dict(in_channels=5, num_classes=4, base_filters=32,
             use_modality_stems=True, use_cross_modal=True,
             use_freq=True,
             use_spectral_swin=True,
             output_mode="softmax"),
        "softmax", "transformer",
    ),
    "uncertainty": (
        TransResUNet3D,
        dict(in_channels=5, num_classes=4, base_filters=32,
             use_modality_stems=True, use_cross_modal=True,
             use_freq=True,
             use_spectral_swin=True,
             use_uncertainty=True,
             output_mode="softmax"),
        "softmax", "transformer",
    ),
    "boundary": (
        TransResUNet3D,
        dict(in_channels=5, num_classes=4, base_filters=32,
             use_modality_stems=True, use_cross_modal=True,
             use_freq=True,
             use_spectral_swin=True,
             use_uncertainty=True,
             use_boundary=True,
             output_mode="softmax"),
        "softmax", "transformer",
    ),
    # Phase 6: deeper Swin (4 blocks/stage), extra encoder depth at 16^3,
    # multi-scale fusion head. Same Phase-1-5 flags as `boundary`; different
    # architecture from `boundary` (own checkpoint). Trained with top-K=5
    # snapshot saving; evaluated with ensemble + extended TTA + tuned V_min.
    #
    # decoder_dropout_final=0.05: first run used 0.10 and underfit (ET dropped
    # 0.066 vs boundary). boundary's TransResUNet3D defaults dropout to 0.0;
    # 0.05 here is a compromise that preserves MC Dropout uncertainty at eval
    # without over-regularising the larger model.
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

    # ---- Phase 9: leave-one-out TRUE ablation from `full` --------------------
    # Each variant removes EXACTLY ONE component from `full`. They are trained
    # with a `full_preset`-derived recipe (see TRAINING_PRESETS) so the ONLY
    # difference vs `full` is the removed component — not the training schedule.
    # `spectral_swin` and `modality_stems` are intentionally absent: the
    # monotonic guards in trans_resunet.py couple them to dependent heads, so
    # they cannot be cleanly leave-one-out ablated (report via chain delta).
    "full_no_cross_modal": (
        TransResUNet3D,
        dict(in_channels=5, num_classes=4, base_filters=32,
             use_modality_stems=True, use_cross_modal=False,   # <-- removed
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
    "full_no_freq": (
        TransResUNet3D,
        dict(in_channels=5, num_classes=4, base_filters=32,
             use_modality_stems=True, use_cross_modal=True,
             use_freq=False,                                    # <-- removed
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
    "full_no_uncertainty": (
        TransResUNet3D,
        dict(in_channels=5, num_classes=4, base_filters=32,
             use_modality_stems=True, use_cross_modal=True,
             use_freq=True,
             use_spectral_swin=True,
             use_uncertainty=False,                             # <-- removed
             use_boundary=True,
             spectral_blocks_per_stage=4,
             encoder_extra_depth=True,
             use_multiscale_fusion_head=True,
             decoder_dropout_final=0.05,
             output_mode="softmax"),
        "softmax", "transformer",
    ),
    "full_no_boundary": (
        TransResUNet3D,
        dict(in_channels=5, num_classes=4, base_filters=32,
             use_modality_stems=True, use_cross_modal=True,
             use_freq=True,
             use_spectral_swin=True,
             use_uncertainty=True,
             use_boundary=False,                                # <-- removed
             spectral_blocks_per_stage=4,
             encoder_extra_depth=True,
             use_multiscale_fusion_head=True,
             decoder_dropout_final=0.05,
             output_mode="softmax"),
        "softmax", "transformer",
    ),
    # `full` minus the three Phase-6 architectural upgrades only. Matches the
    # `boundary` architecture EXCEPT decoder_dropout_final stays at 0.05 (vs
    # boundary's 0.0) so the sole delta vs `full` is the arch trio, not also
    # the dropout. spectral_blocks_per_stage=2 / extra_depth off / fusion off
    # reproduce the TransResUNet3D defaults `boundary` uses.
    "full_no_arch": (
        TransResUNet3D,
        dict(in_channels=5, num_classes=4, base_filters=32,
             use_modality_stems=True, use_cross_modal=True,
             use_freq=True,
             use_spectral_swin=True,
             use_uncertainty=True,
             use_boundary=True,
             spectral_blocks_per_stage=2,                       # <-- removed (4->2 default)
             encoder_extra_depth=False,                         # <-- removed
             use_multiscale_fusion_head=False,                  # <-- removed
             decoder_dropout_final=0.05,
             output_mode="softmax"),
        "softmax", "transformer",
    ),

    # ---- Efficient variant (Phase 6b — pruned/distilled from `full`) ---------
    "full_efficient":   (_not_implemented("full_efficient"),   {}, "softmax", "transformer"),

    # ---- External baselines (Phase 7) ----------------------------------------
    # Wired up when each baseline is integrated. MONAI ones are drop-in;
    # nnU-Net runs as an external framework so it lives outside this registry.
    # The 3 below are the chosen comparison set: SOTA transformer (swinunetr),
    # SOTA CNN / BraTS-2018 winner (segresnet), lower-bound baseline (unet3d).
    # All run in standard, no-deep-supervision config — the fair comparison.
    "swinunetr": (
        build_swinunetr,
        dict(in_channels=5, num_classes=4, feature_size=48),
        "softmax", "transformer",
    ),
    "unetr":            (_not_implemented("unetr"),            {}, "softmax", "transformer"),
    "segresnet": (
        build_segresnet,
        dict(in_channels=5, num_classes=4, init_filters=32),
        "softmax", "cnn",
    ),
    "transbts":         (_not_implemented("transbts"),         {}, "softmax", "transformer"),
    "vtunet":           (_not_implemented("vtunet"),           {}, "softmax", "transformer"),
    "mednext":          (_not_implemented("mednext"),          {}, "softmax", "cnn"),
    "unet3d": (
        build_unet3d,
        dict(in_channels=5, num_classes=4),
        "softmax", "cnn",
    ),
}


# Product-facing display names. The registry keys above are load-bearing
# (logs/run_<key>_*, results/<key>/, scripts/phase*.sh, checkpoint matching)
# and must NEVER change. These are the brand names shown in the web UI,
# report exports, and figures. `full` is AURAS; the ablation rows are the
# AURAS family. Anything not listed falls back to its raw key.
#
# AURAS = All-modality, Uncertainty-aware, Residual, Aggregation, Spectral.
DISPLAY_NAMES: Dict[str, str] = {
    "full":          "AURAS",
    "boundary":      "AURAS-B",
    "uncertainty":   "AURAS-U",
    "spectral_swin": "AURAS-S",
    "frequency":     "AURAS-F",
    "cross_modal":   "AURAS-CM",
    "base_cnn":      "baseline",
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
