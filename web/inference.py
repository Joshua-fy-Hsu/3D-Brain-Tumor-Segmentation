"""Inference wrapper: pinned model load + one-shot SW-predict → labels + region probs.

Reuses existing primitives:
  - build_variant("hybrid")               from src.model.registry
  - find_latest_checkpoint                from src.evaluation._core
  - wrap_for_eval (DictToSegAdapter)      from src.evaluation._core
  - sw_predict / _logits_to_4ch_probs     from src.evaluation.uncertainty
  - postprocess_et                         from src.evaluation.postprocess
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.append(SRC)

from configs import config  # noqa: E402
from evaluation import _core as _C  # noqa: E402
from evaluation import postprocess as PP  # noqa: E402
from evaluation import uncertainty as U  # noqa: E402
from model.registry import (  # noqa: E402
    build_variant, get_arch_family, get_display_name, get_output_mode,
)

VARIANT_NAME = "hybrid"
ET_TAU = 0.5
ET_VMIN = 1000

# Product-facing identity. Resolved from the registry so the brand name
# lives in exactly one place; the internal variant/run name is kept
# only for provenance inside metrics.json, never shown in the UI.
MODEL_DISPLAY_NAME = get_display_name(VARIANT_NAME)  # "hybrid" -> "AURA"
MODEL_VERSION = "v1.0"


@dataclass
class LoadedModel:
    model: torch.nn.Module          # already wrap_for_eval'd, .eval(), on device
    variant: str
    checkpoint_path: str
    run_name: str                   # for display in the UI header
    device: torch.device
    output_mode: str                # "softmax" | "sigmoid"
    amp_dtype: torch.dtype


def load_model(logs_dir: Optional[str] = None,
               variant: str = VARIANT_NAME,
               device: Optional[torch.device] = None) -> LoadedModel:
    """Build the variant, find the latest compatible checkpoint, load it.

    Fails loud (RuntimeError) if no compatible checkpoint exists.
    """
    if logs_dir is None:
        logs_dir = os.path.join(ROOT, "logs")
    if device is None:
        device = torch.device(config.DEVICE)

    model = build_variant(variant)
    ckpt = _C.find_latest_checkpoint(logs_dir, model=model, arch_label=variant)
    if ckpt is None:
        raise RuntimeError(
            f"No compatible best_model.pth found for variant '{variant}' "
            f"under {logs_dir}. Train the variant first or copy a checkpoint."
        )

    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = _C.wrap_for_eval(model).to(device).eval()

    arch = get_arch_family(variant)
    amp_dtype = torch.bfloat16 if arch == "transformer" else torch.float16
    output_mode = get_output_mode(variant)
    run_name = os.path.basename(os.path.dirname(ckpt))

    return LoadedModel(
        model=model,
        variant=variant,
        checkpoint_path=ckpt,
        run_name=run_name,
        device=device,
        output_mode=output_mode,
        amp_dtype=amp_dtype,
    )


@dataclass
class InferenceResult:
    probs_4ch: np.ndarray   # (4, D, H, W) float32, softmax probabilities
    labels: np.ndarray      # (D, H, W) uint8 in {0,1,2,3}, post-processed


@torch.no_grad()
def run(loaded: LoadedModel, x: torch.Tensor,
        roi: tuple[int, int, int] = (128, 128, 128),
        overlap: float = 0.5) -> InferenceResult:
    """Run sliding-window inference and decode to labels with ET cleanup.

    Args:
        loaded: a LoadedModel from load_model().
        x: (1, 5, D, H, W) float32 tensor on loaded.device.
    """
    logits = U.sw_predict(loaded.model, x, roi=roi, overlap=overlap,
                          amp_dtype=loaded.amp_dtype)
    probs_t = U._logits_to_4ch_probs(logits, output_mode=loaded.output_mode)
    probs = probs_t[0].detach().cpu().numpy().astype(np.float32)

    if loaded.output_mode == "softmax":
        labels = PP.postprocess_et(probs, tau_et=ET_TAU, v_min=ET_VMIN).astype(np.uint8)
    else:
        labels = _C.decode_labels(probs, output_mode=loaded.output_mode)

    return InferenceResult(probs_4ch=probs, labels=labels)


def run_uncertainty(loaded: LoadedModel, x: torch.Tensor,
                    T: int = 10,
                    roi: tuple[int, int, int] = (128, 128, 128),
                    overlap: float = 0.5) -> Optional[np.ndarray]:
    """Run T MC-Dropout forward passes and return predictive entropy map.

    Returns a (D, H, W) float32 array, or None if the model has no dropout.
    x must still be on loaded.device.
    """
    if not U.model_has_dropout(loaded.model):
        return None
    with torch.no_grad():
        _, entropy, _ = U.mc_dropout_predict(
            loaded.model, x, T=T, roi=roi, overlap=overlap,
            output_mode=loaded.output_mode,
        )
    if entropy is None:
        return None
    # entropy shape: (1, 1, D, H, W) → (D, H, W)
    return entropy[0, 0].detach().cpu().numpy().astype(np.float32)
