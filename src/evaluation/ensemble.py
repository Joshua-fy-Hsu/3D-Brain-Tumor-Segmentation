"""Phase 6 — snapshot-ensemble predictor.

Wraps N models of the same architecture so that one forward pass returns
mean logits over the ensemble. The members are each pre-wrapped with
``DictToSegAdapter`` so their forward returns a plain ``(B, C, D, H, W)``
tensor (consistent with what ``sliding_window_inference`` consumes).

Logits are summed across members and divided by N; downstream softmax /
sigmoid happens in the existing inference path (``infer_modes``) without
modification.

Typical use:

    members = sorted(glob.glob('logs/run_full_*/snapshot_top*.pth'))
    ens = load_ensemble(members, variant='full', device='cuda')
    ens.eval()
    # Pass `ens` wherever a regular model + DictToSegAdapter would go.
"""
from __future__ import annotations

import glob
import os
from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from model.registry import build_variant
from evaluation._core import DictToSegAdapter


class EnsemblePredictor(nn.Module):
    """Average N already-adapted models' logits at each forward call.

    Members must produce identically-shaped logits. ``forward`` returns the
    mean over members (no softmax — downstream code applies it).
    """

    def __init__(self, members: Iterable[nn.Module]):
        super().__init__()
        members = list(members)
        if len(members) == 0:
            raise ValueError("EnsemblePredictor needs at least one member")
        self.members = nn.ModuleList(members)
        # Each member should already be a DictToSegAdapter so dict outputs
        # collapse to plain tensors. Soft-assert (don't crash on plain
        # tensor-returning models, for flexibility).
        n_adapted = sum(1 for m in self.members if isinstance(m, DictToSegAdapter))
        if n_adapted not in (0, len(self.members)):
            raise ValueError(
                "EnsemblePredictor members must be uniformly wrapped: either "
                "all DictToSegAdapter or none. Got a mix."
            )

    def __len__(self) -> int:
        return len(self.members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sequential to keep activation memory bounded to a single member.
        total: Optional[torch.Tensor] = None
        for m in self.members:
            out = m(x)
            total = out if total is None else total + out
        return total / float(len(self.members))


def model_has_dropout(model: nn.Module) -> bool:
    """Delegate to the first member when given an EnsemblePredictor."""
    from evaluation.uncertainty import model_has_dropout as _has
    if isinstance(model, EnsemblePredictor):
        return _has(model.members[0])
    return _has(model)


def resolve_ckpt_glob(pattern: str) -> List[str]:
    """Expand a glob (in Python — works on Windows + tmux). Returns sorted
    matches. Fails loud if no matches."""
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no checkpoints matched glob: {pattern!r}")
    return matches


def load_ensemble(
    ckpt_paths: Iterable[str],
    variant: str,
    device: str | torch.device = "cuda",
    **build_kwargs,
) -> EnsemblePredictor:
    """Instantiate N copies of ``variant`` and load N checkpoints into them.

    Each member is wrapped with ``DictToSegAdapter`` and moved to ``device``.
    """
    ckpt_paths = list(ckpt_paths)
    if len(ckpt_paths) == 0:
        raise ValueError("load_ensemble: ckpt_paths is empty")
    members = []
    for p in ckpt_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        m = build_variant(variant, **build_kwargs)
        sd = torch.load(p, map_location="cpu", weights_only=True)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint {p} doesn't match variant '{variant}': "
                f"missing={list(missing)[:5]}... unexpected={list(unexpected)[:5]}..."
            )
        m.eval()
        members.append(DictToSegAdapter(m).to(device))
    ens = EnsemblePredictor(members).to(device)
    ens.eval()
    return ens
