"""Per-stage boundary logit head (Phase 5).

Tiny Conv3d stack producing a single-channel boundary logit from a decoder
stage feature map. Three instances live in `TransResUNet3D` when
`use_boundary=True`, one for each decoder stage (full / 1-2 / 1-4 resolution),
so the boundary supervision mirrors the seg path's deep-supervision pattern.

The head is intentionally small (Conv3d C→16 → IN → LeakyReLU → Conv3d 16→1):
the boundary signal is morphologically simple (one-voxel edge mask), and
overparameterising the head was flagged as a Phase-5 risk — boundary loss
should sharpen the seg path, not learn a separate edge detector that drifts
from the seg targets.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BoundaryHead(nn.Module):
    """Decoder-stage boundary logit head.

    Conv3d(C → 16, k=3, p=1, bias=False) → InstanceNorm3d → LeakyReLU(0.01)
        → Conv3d(16 → 1, k=1).

    Output is a single-channel logit at the same spatial resolution as the
    input feature map. The BoundaryAwareLoss applies BCE-with-logits + an
    edge-restricted Dice term against an online morphological-gradient edge
    GT, so the head's output is unbounded (no sigmoid here).
    """

    def __init__(self, in_channels: int, hidden_channels: int = 16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(hidden_channels, 1, kernel_size=1, bias=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.body.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)
