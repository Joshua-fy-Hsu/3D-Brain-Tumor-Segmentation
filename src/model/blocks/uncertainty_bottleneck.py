"""Uncertainty-guided bottleneck (Phase 4).

Adds a small predictive-variance head at the bottleneck (`(B, 512, 8, 8, 8)`
after the SpectralSwinStage). The variance map is used two ways:

  1. **Soft gating of the bottleneck features.** A learnable scalar α (init=0)
     turns the gate into an identity at the start of training:

         gate = sigmoid(variance) * α + 1.0
         out  = features * gate

     If `variance` carries no useful signal, α stays near 0 and the block is a
     no-op. If high-variance regions correspond to harder voxels that benefit
     from a different gain on the bottleneck features, α drifts off zero.

  2. **Eval-time uncertainty diagnostics.** The (B, 1, 8, 8, 8) variance map is
     trilinearly upsampled to the full patch resolution (128³) and exposed via
     the model's output dict, so the loss / evaluator can consume it without
     re-running inference.

The variance head uses a Softplus tail to enforce non-negativity, matching the
heteroscedastic-aleatoric formulation in Kendall & Gal 2017.

Used only when `TransResUNet3D(use_uncertainty=True)`.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyBottleneck(nn.Module):
    """Predictive-variance head + soft gate at the bottleneck."""

    def __init__(
        self,
        in_channels: int,
        upsample_size: Tuple[int, int, int] = (128, 128, 128),
        hidden_div: int = 4,
    ):
        super().__init__()
        if in_channels % hidden_div != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by hidden_div ({hidden_div})"
            )
        hidden = in_channels // hidden_div

        self.var_head = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(hidden, 1, kernel_size=1, bias=True),
            nn.Softplus(),
        )

        # Gate scalar: gate = sigmoid(variance) * alpha + 1.0. alpha=0 at init
        # keeps the block as identity until variance carries useful signal.
        self.alpha = nn.Parameter(torch.zeros(1))

        self.upsample_size = tuple(upsample_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.var_head.modules():
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

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """features: (B, C, D, H, W) bottleneck features.

        Returns (gated_features, variance_full_res) where:
          - gated_features has the same shape as `features`
          - variance_full_res is upsampled to `self.upsample_size`, shape
            (B, 1, *upsample_size). Non-negative (Softplus).
        """
        variance = self.var_head(features)                  # (B, 1, D, H, W)
        gate = torch.sigmoid(variance) * self.alpha + 1.0
        gated = features * gate

        var_full = F.interpolate(
            variance.float(), size=self.upsample_size,
            mode="trilinear", align_corners=False,
        )
        return gated, var_full
