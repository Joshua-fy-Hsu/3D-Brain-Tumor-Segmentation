"""Multi-scale fusion seg head (Phase 6).

Residual refinement of the standard ``final_conv(d1)`` head with multi-scale
context from d2 (1/2 res) and d3 (1/4 res). When ``use_multiscale_fusion_head=True``
in TransResUNet3D, this replaces the plain final 1x1 conv. The head computes:

    dec2_up = trilinear(Conv1x1(d2: c2 -> mid), size=d1.shape[2:])
    dec3_up = trilinear(Conv1x1(d3: c3 -> mid), size=d1.shape[2:])
    refine  = Conv3x3(concat([d1, dec2_up, dec3_up], dim=1): c1+2*mid -> c1)
              + InstanceNorm3d + LeakyReLU(0.01)
    final   = Conv1x1(d1 + alpha * refine -> head_channels)

The learnable scalar ``alpha`` is initialised to 0, so at init the head is
identical to ``final_conv(d1)`` (the proven baseline used by boundary). Multi-
scale context is added residually as the network finds it useful. This mirrors
the Phase-3 SpectralWindowedBlock and Phase-4 UncertaintyBottleneck alpha-gate
pattern, and was added in Phase-6 v2 after v1 underfit (boundary head + deep
supervision + fusion head all pulled on d1 with conflicting objectives).

Boundary heads keep reading the raw d1/d2/d3 features (the Phase-5 supervision
contract is unchanged). Deep-supervision heads stay attached to d2/d3 as well
- they only train, the fused head owns the final prediction.

Justification: multi-scale fusion is the well-trodden HD95 reducer in BraTS
(Lin 2017 FPN; Wang 2018 TransBTS+). The fine-resolution d1 carries boundary
cues, d3 carries mid-range tumor context. The alpha gate prevents the fused
features from disrupting d1 during early training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFusionHead(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        c3: int,
        mid: int = 16,
        head_channels: int = 4,
    ):
        super().__init__()
        self.proj2 = nn.Conv3d(c2, mid, kernel_size=1, bias=False)
        self.proj3 = nn.Conv3d(c3, mid, kernel_size=1, bias=False)
        # 3x3 fuse projects (d1 + d2_up + d3_up) back to c1 channels so the
        # residual ``d1 + alpha * refine`` is dimensionally consistent.
        self.fuse = nn.Sequential(
            nn.Conv3d(c1 + 2 * mid, c1, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(c1),
            nn.LeakyReLU(0.01, inplace=True),
        )
        # Learnable gate: init=0 so the head starts as final_conv(d1).
        self.alpha = nn.Parameter(torch.zeros(1))
        self.final_conv = nn.Conv3d(c1, head_channels, kernel_size=1, bias=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
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

    def forward(
        self,
        d1: torch.Tensor,
        d2: torch.Tensor,
        d3: torch.Tensor,
    ) -> torch.Tensor:
        size = d1.shape[2:]
        d2u = F.interpolate(self.proj2(d2), size=size, mode="trilinear",
                            align_corners=False)
        d3u = F.interpolate(self.proj3(d3), size=size, mode="trilinear",
                            align_corners=False)
        refine = self.fuse(torch.cat((d1, d2u, d3u), dim=1))
        return self.final_conv(d1 + self.alpha * refine)
