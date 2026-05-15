"""Per-modality stem for TransResUNet3D.

Splits the 5-channel input (T1, T1CE, T2, FLAIR, foreground) into
4 modality-specific stems + a separately-carried foreground channel.
Each stem is Conv3d(1, stem_channels) -> InstanceNorm3d -> LeakyReLU(0.01),
weights independent across modalities so each can learn its own low-level
filters before fusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ModalityStem(nn.Module):
    def __init__(self, stem_channels: int = 8, num_modalities: int = 4):
        super().__init__()
        self.num_modalities = num_modalities
        self.stem_channels = stem_channels
        self.stems = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(1, stem_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(stem_channels),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
            )
            for _ in range(num_modalities)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 5, D, H, W). Channels 0..3 are T1/T1CE/T2/FLAIR, channel 4
               is the precomputed foreground mask.
        Returns:
            mod_feats: (B, num_modalities, stem_channels, D, H, W) — stacked.
            fg:        (B, 1, D, H, W) — foreground channel, untouched.
        """
        assert x.shape[1] >= self.num_modalities + 1, (
            f"ModalityStem expects >= {self.num_modalities + 1} input channels, "
            f"got {x.shape[1]}"
        )
        mods = [self.stems[m](x[:, m:m + 1]) for m in range(self.num_modalities)]
        mod_feats = torch.stack(mods, dim=1)        # (B, M, C, D, H, W)
        fg = x[:, self.num_modalities:self.num_modalities + 1]  # (B, 1, D, H, W)
        return mod_feats, fg
