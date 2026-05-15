"""Frequency-aware feature extraction block (Phase 2).

Sits between the cross-modal fusion stem and `enc2`. Augments the spatial
feature map with a learned spectral filter:

    X (B,C,D,H,W) ──► rFFTn ──► band-gated ──► irFFTn ──► X_freq
                                                 │
                                                 ▼
                                  concat([X, X_freq]) ─► 1x1 conv ─► IN ─► LReLU
                                                 │
                                                 + residual(X)

The filter is a per-channel × per-band scalar gain over 3 radial bands
(low / mid / high) of the rFFTn output. Initialized to 1.0 so the block
is identity at start; gains drift away from 1.0 only if they help.

FFT autocast gotcha: `torch.fft.*` does not support autocast on CUDA — wrap
the FFT path in `torch.amp.autocast("cuda", enabled=False)` and cast in/out.

Memory: rfftn on (32, 128^3) is ~4 MB scratch. Negligible vs encoder maps.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyAwareBlock(nn.Module):
    def __init__(self, channels: int = 32, num_bands: int = 3):
        super().__init__()
        self.channels = channels
        self.num_bands = num_bands

        # Per-channel × per-band gain. Init at 1.0 → identity filter.
        self.band_gain = nn.Parameter(torch.ones(channels, num_bands))

        # Spectral-spatial fusion: concat(X, X_freq) -> 1x1 conv back to C.
        self.fuse = nn.Sequential(
            nn.Conv3d(2 * channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        # Lazy cache of (band_onehot) keyed by spatial shape. Allocated as a
        # buffer so it follows .to(device) on first use.
        self._cached_shape: Tuple[int, int, int] | None = None
        self.register_buffer("_band_onehot", torch.empty(0), persistent=False)

    # -----------------------------------------------------------------------
    def _build_band_onehot(self, shape: Tuple[int, int, int],
                           device: torch.device) -> torch.Tensor:
        """One-hot band membership over the rfftn grid. Shape (B_bands, D, H, Wr)."""
        D, H, W = shape
        Wr = W // 2 + 1
        kd = torch.fft.fftfreq(D, device=device)
        kh = torch.fft.fftfreq(H, device=device)
        kw = torch.fft.rfftfreq(W, device=device)
        # |k|/k_max in each axis is in [0, 0.5]. Aggregate radius normalized
        # to [0, 1] by dividing by the corner magnitude (0.5 * sqrt(3)).
        gd, gh, gw = torch.meshgrid(kd, kh, kw, indexing="ij")
        r = torch.sqrt(gd * gd + gh * gh + gw * gw) / (0.5 * (3 ** 0.5))
        r = r.clamp_(0.0, 1.0)

        # Equal-width radial bins: low [0, 1/3), mid [1/3, 2/3), high [2/3, 1].
        thresholds = torch.linspace(0.0, 1.0, self.num_bands + 1, device=device)
        onehot = torch.zeros(self.num_bands, D, H, Wr, device=device)
        for b in range(self.num_bands):
            lo, hi = thresholds[b], thresholds[b + 1]
            if b == self.num_bands - 1:
                mask = (r >= lo) & (r <= hi)
            else:
                mask = (r >= lo) & (r < hi)
            onehot[b] = mask.float()
        return onehot

    def _get_band_onehot(self, shape: Tuple[int, int, int],
                         device: torch.device) -> torch.Tensor:
        if self._cached_shape != shape or self._band_onehot.numel() == 0 \
                or self._band_onehot.device != device:
            onehot = self._build_band_onehot(shape, device)
            self._band_onehot = onehot
            self._cached_shape = shape
        return self._band_onehot

    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        assert C == self.channels, \
            f"FrequencyAwareBlock expected {self.channels} channels, got {C}"

        # FFT path: must run in fp32. torch.fft.* doesn't honor autocast on CUDA
        # and fp16 complex math is fragile.
        device_type = "cuda" if x.is_cuda else "cpu"
        x_dtype = x.dtype
        with torch.amp.autocast(device_type, enabled=False):
            x32 = x.float()
            x_freq = torch.fft.rfftn(x32, dim=(-3, -2, -1))      # complex64
            band_onehot = self._get_band_onehot((D, H, W), x.device)  # (Bands, D, H, Wr)
            # Mask per channel: (C, D, H, Wr), real fp32.
            mask = torch.einsum("cb,bdhw->cdhw", self.band_gain, band_onehot)
            x_freq = x_freq * mask                                 # broadcast over batch
            x_filt32 = torch.fft.irfftn(x_freq, s=(D, H, W), dim=(-3, -2, -1))
        x_filt = x_filt32.to(x_dtype)

        fused = self.fuse(torch.cat([x, x_filt], dim=1))
        return x + fused
