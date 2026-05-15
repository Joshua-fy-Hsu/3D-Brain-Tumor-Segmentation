"""Hierarchical Spectral Swin stage (Phase 3).

Replaces the `enc4 → bottleneck` path in TransResUNet3D when
`use_spectral_swin=True`:

    enc4 (B, 256, 16, 16, 16)
       │
       ▼
    Stage 1: 2 × SpectralWindowedBlock(dim=256, ws=4, shift={0,2}) at 16³
       │
       ├─► skip_16  (B, 256, 16, 16, 16)        # for dec4 skip-concat
       │
       ▼
    Patch merging: Conv3d(256→512, k=2, s=2) + LayerNorm
       │
       ▼
    Stage 2: 2 × SpectralWindowedBlock(dim=512, ws=4, shift={0,2}) at 8³
       │
       ▼
    deep_8   (B, 512, 8, 8, 8)                  # bottleneck-equivalent → up4

Each `SpectralWindowedBlock` is a Swin block (windowed attention + MLP) plus a
PARALLEL `FrequencyAwareBlock` operating on the same feature map. The two
branches are combined as `out = attn_branch + α · spec_branch`, where α is a
learnable scalar initialized to 0 — so at init the block is identical to a
vanilla Swin block, and α drifts away from 0 only if spectral context is
useful.

Reuses `WindowAttention3D`, `window_partition_3d`, `window_reverse_3d`, and
`DropPath` from [model_transformer.py](../model_transformer.py).

Notes:
  - bf16 autocast mandatory (fp16 overflows window-attention softmax). The
    `FrequencyAwareBlock` already wraps its FFT path in autocast(False).
  - Residual projections are zero-initialized so the Swin branch starts as
    identity (faster early convergence).
  - drop_path is scheduled linearly across all 4 blocks (stage1 + stage2).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from model.model_transformer import (
    DropPath,
    WindowAttention3D,
    window_partition_3d,
    window_reverse_3d,
    WindowedTransformerStage,
)
from model.blocks.frequency import FrequencyAwareBlock


# ---------------------------------------------------------------------------
# Single block: windowed attention + parallel spectral branch + MLP
# ---------------------------------------------------------------------------
class SpectralWindowedBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift: int,
        ffn_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        drop_path: float = 0.0,
        spectral_bands: int = 3,
    ):
        super().__init__()
        self.dim = dim
        self.ws = window_size
        self.shift = shift

        # --- Windowed attention branch (Swin-style) ---
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads,
                                      attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * ffn_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(hidden, dim),
            nn.Dropout(mlp_drop),
        )
        self.drop_path = DropPath(drop_path)

        # --- Parallel spectral branch ---
        # Operates on (B, C, D, H, W). FrequencyAwareBlock has its own residual
        # `x + fused`, so the gated mix `attn + α·spec` ≈ `attn + α·(x + fused)`.
        # At α=0 (init) the block is pure Swin → identity wrt the spectral path.
        self.spectral = FrequencyAwareBlock(channels=dim, num_bands=spectral_bands)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x_cl: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """x_cl: (B, D, H, W, C) channels-last."""
        B, D, H, W, C = x_cl.shape
        shortcut = x_cl

        # ---- Windowed-attention branch (channels-last throughout) ----
        y = self.norm1(x_cl)
        if self.shift > 0:
            y = torch.roll(y, shifts=(-self.shift, -self.shift, -self.shift),
                           dims=(1, 2, 3))
        y_w = window_partition_3d(y, self.ws)                    # (B*nW, ws^3, C)
        y_w = self.attn(y_w, mask=attn_mask if self.shift > 0 else None)
        y = window_reverse_3d(y_w, self.ws, D, H, W)              # (B, D, H, W, C)
        if self.shift > 0:
            y = torch.roll(y, shifts=(self.shift, self.shift, self.shift),
                           dims=(1, 2, 3))

        # ---- Parallel spectral branch (channels-first 5D for FFT) ----
        x_5d = x_cl.permute(0, 4, 1, 2, 3).contiguous()           # (B, C, D, H, W)
        spec_5d = self.spectral(x_5d)
        spec = spec_5d.permute(0, 2, 3, 4, 1).contiguous()         # (B, D, H, W, C)

        # Gated fusion + residual
        x_cl = shortcut + self.drop_path(y + self.alpha * spec)

        # MLP branch
        x_cl = x_cl + self.drop_path(self.mlp(self.norm2(x_cl)))
        return x_cl


# ---------------------------------------------------------------------------
# 2-stage hierarchical spectral Swin
# ---------------------------------------------------------------------------
class SpectralSwinStage(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 512,
        window_size: int = 4,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        depth_stage1: int = 2,
        depth_stage2: int = 2,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        drop_path_max: float = 0.1,
        grid_stage1: Tuple[int, int, int] = (16, 16, 16),
        grid_stage2: Tuple[int, int, int] = (8, 8, 8),
        spectral_bands: int = 3,
    ):
        super().__init__()
        for g in (grid_stage1, grid_stage2):
            assert all(s % window_size == 0 for s in g), \
                f"grid {g} not divisible by window {window_size}"
        assert in_channels % num_heads == 0
        assert out_channels % num_heads == 0

        self.window_size = window_size
        self.depth_stage1 = depth_stage1
        self.depth_stage2 = depth_stage2

        total_depth = depth_stage1 + depth_stage2
        dpr = [drop_path_max * i / max(total_depth - 1, 1)
               for i in range(total_depth)]

        # Stage 1 — operates at grid_stage1, in_channels
        self.stage1 = nn.ModuleList([
            SpectralWindowedBlock(
                dim=in_channels, num_heads=num_heads,
                window_size=window_size,
                shift=0 if (i % 2 == 0) else window_size // 2,
                ffn_ratio=ffn_ratio,
                attn_drop=attn_drop, proj_drop=proj_drop, mlp_drop=mlp_drop,
                drop_path=dpr[i],
                spectral_bands=spectral_bands,
            )
            for i in range(depth_stage1)
        ])

        # Patch merging: stride-2 conv halves spatial, doubles channels.
        self.patch_merge = nn.Conv3d(in_channels, out_channels,
                                     kernel_size=2, stride=2, bias=True)
        self.norm_merge = nn.LayerNorm(out_channels)

        # Stage 2 — operates at grid_stage2, out_channels
        self.stage2 = nn.ModuleList([
            SpectralWindowedBlock(
                dim=out_channels, num_heads=num_heads,
                window_size=window_size,
                shift=0 if (i % 2 == 0) else window_size // 2,
                ffn_ratio=ffn_ratio,
                attn_drop=attn_drop, proj_drop=proj_drop, mlp_drop=mlp_drop,
                drop_path=dpr[depth_stage1 + i],
                spectral_bands=spectral_bands,
            )
            for i in range(depth_stage2)
        ])

        # Pre-build SW-MSA attention masks (one per stage).
        self.register_buffer("attn_mask_1",
                             WindowedTransformerStage._build_attn_mask(
                                 grid_stage1, window_size, shift=window_size // 2),
                             persistent=False)
        self.register_buffer("attn_mask_2",
                             WindowedTransformerStage._build_attn_mask(
                                 grid_stage2, window_size, shift=window_size // 2),
                             persistent=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        # Zero-init Swin residual projections so each block starts as identity.
        for blk_list in (self.stage1, self.stage2):
            for blk in blk_list:
                nn.init.zeros_(blk.attn.proj.weight)
                if blk.attn.proj.bias is not None:
                    nn.init.zeros_(blk.attn.proj.bias)
                nn.init.zeros_(blk.mlp[3].weight)
                if blk.mlp[3].bias is not None:
                    nn.init.zeros_(blk.mlp[3].bias)
        # Patch-merge conv keeps Kaiming init (downsampling op needs to learn).
        nn.init.kaiming_normal_(self.patch_merge.weight, mode="fan_out",
                                nonlinearity="leaky_relu")
        if self.patch_merge.bias is not None:
            nn.init.zeros_(self.patch_merge.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, in_channels, D1, H1, W1) at grid_stage1.
        Returns (skip_16, deep_8): the Stage-1 refined feature for the dec4
        skip, and the Stage-2 deep feature that replaces the bottleneck."""
        B = x.shape[0]

        # ---- Stage 1 (channels-last) ----
        x_cl = x.permute(0, 2, 3, 4, 1).contiguous()              # (B, D, H, W, C)
        for blk in self.stage1:
            mask = self.attn_mask_1 if blk.shift > 0 else None
            x_cl = blk(x_cl, mask)
        skip = x_cl.permute(0, 4, 1, 2, 3).contiguous()            # (B, C_in, D1, H1, W1)

        # ---- Patch merging: stride-2 conv + LayerNorm ----
        merged = self.patch_merge(skip)                            # (B, C_out, D2, H2, W2)
        merged_cl = merged.permute(0, 2, 3, 4, 1).contiguous()
        merged_cl = self.norm_merge(merged_cl)

        # ---- Stage 2 ----
        x_cl2 = merged_cl
        for blk in self.stage2:
            mask = self.attn_mask_2 if blk.shift > 0 else None
            x_cl2 = blk(x_cl2, mask)
        deep = x_cl2.permute(0, 4, 1, 2, 3).contiguous()           # (B, C_out, D2, H2, W2)

        return skip, deep
