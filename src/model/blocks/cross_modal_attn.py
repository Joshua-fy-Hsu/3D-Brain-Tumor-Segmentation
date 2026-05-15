"""Cross-modal attention block.

Operates on per-modality stem features. At each spatial location of a coarse
grid, treats the 4 modalities as a 4-token sequence and runs multi-head
self-attention across them so each modality can read the others.

Pipeline:
  - Input mod_feats:  (B, M=4, C=8, D, H, W) at full resolution (128^3)
  - Pool each modality independently to attn_grid (32^3) for tractable attention
  - Reshape to (B*N_pos, M, C) where N_pos = 32^3
  - MultiheadAttention(embed_dim=C, num_heads=2)
  - Reshape back, concat modalities along channel -> (B, M*C, 32^3)
  - Trilinear upsample to full resolution (B, M*C, 128^3)
  - Concat foreground channel -> (B, M*C+1, 128^3)
  - 1x1x1 conv -> (B, out_channels, 128^3) to match enc1 spec
  - Residual: skip-add a 1x1x1 projection of the original input

Memory: attn op is (B*32^3, 4, 8) = 65k tokens of dim 8. ~ MB-scale.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    def __init__(
        self,
        in_channels_residual: int = 5,
        num_modalities: int = 4,
        stem_channels: int = 8,
        out_channels: int = 32,
        num_heads: int = 2,
        attn_grid: int = 32,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.stem_channels = stem_channels
        self.out_channels = out_channels
        self.attn_grid = attn_grid

        assert stem_channels % num_heads == 0, \
            f"stem_channels ({stem_channels}) must be divisible by num_heads ({num_heads})"

        # Manual matmul attention. We don't use nn.MultiheadAttention because
        # PyTorch's SDPA kernel selection trips over head_dim=4 (embed=8 / heads=2)
        # with "invalid configuration argument" on CUDA. With M=4 tokens the
        # softmax+matmul is tiny — kernel selection overhead is pointless anyway.
        self.num_heads = num_heads
        self.head_dim = stem_channels // num_heads
        self.qkv = nn.Linear(stem_channels, 3 * stem_channels, bias=True)
        self.out_proj = nn.Linear(stem_channels, stem_channels, bias=True)
        self.attn_scale = self.head_dim ** -0.5
        self.attn_norm = nn.LayerNorm(stem_channels)

        # Fuse concatenated modality channels + foreground channel into out_channels
        fuse_in = num_modalities * stem_channels + 1   # +1 for foreground
        self.fuse = nn.Sequential(
            nn.Conv3d(fuse_in, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        # Residual projection from raw input (5ch) -> out_channels for skip-add
        self.residual_proj = nn.Conv3d(in_channels_residual, out_channels,
                                       kernel_size=1, bias=False)

    def forward(
        self,
        mod_feats: torch.Tensor,    # (B, M, C, D, H, W)
        fg: torch.Tensor,           # (B, 1, D, H, W)
        x_in: torch.Tensor,         # (B, in_channels_residual, D, H, W) — original input
    ) -> torch.Tensor:
        B, M, C, D, H, W = mod_feats.shape
        assert M == self.num_modalities and C == self.stem_channels

        # Pool to coarse attention grid for tractable sequence length.
        flat = mod_feats.reshape(B * M, C, D, H, W)
        flat_pooled = F.adaptive_avg_pool3d(flat, self.attn_grid)  # (B*M, C, g, g, g)
        g = self.attn_grid
        N = g * g * g

        # Rearrange to (B*N, M, C): for each spatial position there are M tokens.
        # flat_pooled -> (B, M, C, g, g, g) -> (B, g, g, g, M, C) -> (B*N, M, C)
        pooled = flat_pooled.reshape(B, M, C, g, g, g)
        pooled = pooled.permute(0, 3, 4, 5, 1, 2).contiguous()
        seq = pooled.reshape(B * N, M, C)

        # Cross-modal attention across the M-token axis at each spatial position.
        # Pre-norm helps stability under fp16/bf16.
        seq_n = self.attn_norm(seq)
        Bn = seq_n.shape[0]
        nh, Dh = self.num_heads, self.head_dim
        qkv = self.qkv(seq_n).reshape(Bn, M, 3, nh, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                   # (Bn, nh, M, Dh)
        attn_logits = (q @ k.transpose(-2, -1)) * self.attn_scale  # (Bn, nh, M, M)
        attn_w = attn_logits.softmax(dim=-1)
        attn_out = (attn_w @ v).transpose(1, 2).reshape(Bn, M, C)
        attn_out = self.out_proj(attn_out)
        seq = seq + attn_out  # residual within attention

        # Reshape back to (B, M*C, g, g, g)
        seq = seq.reshape(B, g, g, g, M, C).permute(0, 4, 5, 1, 2, 3).contiguous()
        attn_feat = seq.reshape(B, M * C, g, g, g)

        # Upsample to full resolution.
        attn_feat = F.interpolate(attn_feat, size=(D, H, W),
                                  mode="trilinear", align_corners=False)

        # Concat foreground channel and fuse.
        fused_in = torch.cat([attn_feat, fg], dim=1)
        out = self.fuse(fused_in)

        # Residual skip from the original 5-channel input.
        out = out + self.residual_proj(x_in)
        return out
