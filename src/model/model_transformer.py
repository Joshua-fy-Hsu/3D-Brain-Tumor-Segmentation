"""Windowed (Swin-style) 3D attention building blocks.

`DropPath`, `window_partition_3d` / `window_reverse_3d`, `WindowAttention3D`,
`WindowedBlock`, and `WindowedTransformerStage` are the reusable primitives for
the spectral-Swin stage of the `full` model — see
[blocks/spectral_swin.py](blocks/spectral_swin.py), which imports them.
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Stochastic Depth (DropPath) — drops the entire residual contribution with
# probability p_drop during training. Per-sample, not per-element.
# ---------------------------------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        # Per-sample mask broadcast over remaining dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


# ===========================================================================
# Windowed attention at 1/8 resolution (Swin-style)
# ===========================================================================

def window_partition_3d(x: torch.Tensor, ws: int) -> torch.Tensor:
    """(B, D, H, W, C) -> (B*nW, ws^3, C). D/H/W must be divisible by ws."""
    B, D, H, W, C = x.shape
    x = x.view(B, D // ws, ws, H // ws, ws, W // ws, ws, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(-1, ws * ws * ws, C)


def window_reverse_3d(windows: torch.Tensor, ws: int, D: int, H: int, W: int) -> torch.Tensor:
    """(B*nW, ws^3, C) -> (B, D, H, W, C)."""
    B = int(windows.shape[0] / ((D * H * W) // (ws ** 3)))
    x = windows.view(B, D // ws, H // ws, W // ws, ws, ws, ws, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(B, D, H, W, -1)


class WindowAttention3D(nn.Module):
    """Multi-head self-attention inside a 3D window with relative position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 attn_drop: float = 0.1, proj_drop: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.ws = window_size

        # Relative bias table inside a window
        n_rel = (2 * window_size - 1) ** 3
        self.bias_table = nn.Parameter(torch.zeros(n_rel, num_heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)
        self.register_buffer("rel_index",
                             self._build_rel_index(window_size), persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    @staticmethod
    def _build_rel_index(ws: int) -> torch.Tensor:
        coords = torch.stack(torch.meshgrid(
            torch.arange(ws), torch.arange(ws), torch.arange(ws), indexing="ij"
        )).flatten(1)  # (3, ws^3)
        rel = coords[:, :, None] - coords[:, None, :]
        rel[0] += ws - 1; rel[1] += ws - 1; rel[2] += ws - 1
        return (rel[0] * (2 * ws - 1) ** 2
                + rel[1] * (2 * ws - 1)
                + rel[2])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B*nW, N, C) where N = ws^3
        Bn, N, C = x.shape
        qkv = self.qkv(x).reshape(Bn, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        bias = self.bias_table[self.rel_index].permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            # mask: (nW, N, N) — apply per window
            nW = mask.shape[0]
            attn = attn.view(Bn // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(Bn, self.num_heads, N, N)
        attn = attn - attn.amax(dim=-1, keepdim=True).detach()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(Bn, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class WindowedBlock(nn.Module):
    """Pre-norm windowed transformer block with optional cyclic shift."""

    def __init__(self, dim: int, num_heads: int, window_size: int, shift: int,
                 ffn_ratio: float = 4.0, attn_drop: float = 0.1,
                 proj_drop: float = 0.1, mlp_drop: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.dim = dim
        self.ws = window_size
        self.shift = shift  # 0 = no shift, ws//2 = shifted
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

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, D, H, W, C)
        B, D, H, W, C = x.shape
        shortcut = x

        x = self.norm1(x)
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift, -self.shift), dims=(1, 2, 3))

        x_w = window_partition_3d(x, self.ws)        # (B*nW, ws^3, C)
        x_w = self.attn(x_w, mask=attn_mask if self.shift > 0 else None)
        x = window_reverse_3d(x_w, self.ws, D, H, W)  # (B, D, H, W, C)

        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift, self.shift), dims=(1, 2, 3))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WindowedTransformerStage(nn.Module):
    """Stack of windowed transformer blocks with alternating shift.
    Operates on (B, C, D, H, W) tensors; converts to channels-last internally."""

    def __init__(self, dim: int, depth: int, num_heads: int,
                 grid_size=(16, 16, 16), window_size: int = 4,
                 ffn_ratio: float = 4.0, attn_drop: float = 0.1,
                 proj_drop: float = 0.1, mlp_drop: float = 0.1,
                 drop_path_max: float = 0.1):
        super().__init__()
        assert all(s % window_size == 0 for s in grid_size), \
            f"grid {grid_size} not divisible by window {window_size}"
        self.dim = dim
        self.ws = window_size
        self.grid_size = grid_size

        dpr = [drop_path_max * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList([
            WindowedBlock(
                dim, num_heads, window_size,
                shift=0 if (i % 2 == 0) else window_size // 2,
                ffn_ratio=ffn_ratio,
                attn_drop=attn_drop, proj_drop=proj_drop,
                mlp_drop=mlp_drop, drop_path=dpr[i],
            )
            for i in range(depth)
        ])

        # Pre-compute SW-MSA attention mask once.
        self.register_buffer("attn_mask",
                             self._build_attn_mask(grid_size, window_size,
                                                   shift=window_size // 2),
                             persistent=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init residual projections so the stage starts as identity.
        for blk in self.blocks:
            nn.init.zeros_(blk.attn.proj.weight)
            if blk.attn.proj.bias is not None:
                nn.init.zeros_(blk.attn.proj.bias)
            nn.init.zeros_(blk.mlp[3].weight)
            if blk.mlp[3].bias is not None:
                nn.init.zeros_(blk.mlp[3].bias)

    @staticmethod
    def _build_attn_mask(grid_size, ws, shift):
        D, H, W = grid_size
        img_mask = torch.zeros(1, D, H, W, 1)
        slices = [slice(0, -ws), slice(-ws, -shift), slice(-shift, None)]
        cnt = 0
        for d in slices:
            for h in slices:
                for w in slices:
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
        # (nW, ws^3, 1)
        windows = window_partition_3d(img_mask, ws).squeeze(-1)
        # (nW, ws^3, ws^3) — 0 if same region, -inf otherwise
        mask = windows.unsqueeze(1) - windows.unsqueeze(2)
        mask = mask.masked_fill(mask != 0, float(-100.0))
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W) -> (B, D, H, W, C)
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        for blk in self.blocks:
            mask = self.attn_mask if blk.shift > 0 else None
            if self.training:
                x = checkpoint(blk, x, mask, use_reentrant=False)
            else:
                x = blk(x, mask)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x
