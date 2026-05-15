import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.model import ResidualBlock


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


# ---------------------------------------------------------------------------
# Relative position bias — one learned bias per (Δd, Δh, Δw) offset between
# token pairs at the bottleneck (8x8x8 grid by default).
# ---------------------------------------------------------------------------
class RelativePositionBias3D(nn.Module):
    def __init__(self, grid_size=(8, 8, 8), num_heads: int = 8):
        super().__init__()
        D, H, W = grid_size
        self.grid_size = grid_size
        self.num_heads = num_heads
        # 2d-1 unique offsets per axis -> table of size (2D-1)*(2H-1)*(2W-1)
        n_rel = (2 * D - 1) * (2 * H - 1) * (2 * W - 1)
        self.bias_table = nn.Parameter(torch.zeros(n_rel, num_heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)
        self.register_buffer("rel_index", self._build_index(D, H, W), persistent=False)

    @staticmethod
    def _build_index(D, H, W):
        coords = torch.stack(torch.meshgrid(
            torch.arange(D), torch.arange(H), torch.arange(W), indexing="ij"
        )).flatten(1)  # (3, N)
        rel = coords[:, :, None] - coords[:, None, :]  # (3, N, N)
        rel[0] += D - 1; rel[1] += H - 1; rel[2] += W - 1
        idx = (rel[0] * (2 * H - 1) * (2 * W - 1)
               + rel[1] * (2 * W - 1)
               + rel[2])
        return idx  # (N, N)

    def forward(self):
        # (N, N, num_heads) -> (num_heads, N, N) for additive attention bias
        bias = self.bias_table[self.rel_index]
        return bias.permute(2, 0, 1).contiguous()


# ---------------------------------------------------------------------------
# Self-attention with relative position bias + attention dropout
# ---------------------------------------------------------------------------
class RelAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rel_bias: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        if rel_bias is not None:
            attn = attn + rel_bias.unsqueeze(0)  # broadcast over batch
        attn = attn - attn.amax(dim=-1, keepdim=True).detach()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ---------------------------------------------------------------------------
# Pre-norm transformer block with DropPath and dropout in attn + MLP.
# Output projections are zero-initialised so the block starts as identity.
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_ratio: float,
                 attn_drop: float = 0.1, proj_drop: float = 0.1,
                 mlp_drop: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RelAttention(dim, heads, attn_drop=attn_drop, proj_drop=proj_drop)
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

    def forward(self, x: torch.Tensor, rel_bias: torch.Tensor = None) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), rel_bias=rel_bias))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Bottleneck transformer — grid-shaped tokens (D x H x W), shared rel-pos
# bias, optional token dropout for robustness.
# ---------------------------------------------------------------------------
class BottleneckTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, ffn_ratio: float,
                 grid_size=(8, 8, 8),
                 attn_drop: float = 0.1, proj_drop: float = 0.1,
                 mlp_drop: float = 0.1, drop_path_max: float = 0.1,
                 token_drop: float = 0.1):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        self.token_drop = token_drop

        # Linearly-scaled per-block stochastic depth (0 -> drop_path_max)
        dpr = [drop_path_max * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, ffn_ratio,
                             attn_drop=attn_drop, proj_drop=proj_drop,
                             mlp_drop=mlp_drop, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.rel_pos = RelativePositionBias3D(grid_size, num_heads=heads)

        self._init_weights()

    def _init_weights(self):
        # Standard ViT init for linears, then zero-init the residual output
        # projections so the block starts as identity (faster convergence,
        # better stability).
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for blk in self.blocks:
            nn.init.zeros_(blk.attn.proj.weight)
            if blk.attn.proj.bias is not None:
                nn.init.zeros_(blk.attn.proj.bias)
            # MLP last linear (idx 3 in the Sequential)
            nn.init.zeros_(blk.mlp[3].weight)
            if blk.mlp[3].bias is not None:
                nn.init.zeros_(blk.mlp[3].bias)

    def _drop_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """During training only, zero-out a fraction of tokens. Forces
        attention to be robust to missing context."""
        if not self.training or self.token_drop == 0.0:
            return tokens
        B, N, C = tokens.shape
        keep = 1.0 - self.token_drop
        mask = tokens.new_empty(B, N, 1).bernoulli_(keep)
        return tokens * mask / keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self._drop_tokens(tokens)
        rel_bias = self.rel_pos()

        # No checkpointing here — at 8^3=512 tokens activations are tiny
        # (~50 MB), so the 2x compute cost wasn't worth it.
        for blk in self.blocks:
            tokens = blk(tokens, rel_bias)

        x = tokens.transpose(1, 2).reshape(B, C, D, H, W)
        return x


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


# ---------------------------------------------------------------------------
# ResUnet3DTransformer — same skeleton as before, plus decoder Dropout3d.
# ---------------------------------------------------------------------------
class ResUnet3DTransformer(nn.Module):
    def __init__(self, in_channels: int = 5, num_classes: int = 4,
                 base_filters: int = 32,
                 transformer_depth: int = 2,
                 transformer_heads: int = 8,
                 transformer_ffn_ratio: float = 2.0,
                 # Windowed transformer at enc4 (1/8 res, 16^3 tokens, 4^3 windows)
                 use_windowed_stage: bool = True,
                 windowed_depth: int = 2,
                 windowed_heads: int = 8,
                 windowed_window_size: int = 4,
                 windowed_ffn_ratio: float = 4.0,
                 # Output: "softmax" -> num_classes channels; "sigmoid" -> 3
                 # channels predicting ET/TC/WT independently (nnU-Net style).
                 output_mode: str = "sigmoid",
                 # Regularization knobs. decoder_dropout is the legacy single
                 # value; if either *_inner / *_final is None it falls back to
                 # decoder_dropout, so existing callers keep working.
                 decoder_dropout: float = 0.1,
                 decoder_dropout_inner: float = None,
                 decoder_dropout_final: float = None,
                 attn_drop: float = 0.1,
                 proj_drop: float = 0.1,
                 mlp_drop: float = 0.1,
                 drop_path_max: float = 0.1,
                 token_drop: float = 0.1):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base_filters, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(base_filters),
            nn.LeakyReLU(inplace=True)
        )
        self.enc2 = ResidualBlock(base_filters, base_filters * 2, stride=2)
        self.enc3 = ResidualBlock(base_filters * 2, base_filters * 4, stride=2)
        self.enc4 = ResidualBlock(base_filters * 4, base_filters * 8, stride=2)

        # Optional windowed transformer at 1/8 resolution. enc4 output is
        # (B, base*8, 16, 16, 16); window=4 gives 4^3=64 windows of 64 tokens.
        self.use_windowed_stage = use_windowed_stage
        if use_windowed_stage:
            self.windowed_stage = WindowedTransformerStage(
                dim=base_filters * 8,
                depth=windowed_depth,
                num_heads=windowed_heads,
                grid_size=(16, 16, 16),
                window_size=windowed_window_size,
                ffn_ratio=windowed_ffn_ratio,
                attn_drop=attn_drop, proj_drop=proj_drop, mlp_drop=mlp_drop,
                drop_path_max=drop_path_max,
            )

        self.bottleneck = ResidualBlock(base_filters * 8, base_filters * 16, stride=2)
        self.bottleneck_transformer = BottleneckTransformer(
            dim=base_filters * 16,
            depth=transformer_depth,
            heads=transformer_heads,
            ffn_ratio=transformer_ffn_ratio,
            grid_size=(8, 8, 8),
            attn_drop=attn_drop, proj_drop=proj_drop, mlp_drop=mlp_drop,
            drop_path_max=drop_path_max, token_drop=token_drop,
        )

        self.output_mode = output_mode
        out_ch = 3 if output_mode == "sigmoid" else num_classes

        # Decoder with Dropout3d. drop1 keeps a non-zero rate so the model
        # advertises dropout for MC Dropout at inference; inner stages can be
        # zeroed to preserve boundary cues NCR depends on.
        p_inner = decoder_dropout if decoder_dropout_inner is None else decoder_dropout_inner
        p_final = decoder_dropout if decoder_dropout_final is None else decoder_dropout_final

        self.up4 = nn.ConvTranspose3d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(base_filters * 16, base_filters * 8)
        self.drop4 = nn.Dropout3d(p_inner)

        self.up3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(base_filters * 8, base_filters * 4)
        self.drop3 = nn.Dropout3d(p_inner)
        self.ds2_cls = nn.Conv3d(base_filters * 4, out_ch, kernel_size=1)

        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(base_filters * 4, base_filters * 2)
        self.drop2 = nn.Dropout3d(p_inner)
        self.ds1_cls = nn.Conv3d(base_filters * 2, out_ch, kernel_size=1)

        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(base_filters * 2, base_filters)
        self.drop1 = nn.Dropout3d(p_final)
        self.final_conv = nn.Conv3d(base_filters, out_ch, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None: nn.init.constant_(m.weight, 1)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
        # The transformer initialises its own Linear/LayerNorm layers in
        # BottleneckTransformer._init_weights — running over them here would
        # un-do the zero-init residual trick.

    # ------------------------------------------------------------------
    # Helpers for layer-wise LR decay in the trainer.
    # Returns three parameter groups: cnn, transformer, decoder (all CNN
    # decoder params are bundled with cnn). Only used by the trainer.
    # ------------------------------------------------------------------
    def parameter_groups(self):
        transformer_params, cnn_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if ("bottleneck_transformer" in name) or ("windowed_stage" in name):
                transformer_params.append(p)
            else:
                cnn_params.append(p)
        return cnn_params, transformer_params

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        if self.use_windowed_stage:
            x4 = self.windowed_stage(x4)  # refines enc4 features and is also the dec4 skip
        x_bot = self.bottleneck(x4)
        x_bot = self.bottleneck_transformer(x_bot)

        d4 = self.up4(x_bot)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.drop4(self.dec4(d4))

        d3 = self.up3(d4)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.drop3(self.dec3(d3))
        out_ds2 = self.ds2_cls(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.drop2(self.dec2(d2))
        out_ds1 = self.ds1_cls(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.drop1(self.dec1(d1))

        out_final = self.final_conv(d1)

        if self.training:
            return out_final, out_ds1, out_ds2
        else:
            return out_final
