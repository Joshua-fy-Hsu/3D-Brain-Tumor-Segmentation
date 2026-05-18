"""HybridUNet3D — a deliberately minimal CNN+Transformer for 3D BraTS.

This is a clean-slate model whose ONLY goal is to beat the `unet3d`
lower-bound baseline on the same harness. It is a TransBTS-style hybrid
(Wang et al., MICCAI 2021): a 3D residual-conv encoder, a small transformer
at the lowest-resolution bottleneck, and a 3D residual-conv decoder with
skip connections and deep supervision.

Design rules (see docs/plan — `hybrid`):
  - NO imports from `model/` or `training/`. Fully standalone.
  - Plain residual convs (InstanceNorm + LeakyReLU(0.01)), the nnU-Net idiom.
  - Transformer only at the 8^3 bottleneck (512 global tokens) — cheap and
    fp16-stable (no windowed-attention softmax to overflow).
  - 4-class softmax head → standard argmax decode, structurally avoiding the
    sigmoid-head TC-leak that sank the AURAS family.
  - Two Dropout3d(0.10) layers in the decoder → the eval pipeline's
    MC-Dropout pass (`U.model_has_dropout`) runs for free → uncertainty maps.
  - Deep supervision: train-mode forward returns (final, ds1, ds2) so the
    project's `RegionWiseDiceFocalLoss.forward` consumes inputs_list[0/1/2].

Forward contract (matches the generic trainer/evaluator):
  - train mode → tuple (final[B,4,128^3], ds1[B,4,64^3], ds2[B,4,32^3])
  - eval  mode → bare tensor final[B,4,128^3]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def conv_norm_act(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    """Conv3d(k3) → InstanceNorm3d → LeakyReLU(0.01). The nnU-Net idiom."""
    return nn.Sequential(
        nn.Conv3d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.InstanceNorm3d(cout, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
    )


class ResidualBlock(nn.Module):
    """Two conv_norm_act with a (projected) identity skip; LeakyReLU after add."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = conv_norm_act(cin, cout)
        self.conv2 = nn.Sequential(
            nn.Conv3d(cout, cout, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(cout, affine=True),
        )
        if cin == cout:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv3d(cin, cout, kernel_size=1, bias=False),
                nn.InstanceNorm3d(cout, affine=True),
            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class TransformerBottleneck(nn.Module):
    """Standard pre-norm transformer encoder applied at the 3D bottleneck.

    A 3D learned positional embedding is added before flattening to tokens,
    so the block is robust to a bottleneck spatial size other than the
    default (interpolated if it differs — e.g. non-128^3 inference patches).
    """

    def __init__(
        self,
        dim: int,
        depth: int = 4,
        heads: int = 8,
        mlp_ratio: int = 4,
        default_size: int = 8,
    ):
        super().__init__()
        self.dim = dim
        # (1, C, D, H, W) positional embedding — interpolated if the runtime
        # bottleneck size differs from default_size.
        self.pos = nn.Parameter(
            torch.zeros(1, dim, default_size, default_size, default_size)
        )
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_ratio * dim,
            dropout=0.0,                 # matches config.ATTN_DROP = 0
            activation="gelu",
            batch_first=True,
            norm_first=True,             # pre-norm: fp16-stable
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        pos = self.pos
        if pos.shape[2:] != (D, H, W):
            pos = F.interpolate(
                pos, size=(D, H, W), mode="trilinear", align_corners=False
            )
        x = x + pos
        tokens = x.flatten(2).transpose(1, 2)        # (B, N, C)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return tokens.transpose(1, 2).reshape(B, C, D, H, W)


# ---------------------------------------------------------------------------
# HybridUNet3D
# ---------------------------------------------------------------------------
class HybridUNet3D(nn.Module):
    """TransBTS-style residual CNN encoder + transformer bottleneck + CNN
    decoder with deep supervision and MC-Dropout.

    Args:
        in_channels: 5 (T1,T1CE,T2,FLAIR,foreground).
        num_classes: 4 (BG/NCR/ED/ET) — softmax head.
        base_filters: encoder width base. Ladder = f, 2f, 4f, 8f, 16f
            (32→64→128→256→512 at f=32, matching `unet3d`/`base_cnn`).
        tx_depth / tx_heads: transformer encoder layers / heads at the
            bottleneck.
        decoder_dropout: Dropout3d prob in the last two decoder stages
            (= config.DECODER_DROPOUT). >0 enables eval MC-Dropout.
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 4,
        base_filters: int = 32,
        tx_depth: int = 4,
        tx_heads: int = 8,
        decoder_dropout: float = 0.10,
    ):
        super().__init__()
        f = base_filters
        c0, c1, c2, c3, c4 = f, 2 * f, 4 * f, 8 * f, 16 * f  # 32,64,128,256,512

        # ---- Encoder ----
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c0, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(c0, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            ResidualBlock(c0, c0),
        )
        self.down0 = nn.Sequential(conv_norm_act(c0, c1, stride=2), ResidualBlock(c1, c1))
        self.down1 = nn.Sequential(conv_norm_act(c1, c2, stride=2), ResidualBlock(c2, c2))
        self.down2 = nn.Sequential(conv_norm_act(c2, c3, stride=2), ResidualBlock(c3, c3))
        self.down3 = nn.Sequential(conv_norm_act(c3, c4, stride=2), ResidualBlock(c4, c4))

        # ---- Transformer bottleneck (8^3 = 512 tokens at 128^3 input) ----
        self.bottleneck_tx = TransformerBottleneck(
            dim=c4, depth=tx_depth, heads=tx_heads, default_size=8
        )

        # ---- Decoder ----
        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(c3 + c3, c3)        # + skip3
        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(c2 + c2, c2)        # + skip2
        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(c1 + c1, c1)        # + skip1
        self.up0 = nn.ConvTranspose3d(c1, c0, kernel_size=2, stride=2)
        self.dec0 = ResidualBlock(c0 + c0, c0)        # + skip0

        # MC-Dropout in the last two decoder stages.
        self.drop1 = nn.Dropout3d(decoder_dropout)
        self.drop0 = nn.Dropout3d(decoder_dropout)

        # ---- Heads (deep supervision) ----
        self.head_final = nn.Conv3d(c0, num_classes, kernel_size=1)   # 128^3
        self.head_ds1 = nn.Conv3d(c1, num_classes, kernel_size=1)     # 64^3
        self.head_ds2 = nn.Conv3d(c2, num_classes, kernel_size=1)     # 32^3

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm3d) and m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # -- separate optimizer groups (kept for completeness; the hybrid preset
    #    uses a single group, so this stays unused but harmless) --
    def parameter_groups(self):
        tx_ids = {id(p) for p in self.bottleneck_tx.parameters()}
        tx_params = [p for p in self.parameters() if id(p) in tx_ids]
        cnn_params = [p for p in self.parameters() if id(p) not in tx_ids]
        return cnn_params, tx_params

    def forward(self, x: torch.Tensor):
        # Encoder
        s0 = self.stem(x)        # c0 @ 128^3
        s1 = self.down0(s0)      # c1 @ 64^3
        s2 = self.down1(s1)      # c2 @ 32^3
        s3 = self.down2(s2)      # c3 @ 16^3
        b = self.down3(s3)       # c4 @ 8^3

        # Transformer bottleneck (residual)
        b = b + self.bottleneck_tx(b)

        # Decoder with skips
        d3 = self.dec3(torch.cat([self.up3(b), s3], dim=1))    # c3 @ 16^3
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))   # c2 @ 32^3
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))   # c1 @ 64^3
        d1 = self.drop1(d1)
        d0 = self.dec0(torch.cat([self.up0(d1), s0], dim=1))   # c0 @ 128^3
        d0 = self.drop0(d0)

        final = self.head_final(d0)        # (B,4,128^3)
        if not self.training:
            return final
        ds1 = self.head_ds1(d1)            # (B,4,64^3)
        ds2 = self.head_ds2(d2)            # (B,4,32^3)
        return final, ds1, ds2


# ---------------------------------------------------------------------------
# CUDA sanity (mirrors the phase scripts' sanity step)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sanity] device={dev}")
    model = HybridUNet3D(in_channels=5, num_classes=4, base_filters=32).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[sanity] params: {n_params:,}")

    x = torch.randn(2, 5, 128, 128, 128, device=dev)

    # train mode → 3-tuple with deep-supervision shapes
    model.train()
    final, ds1, ds2 = model(x)
    assert final.shape == (2, 4, 128, 128, 128), final.shape
    assert ds1.shape == (2, 4, 64, 64, 64), ds1.shape
    assert ds2.shape == (2, 4, 32, 32, 32), ds2.shape
    print(f"[sanity] train shapes ok: {tuple(final.shape)}, "
          f"{tuple(ds1.shape)}, {tuple(ds2.shape)}")

    # backward → finite grads
    loss = final.float().mean() + 0.5 * ds1.float().mean() + 0.25 * ds2.float().mean()
    loss.backward()
    g = [p.grad for p in model.parameters() if p.grad is not None]
    assert g and all(torch.isfinite(gi).all() for gi in g), "non-finite grads"
    print(f"[sanity] backward ok: {len(g)} tensors with finite grads")

    # eval mode → bare tensor
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert torch.is_tensor(out) and out.shape == (2, 4, 128, 128, 128), out.shape
    print(f"[sanity] eval shape ok: {tuple(out.shape)}")

    # MC-Dropout detectable
    has_drop = any(isinstance(m, torch.nn.Dropout3d) for m in model.modules())
    assert has_drop, "no Dropout3d found — MC-Dropout would be skipped"
    print("[sanity] Dropout3d present → eval MC-Dropout will run")
    print("[sanity] ALL OK")
