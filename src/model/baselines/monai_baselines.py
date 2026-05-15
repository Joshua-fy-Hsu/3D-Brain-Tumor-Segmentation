"""Phase 7 external baselines — thin wrappers around MONAI reference nets.

These exist so the ablation table has fair, *standard-config* comparisons
against the AURAS family:

  - ``swinunetr`` : Swin UNETR — transformer, de-facto BraTS transformer SOTA
  - ``segresnet`` : SegResNet  — CNN, BraTS-2018 winner
  - ``unet3d``    : 3D U-Net   — the universal lower-bound baseline

Contract (kept identical to ``ResUnet3D`` / ``TransResUNet3D`` so the generic
``train_variant.py`` / ``evaluate_variant.py`` treat a baseline like any other
registry variant):

  - 5-channel input (T1, T1CE, T2, FLAIR, foreground), 128³ patches
  - 4-channel softmax logits out (label space {0=BG,1=NCR,2=ED,3=ET})
  - **train mode** → returns a 1-tuple ``(logits,)`` so
    ``RegionWiseDiceFocalLoss.forward`` (which indexes ``inputs_list[0/1/2]``
    for deep supervision) consumes it as a single-resolution loss with no DS
    term. Baselines are run in their published, no-deep-supervision config —
    that is the fair comparison reviewers expect.
  - **eval mode** → returns the plain ``logits`` tensor, so
    ``DictToSegAdapter`` / sliding-window inference see a tensor unchanged.

MONAI is imported lazily *inside each factory* so importing the registry
(hence the trainer CLI) never requires MONAI. It stays a hard dependency of
*running* a baseline only — mirroring the rest of the codebase, where MONAI
is a hard dep of evaluation but not of registry import.
"""
from __future__ import annotations

import inspect
from typing import Sequence

import torch.nn as nn


class _SegWrapper(nn.Module):
    """Adapt a plain ``(B,C,D,H,W)``-logits net to the project's forward
    contract: a 1-tuple in train mode (deep-supervision loss expects an
    indexable list), a bare tensor in eval mode."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        logits = self.net(x)
        if self.training:
            return (logits,)
        return logits


def build_swinunetr(
    in_channels: int = 5,
    num_classes: int = 4,
    feature_size: int = 48,
    img_size: Sequence[int] = (128, 128, 128),
    use_checkpoint: bool = False,
    **_ignore,
) -> nn.Module:
    """Swin UNETR (transformer baseline). ``feature_size=48`` is the config
    used in the original Swin UNETR BraTS submission.

    ``img_size`` was a required arg in MONAI < 1.3, then deprecated/removed in
    later versions — we only pass it if the installed signature still accepts
    it, with a TypeError fallback for the deprecated-and-rejected case.
    """
    from monai.networks.nets import SwinUNETR

    params = inspect.signature(SwinUNETR.__init__).parameters
    kwargs = dict(
        in_channels=in_channels,
        out_channels=num_classes,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
    )
    if "spatial_dims" in params:
        kwargs["spatial_dims"] = 3
    if "img_size" in params:
        try:
            net = SwinUNETR(img_size=img_size, **kwargs)
        except TypeError:
            # signature lists img_size but the constructor rejects it
            # (newer MONAI where it is fully removed/deprecated-error).
            net = SwinUNETR(**kwargs)
    else:
        net = SwinUNETR(**kwargs)
    return _SegWrapper(net)


def build_segresnet(
    in_channels: int = 5,
    num_classes: int = 4,
    init_filters: int = 32,
    blocks_down: Sequence[int] = (1, 2, 2, 4),
    blocks_up: Sequence[int] = (1, 1, 1),
    dropout_prob: float = 0.2,
    **_ignore,
) -> nn.Module:
    """SegResNet (CNN baseline). These are the canonical BraTS settings from
    the MONAI BraTS tutorial / the SegResNet paper (init_filters=32,
    down=(1,2,2,4), up=(1,1,1), dropout=0.2). ``dropout_prob=0.2`` also lets
    the eval pipeline's MC-Dropout pass run (``model_has_dropout`` is True)."""
    from monai.networks.nets import SegResNet

    net = SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
        init_filters=init_filters,
        blocks_down=tuple(blocks_down),
        blocks_up=tuple(blocks_up),
        dropout_prob=dropout_prob,
    )
    return _SegWrapper(net)


def build_unet3d(
    in_channels: int = 5,
    num_classes: int = 4,
    features: Sequence[int] = (32, 64, 128, 256, 512, 32),
    dropout: float = 0.0,
    **_ignore,
) -> nn.Module:
    """Vanilla 3D U-Net (lower-bound baseline) via MONAI ``BasicUNet`` —
    no residual units, the truthful Çiçek-style U-Net.

    ``features`` is the (enc1..enc5, final) channel ladder. We use
    32→64→128→256→512 so the width matches ``base_cnn``'s encoder ladder —
    the fairest direct "what does the plain U-Net get at the same capacity"
    contrast, isolating architecture from width.
    """
    from monai.networks.nets import BasicUNet

    net = BasicUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
        features=tuple(features),
        dropout=dropout,
    )
    return _SegWrapper(net)
