"""TC-Refine head (full_lean).

The whole AURAS ablation chain never beats the plain base_cnn on TC (tumor
core = NCR ∪ ET); the deficit vs the unet3d baseline is entirely TC. Every
shared-decoder component spends its capacity on ET. This head gives the
NCR/ET-defining TC boundary a dedicated, deep-supervised pathway, then
folds it back into the final softmax logits as a gated residual:

    tc1 = head1(d1)          # (B, 1, 128^3)  full-res TC logit
    tc2 = head2(d2)          # (B, 1,  64^3)  1/2-res TC logit (deep-sup only)
    seg[:, [1, 3]] += gate * tc1   # NCR (1) and ET (3) channels only

``gate`` is a learnable scalar initialised to 0, so at init the model is
bit-identical to the same config without this head (the proven full
backbone). The TC gradient flows into the shared decoder by design — the
gate warmup in train_variant.py is the safety mechanism, not detachment.
``tc1``/``tc2`` are also returned for the deep-supervised TC loss
(see TCRefineLoss in training/losses.py).

Channels 0 (background) and 2 (ED) are deliberately left untouched: TC by
definition excludes ED, and the residual must not perturb the regions
AURAS already wins (ET) or ties (WT) beyond what TC supervision implies.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _tc_branch(in_ch: int, mid: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, mid, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm3d(mid),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Conv3d(mid, 1, kernel_size=1, bias=True),
    )


class TCRefineHead(nn.Module):
    """Dedicated TC pathway off d1 (full-res) and d2 (1/2-res).

    Args:
        in_ch_d1: channels of the d1 decoder feature (= base_filters).
        in_ch_d2: channels of the d2 decoder feature (= base_filters * 2).
        mid:      bottleneck width of each branch.
    """

    # Softmax channels that belong to TC (= NCR ∪ ET).
    TC_CHANNELS = (1, 3)

    def __init__(self, in_ch_d1: int, in_ch_d2: int, mid: int = 16):
        super().__init__()
        self.head1 = _tc_branch(in_ch_d1, mid)
        self.head2 = _tc_branch(in_ch_d2, mid)
        # Learnable gate: init 0 → head starts as a no-op residual.
        self.gate = nn.Parameter(torch.zeros(1))
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

    def fuse(self, seg_logits: torch.Tensor, tc1: torch.Tensor) -> torch.Tensor:
        """Add ``gate * tc1`` to the NCR (1) and ET (3) logit channels only.

        seg_logits: (B, num_classes, D, H, W) — same spatial size as tc1.
        tc1:        (B, 1, D, H, W) full-res TC logit.
        """
        out = seg_logits.clone()
        delta = self.gate * tc1.squeeze(1)          # (B, D, H, W)
        for c in self.TC_CHANNELS:
            out[:, c] = out[:, c] + delta
        return out
