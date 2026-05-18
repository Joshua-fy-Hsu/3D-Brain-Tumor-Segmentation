"""TransResUNet3D — single configurable backbone for the Phase 1-5 ablation.

Constructor flags toggle each ablation component. With all flags False, the
model reproduces the existing `ResUnet3D` baseline so registering it as
`base_cnn` (or any pure-CNN variant) gives identical behaviour.

Phase 1 wires `use_modality_stems` + `use_cross_modal`. Phase 2 adds
`use_freq`, Phase 3 adds `use_spectral_swin`, Phase 4 adds `use_uncertainty`,
Phase 5 will add `use_boundary`.

Forward returns:
  - When NO auxiliary head is enabled (use_uncertainty / use_boundary are
    both False) — backwards-compatible tuple/tensor:
      * Training: (seg_final, seg_ds1, seg_ds2)
      * Eval:     seg_final tensor
  - When ANY auxiliary head is enabled — dict:
      * Training: {"seg": (seg_final, seg_ds1, seg_ds2),
                    "variance": tensor or None,
                    "boundary": tensor or None}
      * Eval:     {"seg": seg_final,
                    "variance": tensor or None,
                    "boundary": tensor or None}

The trainer / evaluator handle both shapes — see train_variant.py and
evaluation/_core.py for the wrap-or-detect at the boundary.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from configs import config
from model.model import ResidualBlock
from model.blocks.modality_stem import ModalityStem
from model.blocks.cross_modal_attn import CrossModalAttention
from model.blocks.frequency import FrequencyAwareBlock
from model.blocks.spectral_swin import SpectralSwinStage
from model.blocks.uncertainty_bottleneck import UncertaintyBottleneck
from model.blocks.boundary_head import BoundaryHead
from model.blocks.multiscale_fusion import MultiScaleFusionHead
from model.blocks.tc_refine_head import TCRefineHead


class TransResUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 4,
        base_filters: int = 32,
        num_modalities: int = 4,
        stem_channels: int = 8,
        cross_modal_grid: int = 32,
        cross_modal_heads: int = 2,
        # Phase 3 spectral_swin knobs
        spectral_window_size: int = 4,
        spectral_num_heads: int = 8,
        spectral_ffn_ratio: float = 4.0,
        spectral_drop_path_max: float = 0.10,
        # Ablation flags
        use_modality_stems: bool = False,
        use_cross_modal: bool = False,
        use_freq: bool = False,            # Phase 2
        use_spectral_swin: bool = False,   # Phase 3
        use_uncertainty: bool = False,     # Phase 4
        use_boundary: bool = False,        # Phase 5
        # Phase 6 architectural upgrades (default = legacy behaviour)
        spectral_blocks_per_stage: int = 2,
        encoder_extra_depth: bool = False,
        use_multiscale_fusion_head: bool = False,
        use_tc_refine: bool = False,       # full_lean — dedicated TC pathway
        output_mode: str = "softmax",
        decoder_dropout_inner: float = 0.0,
        decoder_dropout_final: float = 0.0,
    ):
        super().__init__()

        if use_cross_modal and not use_modality_stems:
            raise ValueError("use_cross_modal=True requires use_modality_stems=True")
        if use_uncertainty and not use_spectral_swin:
            raise ValueError(
                "use_uncertainty=True requires use_spectral_swin=True (uncertainty "
                "block is inserted between Stage-2 of SpectralSwinStage and up4)."
            )
        if use_boundary and not use_spectral_swin:
            raise ValueError(
                "use_boundary=True requires use_spectral_swin=True. The boundary "
                "heads themselves are encoder-agnostic, but the ablation matrix "
                "stacks all flags monotonically — relax this guard only if you "
                "intentionally want a boundary-only variant."
            )
        if output_mode not in ("softmax", "sigmoid"):
            raise ValueError(f"output_mode must be 'softmax' or 'sigmoid', got {output_mode!r}")
        if use_multiscale_fusion_head and output_mode != "softmax":
            raise ValueError(
                "use_multiscale_fusion_head=True is only wired for output_mode='softmax'. "
                "Sigmoid heads would need head_channels=3 plumbing that Phase 6 does not implement."
            )
        if spectral_blocks_per_stage < 1:
            raise ValueError(f"spectral_blocks_per_stage must be >=1, got {spectral_blocks_per_stage}")
        if use_tc_refine and not use_multiscale_fusion_head:
            raise ValueError(
                "use_tc_refine=True requires use_multiscale_fusion_head=True "
                "(the TC residual is fused into the fusion-head logits)."
            )
        if use_tc_refine and output_mode != "softmax":
            raise ValueError(
                "use_tc_refine=True is only defined for output_mode='softmax' "
                "(it gates the NCR/ET softmax channels)."
            )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.output_mode = output_mode
        self.use_modality_stems = use_modality_stems
        self.use_cross_modal = use_cross_modal
        self.use_freq = use_freq
        self.use_spectral_swin = use_spectral_swin
        self.use_uncertainty = use_uncertainty
        self.use_boundary = use_boundary
        self.spectral_blocks_per_stage = spectral_blocks_per_stage
        self.encoder_extra_depth = encoder_extra_depth
        self.use_multiscale_fusion_head = use_multiscale_fusion_head
        self.use_tc_refine = use_tc_refine
        self._aux_heads_active = bool(use_uncertainty or use_boundary or use_tc_refine)

        head_channels = num_classes if output_mode == "softmax" else 3

        # ---- Stem (enc1) ----
        if use_modality_stems and use_cross_modal:
            self.modality_stem = ModalityStem(
                stem_channels=stem_channels,
                num_modalities=num_modalities,
            )
            self.cross_modal = CrossModalAttention(
                in_channels_residual=in_channels,
                num_modalities=num_modalities,
                stem_channels=stem_channels,
                out_channels=base_filters,
                num_heads=cross_modal_heads,
                attn_grid=cross_modal_grid,
            )
            self.enc1 = None  # not used
        else:
            self.modality_stem = None
            self.cross_modal = None
            self.enc1 = nn.Sequential(
                nn.Conv3d(in_channels, base_filters, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(base_filters),
                nn.LeakyReLU(inplace=True),
            )

        # ---- Frequency-aware filter (Phase 2) ----
        # Sits between the stem (enc1 / cross-modal output) and enc2.
        self.freq_block = FrequencyAwareBlock(channels=base_filters) if use_freq else None

        # ---- Encoder ----
        self.enc2 = ResidualBlock(base_filters, base_filters * 2, stride=2)
        self.enc3 = ResidualBlock(base_filters * 2, base_filters * 4, stride=2)
        self.enc4 = ResidualBlock(base_filters * 4, base_filters * 8, stride=2)

        # Phase 6 — extra spatial depth at 16^3 before the SpectralSwin (or plain
        # bottleneck). One stride-1 ResidualBlock keeping channel count at C*8.
        if encoder_extra_depth:
            self.enc4b = ResidualBlock(base_filters * 8, base_filters * 8, stride=1)
        else:
            self.enc4b = None

        # ---- Bottleneck ----
        # Default: ResBlock at 8^3 (CNN baseline + Phase 1/2).
        # Phase 3: replace `enc4 → bottleneck` with the SpectralSwinStage.
        # The stage returns (skip_16, deep_8): skip_16 is the Stage-1 refined
        # enc4 feature used as the dec4 skip; deep_8 replaces the bottleneck.
        # Phase 6: deepen each Swin stage via spectral_blocks_per_stage.
        if use_spectral_swin:
            self.bottleneck = None
            self.spectral_swin_stage = SpectralSwinStage(
                in_channels=base_filters * 8,
                out_channels=base_filters * 16,
                window_size=spectral_window_size,
                num_heads=spectral_num_heads,
                ffn_ratio=spectral_ffn_ratio,
                drop_path_max=spectral_drop_path_max,
                depth_stage1=spectral_blocks_per_stage,
                depth_stage2=spectral_blocks_per_stage,
            )
        else:
            self.bottleneck = ResidualBlock(base_filters * 8, base_filters * 16, stride=2)
            self.spectral_swin_stage = None

        # ---- Phase 4 — Uncertainty-guided bottleneck ----
        # Operates on Stage-2 output (B, 512, 8, 8, 8) when use_spectral_swin=True.
        # Variance map upsampled to the configured patch size for the loss /
        # eval-time uncertainty diagnostics. Default tracks config.PATCH_SIZE so
        # variance lines up with the 128^3 training/eval patches.
        if use_uncertainty:
            self.uncertainty_block = UncertaintyBottleneck(
                in_channels=base_filters * 16,
                upsample_size=tuple(config.PATCH_SIZE),
            )
        else:
            self.uncertainty_block = None

        # ---- Decoder ----
        self.up4 = nn.ConvTranspose3d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(base_filters * 16, base_filters * 8)

        self.up3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(base_filters * 8, base_filters * 4)
        self.ds2_cls = nn.Conv3d(base_filters * 4, head_channels, kernel_size=1)

        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(base_filters * 4, base_filters * 2)
        self.ds1_cls = nn.Conv3d(base_filters * 2, head_channels, kernel_size=1)

        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(base_filters * 2, base_filters)

        # Phase 6 — multi-scale fusion seg head consumes d1, d2, d3 instead of
        # just d1. Boundary heads + deep-sup heads still read raw d2/d3.
        if use_multiscale_fusion_head:
            self.fusion_head = MultiScaleFusionHead(
                c1=base_filters,
                c2=base_filters * 2,
                c3=base_filters * 4,
                mid=16,
                head_channels=head_channels,
            )
            self.final_conv = None
        else:
            self.fusion_head = None
            self.final_conv = nn.Conv3d(base_filters, head_channels, kernel_size=1)

        # ---- Phase 5 — Boundary-aware decoder heads ----
        # One small Conv3d head per decoder stage, mirroring the seg deep-sup
        # pattern (full-res / 1/2 / 1/4). Applied to the post-dropout d3/d2/d1
        # features so the boundary supervision sees the same representation
        # the seg head does. boundary_head_3 → 32^3, _2 → 64^3, _1 → 128^3.
        if use_boundary:
            self.boundary_head_1 = BoundaryHead(in_channels=base_filters)         # d1, 128^3
            self.boundary_head_2 = BoundaryHead(in_channels=base_filters * 2)     # d2, 64^3
            self.boundary_head_3 = BoundaryHead(in_channels=base_filters * 4)     # d3, 32^3
        else:
            self.boundary_head_1 = None
            self.boundary_head_2 = None
            self.boundary_head_3 = None

        # ---- full_lean — dedicated TC-Refine pathway ----
        # head1 reads d1 (full-res, base_filters), head2 reads d2 (1/2-res,
        # base_filters*2). The full-res TC logit is fused as a gated residual
        # into the fusion-head softmax logits (NCR/ET channels only).
        if use_tc_refine:
            self.tc_refine = TCRefineHead(base_filters, base_filters * 2)
        else:
            self.tc_refine = None

        # Optional decoder dropout (kept here for parity with model_transformer.py;
        # zero by default for cnn-family variants).
        self.dropout_inner = nn.Dropout3d(p=decoder_dropout_inner) if decoder_dropout_inner > 0 else nn.Identity()
        self.dropout_final = nn.Dropout3d(p=decoder_dropout_final) if decoder_dropout_final > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # -----------------------------------------------------------------
    def parameter_groups(self):
        """Return (cnn_params, transformer_params) for split optimizer groups.

        Used by the variant-aware trainer when the preset has
        `use_param_groups=True`. Anything under the spectral-Swin stage counts
        as transformer; everything else (CNN encoder/decoder, modality stems,
        cross-modal attention, the standalone Phase-2 freq_block) is treated
        as CNN. Returns an empty transformer-group list for variants that
        don't include a transformer stage — the trainer handles that gracefully.
        """
        cnn_params, transformer_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("spectral_swin_stage."):
                transformer_params.append(p)
            else:
                cnn_params.append(p)
        return cnn_params, transformer_params

    # -----------------------------------------------------------------
    def _stem_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_modality_stems and self.use_cross_modal:
            mod_feats, fg = self.modality_stem(x)
            return self.cross_modal(mod_feats, fg, x)
        return self.enc1(x)

    def forward(self, x):
        x1 = self._stem_forward(x)
        if self.freq_block is not None:
            x1 = self.freq_block(x1)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        if self.enc4b is not None:
            x4 = self.enc4b(x4)

        if self.use_spectral_swin:
            # Stage-1 refined enc4 is the dec4 skip; Stage-2 output is the
            # bottleneck-equivalent input to up4.
            x4_skip, x_bot = self.spectral_swin_stage(x4)
        else:
            x4_skip = x4
            x_bot = self.bottleneck(x4)

        # Phase 4 — uncertainty-guided gate at the bottleneck.
        variance_full = None
        if self.uncertainty_block is not None:
            x_bot, variance_full = self.uncertainty_block(x_bot)

        d4 = self.up4(x_bot)
        d4 = torch.cat((x4_skip, d4), dim=1)
        d4 = self.dec4(d4)
        d4 = self.dropout_inner(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.dec3(d3)
        d3 = self.dropout_inner(d3)
        out_ds2 = self.ds2_cls(d3)
        # boundary_head_3 reads d3 BEFORE seg ds2 conv so we route through the
        # same feature map the seg deep-sup branch consumes.
        b3 = self.boundary_head_3(d3) if self.boundary_head_3 is not None else None

        d2 = self.up2(d3)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.dec2(d2)
        d2 = self.dropout_inner(d2)
        out_ds1 = self.ds1_cls(d2)
        b2 = self.boundary_head_2(d2) if self.boundary_head_2 is not None else None

        d1 = self.up1(d2)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.dec1(d1)
        d1 = self.dropout_final(d1)
        b1 = self.boundary_head_1(d1) if self.boundary_head_1 is not None else None

        if self.fusion_head is not None:
            out_final = self.fusion_head(d1, d2, d3)
        else:
            out_final = self.final_conv(d1)

        # full_lean — gated TC residual fused into the final softmax logits
        # (NCR/ET channels only). tc1/tc2 are also surfaced for the deep-
        # supervised TC loss. gate init 0 → identical to the no-TC config.
        tc1 = tc2 = None
        if self.tc_refine is not None:
            tc1 = self.tc_refine.head1(d1)
            tc2 = self.tc_refine.head2(d2)
            out_final = self.tc_refine.fuse(out_final, tc1)

        # Backwards-compatible: variants without auxiliary heads still return a
        # tuple/tensor so existing trainer/evaluator paths are unchanged.
        if not self._aux_heads_active:
            if self.training:
                return out_final, out_ds1, out_ds2
            return out_final

        # TC-refine-only config (no uncertainty/boundary): at eval the TC
        # residual is already folded into out_final, so return the plain
        # legacy tensor — evaluate_variant.py / _core.py need no change.
        # Training surfaces tc=(tc1, tc2) for the deep-supervised TC loss.
        if self.use_tc_refine and not (self.use_uncertainty or self.use_boundary):
            if self.training:
                return {
                    "seg": (out_final, out_ds1, out_ds2),
                    "variance": None,
                    "boundary": None,
                    "tc": (tc1, tc2),
                }
            return out_final

        # Auxiliary heads enabled → dict output. Trainer/evaluator extract
        # "seg" and (optionally) consume "variance"/"boundary".
        # Train mode boundary: (b1, b2, b3) full→coarse mirroring (seg, ds1, ds2).
        # Eval mode boundary: b1 only (full-res) for diagnostics.
        if self.use_boundary:
            boundary_train = (b1, b2, b3)
            boundary_eval = b1
        else:
            boundary_train = None
            boundary_eval = None

        if self.training:
            return {
                "seg": (out_final, out_ds1, out_ds2),
                "variance": variance_full,
                "boundary": boundary_train,
            }
        return {
            "seg": out_final,
            "variance": variance_full,
            "boundary": boundary_eval,
        }
