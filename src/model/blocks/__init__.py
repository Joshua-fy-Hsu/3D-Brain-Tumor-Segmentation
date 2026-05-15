"""Building blocks for TransResUNet3D variants (Phases 1-5).

Each module here is an independent component, gated by a constructor flag in
`src/model/trans_resunet.py`. Reuses `ResidualBlock` from `src/model/model.py`
for encoder/decoder stages.
"""
from model.blocks.modality_stem import ModalityStem
from model.blocks.cross_modal_attn import CrossModalAttention
from model.blocks.frequency import FrequencyAwareBlock
from model.blocks.spectral_swin import SpectralSwinStage, SpectralWindowedBlock
from model.blocks.uncertainty_bottleneck import UncertaintyBottleneck
from model.blocks.boundary_head import BoundaryHead

__all__ = [
    "ModalityStem",
    "CrossModalAttention",
    "FrequencyAwareBlock",
    "SpectralSwinStage",
    "SpectralWindowedBlock",
    "UncertaintyBottleneck",
    "BoundaryHead",
]
