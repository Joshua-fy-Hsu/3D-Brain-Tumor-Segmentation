"""Robustness perturbations: Gaussian noise, bias field, motion, missing modality."""
import numpy as np
import torch

try:
    import torchio as tio
    _HAS_TORCHIO = True
except Exception:
    _HAS_TORCHIO = False


# Apply perturbations only to the 4 MRI channels (idx 0..3).
# Channel 4 is the foreground binary mask — leave alone.
MRI_IDX = (0, 1, 2, 3)
MODALITY_NAMES = ("t1", "t1ce", "t2", "flair")


def add_gaussian_noise(x, sigma):
    """x: (1,5,D,H,W) tensor. Adds N(0, sigma) noise inside the foreground only."""
    out = x.clone()
    fg = out[:, 4:5] > 0.5
    noise = torch.randn_like(out[:, :4]) * sigma
    out[:, :4] = out[:, :4] + noise * fg
    return out


def apply_bias_field(x, coefficients=0.5, order=3):
    if not _HAS_TORCHIO:
        return x
    out = x.clone()
    transform = tio.RandomBiasField(coefficients=coefficients, order=order, p=1.0)
    arr = out[0, :4].cpu().numpy()  # (4, D, H, W)
    subj = tio.Subject(img=tio.ScalarImage(tensor=torch.from_numpy(arr)))
    aug = transform(subj)
    out[0, :4] = aug["img"].data.to(x.device)
    # Re-zero background
    fg = out[0, 4] > 0.5
    out[0, :4] = out[0, :4] * fg
    return out


def apply_motion(x, severity="moderate"):
    if not _HAS_TORCHIO:
        return x
    presets = {
        "mild":     dict(degrees=2,  translation=2,  num_transforms=1),
        "moderate": dict(degrees=5,  translation=5,  num_transforms=2),
        "severe":   dict(degrees=10, translation=10, num_transforms=3),
    }
    p = presets.get(severity, presets["moderate"])
    out = x.clone()
    transform = tio.RandomMotion(degrees=p["degrees"], translation=p["translation"],
                                 num_transforms=p["num_transforms"], p=1.0)
    arr = out[0, :4].cpu().numpy()
    subj = tio.Subject(img=tio.ScalarImage(tensor=torch.from_numpy(arr)))
    aug = transform(subj)
    out[0, :4] = aug["img"].data.to(x.device)
    fg = out[0, 4] > 0.5
    out[0, :4] = out[0, :4] * fg
    return out


def drop_modality(x, modality):
    """Zero out a single modality channel."""
    idx = MODALITY_NAMES.index(modality)
    out = x.clone()
    out[:, idx] = 0.0
    return out


# Spec: list of (name, fn) pairs to iterate in robustness sweeps.
def perturbation_suite():
    return [
        ("noise_0.05",        lambda x: add_gaussian_noise(x, 0.05)),
        ("noise_0.10",        lambda x: add_gaussian_noise(x, 0.10)),
        ("noise_0.20",        lambda x: add_gaussian_noise(x, 0.20)),
        ("bias_0.3",          lambda x: apply_bias_field(x, coefficients=0.3, order=3)),
        ("bias_0.6",          lambda x: apply_bias_field(x, coefficients=0.6, order=3)),
        ("bias_0.9",          lambda x: apply_bias_field(x, coefficients=0.9, order=3)),
        ("motion_mild",       lambda x: apply_motion(x, "mild")),
        ("motion_moderate",   lambda x: apply_motion(x, "moderate")),
        ("motion_severe",     lambda x: apply_motion(x, "severe")),
        ("drop_t1",           lambda x: drop_modality(x, "t1")),
        ("drop_t1ce",         lambda x: drop_modality(x, "t1ce")),
        ("drop_t2",           lambda x: drop_modality(x, "t2")),
        ("drop_flair",        lambda x: drop_modality(x, "flair")),
    ]
