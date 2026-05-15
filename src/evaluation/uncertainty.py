"""Uncertainty estimation: TTA, MC Dropout (if model has dropout), predictive entropy & variance."""
import numpy as np
import torch
import torch.nn.functional as F

try:
    from monai.inferers import sliding_window_inference
    _HAS_MONAI = True
except Exception:
    _HAS_MONAI = False


def _logits_to_4ch_probs(logits: torch.Tensor, output_mode: str = "softmax") -> torch.Tensor:
    """Convert raw network logits to a 4-channel probability tensor that
    sums to 1 along dim=1 with the BraTS class layout {0=BG, 1=NCR, 2=ED, 3=ET}.

    softmax mode: standard 4-class softmax.
    sigmoid mode: 3 region logits in order (ET, TC, WT). We synthesise a
        4-channel distribution by enforcing the BraTS hierarchy
        ET <= TC <= WT and assigning:
            p(ET)  = p_ET
            p(NCR) = p_TC - p_ET
            p(ED)  = p_WT - p_TC
            p(BG)  = 1 - p_WT
        so that probs[3] / probs[1]+probs[3] / probs[1]+probs[2]+probs[3]
        recover the ET / TC / WT region probabilities exactly.
    """
    if output_mode == "softmax":
        return F.softmax(logits, dim=1)
    if output_mode == "sigmoid":
        s = torch.sigmoid(logits)  # (B, 3, D, H, W) -> (ET, TC, WT)
        p_et = s[:, 0:1]
        p_tc = s[:, 1:2]
        p_wt = s[:, 2:3]
        # Enforce hierarchy ET <= TC <= WT to keep all derived probs >= 0.
        p_tc = torch.minimum(p_tc, p_wt)
        p_et = torch.minimum(p_et, p_tc)
        bg  = 1.0 - p_wt
        ncr = p_tc - p_et
        ed  = p_wt - p_tc
        et  = p_et
        return torch.cat([bg, ncr, ed, et], dim=1)
    raise ValueError(f"Unknown output_mode: {output_mode}")


# ----------------------------------------------------------------------
# Sliding-window predictor (MONAI; falls back to error if not installed)
# ----------------------------------------------------------------------
def sw_predict(model, x, roi=(128, 128, 128), overlap=0.5, sw_batch_size=4,
               amp_dtype=torch.float16):
    """Run sliding-window inference and return raw logits (B,C,D,H,W).
       Uses CUDA AMP autocast (default float16) when input is on GPU.
       Pass `amp_dtype=torch.bfloat16` for bf16-trained models (e.g. the
       transformer variant — fp16 can overflow attention softmax)."""
    if not _HAS_MONAI:
        raise RuntimeError("MONAI is required: pip install monai")
    use_amp = x.is_cuda
    ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp else torch.cpu.amp.autocast(enabled=False)
    with ctx:
        out = sliding_window_inference(
            inputs=x,
            roi_size=roi,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
            mode="gaussian",
            sigma_scale=0.125,
            padding_mode="constant",
            cval=0.0,
        )
    return out.float()


# ----------------------------------------------------------------------
# Test-Time Augmentation (8-way flips)
# ----------------------------------------------------------------------
_FLIP_AXES = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]


_ROT_K = (0, 1, 2, 3)  # 90 deg increments around the in-plane axes (dims 2,3)


@torch.no_grad()
def tta_predict_extended(model, x, roi=(128, 128, 128), overlap=0.5,
                         output_mode="softmax"):
    """Phase 6 extended TTA: 8 flips x 4 in-plane rotations = 32 views.

    BraTS volumes load as (B, C, D=240, H=240, W=155). The first two spatial
    dims are isotropic (1 mm x 1 mm); the third (W axis) is anisotropic.
    Rotating in the (D, H) plane (i.e. ``dims=(2, 3)``) is a clean symmetry
    we can exploit; rotating in any plane that touches W would resample
    across voxels of different physical size and is intentionally excluded.

    Returns (mean_probs, entropy, variance).
    """
    model.eval()
    mean_p = None
    M2 = None
    n = 0
    for axes in _FLIP_AXES:
        xf = torch.flip(x, dims=list(axes)) if axes else x
        for k in _ROT_K:
            if k == 0:
                xr = xf
            else:
                xr = torch.rot90(xf, k=k, dims=(2, 3)).contiguous()
            logits = sw_predict(model, xr, roi=roi, overlap=overlap)
            p = _logits_to_4ch_probs(logits, output_mode=output_mode)
            # Undo rotation first, then flip — operations are not commutative
            # in general but here they're applied on separate dim sets so the
            # order is fine; we still match the (rot last, flip outer) order
            # used when building the view.
            if k != 0:
                p = torch.rot90(p, k=-k, dims=(2, 3))
            if axes:
                p = torch.flip(p, dims=list(axes))
            p = p.contiguous()
            n += 1
            if mean_p is None:
                mean_p = p.clone()
                M2 = torch.zeros_like(p)
            else:
                delta = p - mean_p
                mean_p = mean_p + delta / n
                M2 = M2 + delta * (p - mean_p)
            del logits, p
    var_p = (M2 / n).sum(dim=1, keepdim=True)
    eps = 1e-8
    ent = -(mean_p * (mean_p + eps).log()).sum(dim=1, keepdim=True)
    return mean_p, ent, var_p


@torch.no_grad()
def tta_predict(model, x, roi=(128, 128, 128), overlap=0.5, return_stack=False,
                output_mode="softmax"):
    """Returns (mean_probs, entropy, variance). x: (1,C,D,H,W) tensor on device.
       Streams the running mean + M2 (Welford) so only ~2 prob volumes live on GPU at once.
       If return_stack=True, also returns the stacked per-flip probs (memory-heavy)."""
    model.eval()
    mean_p = None
    M2 = None  # Σ(p - mean)^2
    probs_stack = [] if return_stack else None
    n = 0
    for axes in _FLIP_AXES:
        xf = torch.flip(x, dims=list(axes)) if axes else x
        logits = sw_predict(model, xf, roi=roi, overlap=overlap)
        p = _logits_to_4ch_probs(logits, output_mode=output_mode)
        if axes:
            p = torch.flip(p, dims=list(axes))
        if return_stack:
            probs_stack.append(p.clone())
        n += 1
        if mean_p is None:
            mean_p = p.clone()
            M2 = torch.zeros_like(p)
        else:
            delta = p - mean_p
            mean_p = mean_p + delta / n
            M2 = M2 + delta * (p - mean_p)
        del logits, p
    var_p = (M2 / n).sum(dim=1, keepdim=True)
    eps = 1e-8
    ent = -(mean_p * (mean_p + eps).log()).sum(dim=1, keepdim=True)
    if return_stack:
        return mean_p, ent, var_p, torch.stack(probs_stack, dim=0)
    return mean_p, ent, var_p


# ----------------------------------------------------------------------
# MC Dropout — only meaningful if the model contains dropout layers
# ----------------------------------------------------------------------
def model_has_dropout(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            return True
    return False


def _enable_dropout(model):
    n = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.train()
            n += 1
    return n


@torch.no_grad()
def mc_dropout_predict(model, x, T=20, roi=(128, 128, 128), overlap=0.5,
                       output_mode="softmax"):
    """T forward passes with dropout active. Returns (mean_probs, entropy, variance).
       Returns None,None,None if the model has no dropout layers."""
    if not model_has_dropout(model):
        return None, None, None
    model.eval()
    _enable_dropout(model)
    accum = None
    accum_sq = None
    for _ in range(T):
        logits = sw_predict(model, x, roi=roi, overlap=overlap)
        p = _logits_to_4ch_probs(logits, output_mode=output_mode)
        accum = p if accum is None else accum + p
        accum_sq = p * p if accum_sq is None else accum_sq + p * p
    mean_p = accum / T
    var_p = (accum_sq / T - mean_p * mean_p).clamp(min=0).sum(dim=1, keepdim=True)
    eps = 1e-8
    ent = -(mean_p * (mean_p + eps).log()).sum(dim=1, keepdim=True)
    return mean_p, ent, var_p


# ----------------------------------------------------------------------
# Single deterministic pass (baseline)
# ----------------------------------------------------------------------
@torch.no_grad()
def single_predict(model, x, roi=(128, 128, 128), overlap=0.5, output_mode="softmax"):
    """Returns (probs, entropy). Variance is undefined for a single pass."""
    model.eval()
    logits = sw_predict(model, x, roi=roi, overlap=overlap)
    p = _logits_to_4ch_probs(logits, output_mode=output_mode)
    eps = 1e-8
    ent = -(p * (p + eps).log()).sum(dim=1, keepdim=True)
    return p, ent
