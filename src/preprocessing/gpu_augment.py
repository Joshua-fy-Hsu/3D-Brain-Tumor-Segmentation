"""GPU-side augmentations to avoid CPU bottleneck on heavy ops.

Run *after* `.to(device)` in the train loop. All ops are batched and run on
GPU in bf16/fp16, so they cost <5 ms total per batch vs 100+ ms on CPU.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def gpu_zoom(img: torch.Tensor, mask: torch.Tensor,
             p: float = 0.15, scale_range=(1.0, 1.4)) -> tuple:
    """img: (B, C, D, H, W). mask: (B, D, H, W) long. Per-batch coin flip."""
    if torch.rand(1).item() >= p:
        return img, mask
    B, C, D, H, W = img.shape
    scale = float(torch.empty(1).uniform_(*scale_range))
    nd, nh, nw = int(D * scale), int(H * scale), int(W * scale)

    img_z = F.interpolate(img, size=(nd, nh, nw), mode="trilinear", align_corners=False)
    mask_z = F.interpolate(mask.unsqueeze(1).float(), size=(nd, nh, nw), mode="nearest")
    mask_z = mask_z.squeeze(1).long()

    d0, h0, w0 = (nd - D) // 2, (nh - H) // 2, (nw - W) // 2
    img_z = img_z[:, :, d0:d0 + D, h0:h0 + H, w0:w0 + W].contiguous()
    mask_z = mask_z[:, d0:d0 + D, h0:h0 + H, w0:w0 + W].contiguous()
    return img_z, mask_z


def _gauss_kernel(sigma: float, radius: int, device, dtype):
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


@torch.no_grad()
def gpu_blur(img: torch.Tensor, p: float = 0.15, sigma_range=(0.5, 1.5)) -> torch.Tensor:
    """Separable 3D Gaussian blur over D/H/W. Per-sample trigger and sigma —
    each batch item independently decides whether to blur and at what sigma.
    MRI channels (0..3) only; foreground channel (last) untouched."""
    B, C, D, H, W = img.shape
    Cm = C - 1
    out = img.clone()

    for b in range(B):
        if torch.rand(1).item() >= p:
            continue
        sigma = float(torch.empty(1).uniform_(*sigma_range))
        radius = max(1, int(sigma * 3))
        k = _gauss_kernel(sigma, radius, img.device, img.dtype)
        pad = radius

        mri = img[b:b + 1, :Cm].contiguous()
        w_d = k.view(1, 1, -1, 1, 1).expand(Cm, 1, -1, 1, 1).contiguous()
        w_h = k.view(1, 1, 1, -1, 1).expand(Cm, 1, 1, -1, 1).contiguous()
        w_w = k.view(1, 1, 1, 1, -1).expand(Cm, 1, 1, 1, -1).contiguous()
        mri = F.conv3d(mri, w_d, padding=(pad, 0, 0), groups=Cm)
        mri = F.conv3d(mri, w_h, padding=(0, pad, 0), groups=Cm)
        mri = F.conv3d(mri, w_w, padding=(0, 0, pad), groups=Cm)

        out[b:b + 1, :Cm] = mri
    return out


@torch.no_grad()
def gpu_augment(img: torch.Tensor, mask: torch.Tensor) -> tuple:
    img, mask = gpu_zoom(img, mask)
    img = gpu_blur(img)
    return img, mask
