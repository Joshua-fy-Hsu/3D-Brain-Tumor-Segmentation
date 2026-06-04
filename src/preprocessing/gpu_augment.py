"""GPU-side augmentations to avoid CPU bottleneck on heavy ops.

Run *after* `.to(device)` in the train loop. All ops are batched and run on
GPU in bf16/fp16, so they cost a few ms per batch vs 100+ ms on CPU.

Spatial augmentation is deliberately heavier (bigger zoom incl. zoom-out,
small 3D rotation, elastic deformation) to combat data scarcity. Intensity
aug (blur here; noise/brightness/contrast on the CPU side) is left light on
purpose — aggressive intensity jitter mixes the low-contrast NCR core into
edema/healthy tissue and re-damages the tumor core.

img:  (B, C, D, H, W) — channels 0..C-2 are MRI, the last channel is the
      binary foreground mask. Spatial warps move the foreground channel with
      the image, then re-binarize it so the 5th channel stays a hard mask.
mask: (B, D, H, W) long — labels {0,1,2,3}.
"""
import torch
import torch.nn.functional as F


def _rebinarize_fg(img: torch.Tensor) -> torch.Tensor:
    """Foreground channel (last) must stay {0,1} after interpolation."""
    img[:, -1] = (img[:, -1] > 0.5).to(img.dtype)
    return img


@torch.no_grad()
def gpu_zoom(img: torch.Tensor, mask: torch.Tensor,
             p: float = 0.30, scale_range=(0.85, 1.5)) -> tuple:
    """Isotropic zoom. scale>1 → zoom-in (resize up, center-crop back);
    scale<1 → zoom-out (resize down, center-pad back). Per-batch coin flip."""
    if torch.rand(1).item() >= p:
        return img, mask
    B, C, D, H, W = img.shape
    scale = float(torch.empty(1).uniform_(*scale_range))
    nd, nh, nw = max(1, int(D * scale)), max(1, int(H * scale)), max(1, int(W * scale))

    img_z = F.interpolate(img, size=(nd, nh, nw), mode="trilinear",
                          align_corners=False)
    mask_z = F.interpolate(mask.unsqueeze(1).float(), size=(nd, nh, nw),
                           mode="nearest").squeeze(1).long()

    if scale >= 1.0:
        d0, h0, w0 = (nd - D) // 2, (nh - H) // 2, (nw - W) // 2
        img_z = img_z[:, :, d0:d0 + D, h0:h0 + H, w0:w0 + W]
        mask_z = mask_z[:, d0:d0 + D, h0:h0 + H, w0:w0 + W]
    else:
        # center-pad smaller volume back to (D, H, W) with background.
        pd, ph, pw = D - nd, H - nh, W - nw
        pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2, pd // 2, pd - pd // 2)
        img_z = F.pad(img_z, pad, mode="constant", value=0.0)
        mask_z = F.pad(mask_z, pad, mode="constant", value=0)

    img_z = _rebinarize_fg(img_z.contiguous())
    return img_z, mask_z.contiguous()


def _identity_grid(B, D, H, W, device, dtype):
    """affine_grid identity → (B, D, H, W, 3), last dim order (x=W, y=H, z=D)."""
    eye = torch.eye(3, 4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    return F.affine_grid(eye, (B, 1, D, H, W), align_corners=False)


@torch.no_grad()
def gpu_rotate(img: torch.Tensor, mask: torch.Tensor,
               p: float = 0.25, max_deg: float = 15.0) -> tuple:
    """Random small rotation about each axis. Per-batch trigger; one shared
    rotation per batch. img bilinear, mask nearest."""
    if torch.rand(1).item() >= p:
        return img, mask
    B, C, D, H, W = img.shape
    dev, dt = img.device, torch.float32
    ang = (torch.rand(3, device=dev) * 2 - 1) * (max_deg * 3.1415926535 / 180.0)
    cx, sx = torch.cos(ang[0]), torch.sin(ang[0])
    cy, sy = torch.cos(ang[1]), torch.sin(ang[1])
    cz, sz = torch.cos(ang[2]), torch.sin(ang[2])
    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], device=dev, dtype=dt)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], device=dev, dtype=dt)
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], device=dev, dtype=dt)
    R = Rz @ Ry @ Rx
    theta = torch.zeros(B, 3, 4, device=dev, dtype=dt)
    theta[:, :, :3] = R.unsqueeze(0)
    grid = F.affine_grid(theta, (B, 1, D, H, W), align_corners=False)

    img_r = F.grid_sample(img.float(), grid, mode="bilinear",
                          padding_mode="zeros", align_corners=False).to(img.dtype)
    mask_r = F.grid_sample(mask.unsqueeze(1).float(), grid, mode="nearest",
                           padding_mode="zeros", align_corners=False)
    img_r = _rebinarize_fg(img_r)
    return img_r.contiguous(), mask_r.squeeze(1).long().contiguous()


@torch.no_grad()
def gpu_elastic(img: torch.Tensor, mask: torch.Tensor,
                p: float = 0.25, ctrl: int = 5, alpha: float = 0.12) -> tuple:
    """Elastic deformation: a coarse random displacement field (ctrl^3) is
    trilinearly upsampled and added to the identity sampling grid. `alpha`
    is the max displacement in normalized [-1,1] coords (~0.12 ≈ 8 vox at
    128). img bilinear, mask nearest."""
    if torch.rand(1).item() >= p:
        return img, mask
    B, C, D, H, W = img.shape
    dev = img.device
    disp = (torch.rand(B, 3, ctrl, ctrl, ctrl, device=dev) * 2 - 1) * alpha
    disp = F.interpolate(disp, size=(D, H, W), mode="trilinear",
                         align_corners=False)
    # grid last-dim order is (x=W, y=H, z=D); permute disp channels to match.
    disp = disp.permute(0, 2, 3, 4, 1)  # (B, D, H, W, 3) as (dz, dy, dx)
    disp = disp[..., [2, 1, 0]]         # -> (dx, dy, dz)
    grid = _identity_grid(B, D, H, W, dev, torch.float32) + disp

    img_e = F.grid_sample(img.float(), grid, mode="bilinear",
                          padding_mode="zeros", align_corners=False).to(img.dtype)
    mask_e = F.grid_sample(mask.unsqueeze(1).float(), grid, mode="nearest",
                           padding_mode="zeros", align_corners=False)
    img_e = _rebinarize_fg(img_e)
    return img_e.contiguous(), mask_e.squeeze(1).long().contiguous()


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
    img, mask = gpu_rotate(img, mask)
    img, mask = gpu_elastic(img, mask)
    img = gpu_blur(img)
    return img, mask
