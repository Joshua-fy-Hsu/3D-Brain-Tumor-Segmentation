import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from configs import config


# Per-worker cache of memory-mapped arrays. Each DataLoader worker is its own
# process, so this dict is process-local; populated lazily inside __getitem__.
# Keys: patient_id -> (image_mm, mask_mm, tumor_coords_or_None).
#
# Cap is small on purpose: an mmap doesn't count toward RSS until pages are
# touched, but once touched they stay resident in the worker's working set.
# With 6 workers and 178 MB per image.npy, a large cap silently consumes tens
# of GB of physical RAM and triggers swapping. The cache still removes the
# per-sample np.load header parse (the actual bottleneck) at small sizes.
_WORKER_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
_WORKER_CACHE_CAP = int(os.environ.get("BRATS_CACHE_CAP", "8"))


class BratsDataset(Dataset):
    """
    BraTS 2021 Dataset with:
    - 5-channel input: 4 MRI modalities + 1 foreground one-hot mask
    - Training augmentations: zoom, per-axis flip, Gaussian noise/blur, brightness, contrast

    `gpu_aug=True` skips the CPU-side zoom and gaussian blur in `augment` because
    the caller (`train_variant.py`) runs equivalent ops on the GPU after
    `.to(device)`. Other CPU augmentations (flips, noise, brightness, contrast)
    still run because they have no GPU counterpart.
    """

    def __init__(
        self,
        phase="train",
        gpu_aug: bool = False,
        ncr_sample_prob: float = 0.0,
        tumor_sample_prob: float = 0.5,
    ):
        self.phase = phase
        self.gpu_aug = gpu_aug
        # Sampling fractions for training crops:
        #   r ~ U(0,1)
        #   r < ncr_sample_prob              -> NCR-centered (skipped if no NCR)
        #   ncr_sample_prob <= r < tumor_sample_prob -> generic tumor-centered
        #   r >= tumor_sample_prob           -> uniformly random
        # Defaults preserve the original 50/50 tumor/random split with no NCR
        # bias, so callers that don't opt in see no behavior change.
        assert 0.0 <= ncr_sample_prob <= tumor_sample_prob <= 1.0, \
            "require 0 <= ncr_sample_prob <= tumor_sample_prob <= 1"
        self.ncr_sample_prob = float(ncr_sample_prob)
        self.tumor_sample_prob = float(tumor_sample_prob)
        self.data_root = config.TRAIN_DATA_PATH
        self.patch_size = config.PATCH_SIZE

        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"Optimized path not found: {self.data_root}")

        all_folders = sorted([
            f for f in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, f))
        ])

        valid_patients = []
        for pat_id in all_folders:
            pat_path = os.path.join(self.data_root, pat_id)
            if os.path.exists(os.path.join(pat_path, "image.npy")) and \
               os.path.exists(os.path.join(pat_path, "mask.npy")):
                valid_patients.append(pat_id)

        split_idx = config.TRAIN_COUNT
        if split_idx > len(valid_patients):
            split_idx = int(len(valid_patients) * 0.8)

        if self.phase == "train":
            self.patient_list = valid_patients[:split_idx]
        elif self.phase == "val":
            self.patient_list = valid_patients[split_idx:]
        else:
            self.patient_list = valid_patients

        print(f"[{self.phase.upper()}] Dataset Loaded: {len(self.patient_list)} patients.")

    def __len__(self):
        return len(self.patient_list)

    # ------------------------------------------------------------------
    # Worker init — called once per DataLoader worker process. With
    # persistent_workers=True it runs once for the whole training run.
    # ------------------------------------------------------------------

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        """Reset the per-worker mmap cache. Each worker process has its own
        copy of the module-level dict; we just make sure it starts empty."""
        _WORKER_CACHE.clear()

    def _get_patient_handles(self, patient_id: str):
        """Return (image_mm, mask_mm, tumor_coords, ncr_coords). Opens and
        caches mmaps on first access; on subsequent accesses returns the cached
        handles. Bounded LRU eviction keeps the open-file-handle count
        predictable. ncr_coords is computed only when NCR-biased sampling is
        enabled and may be None if the patient has no NCR voxels."""
        cached = _WORKER_CACHE.get(patient_id)
        if cached is not None:
            _WORKER_CACHE.move_to_end(patient_id)
            return cached

        patient_path = os.path.join(self.data_root, patient_id)
        image_mm = np.load(os.path.join(patient_path, "image.npy"), mmap_mode="r")
        mask_mm  = np.load(os.path.join(patient_path, "mask.npy"),  mmap_mode="r")

        coords_path = os.path.join(patient_path, "tumor_coords.npy")
        tumor_coords = np.load(coords_path) if os.path.exists(coords_path) else None

        ncr_coords = None
        if self.ncr_sample_prob > 0.0:
            ncr_path = os.path.join(patient_path, "ncr_coords.npy")
            if os.path.exists(ncr_path):
                ncr_coords = np.load(ncr_path)
            else:
                # Derive from mask once per patient per worker. NCR (label 1)
                # is rare; subsample to match tumor_coords budget.
                mask_full = np.asarray(mask_mm)
                ncr_idx = np.argwhere(mask_full == 1)
                if len(ncr_idx) > 0:
                    if len(ncr_idx) > 8192:
                        sel = np.random.choice(len(ncr_idx), size=8192, replace=False)
                        ncr_idx = ncr_idx[sel]
                    ncr_coords = ncr_idx.astype(np.int16)

        entry = (image_mm, mask_mm, tumor_coords, ncr_coords)
        _WORKER_CACHE[patient_id] = entry
        # Evict aggressively — large caches pin too much resident memory across
        # workers. `del` drops the only remaining reference so numpy releases
        # the memmap and the OS can reclaim the pages.
        while len(_WORKER_CACHE) > _WORKER_CACHE_CAP:
            evicted_key, _ = _WORKER_CACHE.popitem(last=False)
            del evicted_key
        return entry

    def __getitem__(self, idx):
        patient_id = self.patient_list[idx]
        ph, pw, pd = self.patch_size
        shape = (240, 240, 155)

        image_mm, mask_mm, tumor_coords, ncr_coords = self._get_patient_handles(patient_id)

        # Crop strategy (training only): see __init__ docstring for fractions.
        if self.phase == "train":
            r = np.random.rand()
            if r < self.ncr_sample_prob and ncr_coords is not None and len(ncr_coords) > 0:
                x, y, z = self.get_tumor_centered_coords(mask_mm, shape, ncr_coords)
            elif r < self.tumor_sample_prob:
                x, y, z = self.get_tumor_centered_coords(mask_mm, shape, tumor_coords)
            else:
                x, y, z = self.get_crop_coords(shape)
        else:
            x, y, z = self.get_crop_coords(shape)
        mask_patch = np.asarray(mask_mm[x:x+ph, y:y+pw, z:z+pd])

        img_patch = np.asarray(image_mm[:, x:x+ph, y:y+pw, z:z+pd]).astype(np.float32)

        # Apply augmentations during training only.
        if self.phase == "train":
            img_patch, mask_patch = self.augment(img_patch, mask_patch)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(img_patch)),
            "mask":  torch.from_numpy(np.ascontiguousarray(mask_patch)).long()
        }

    # ------------------------------------------------------------------
    # Augmentations
    # ------------------------------------------------------------------

    def augment(self, img, mask):
        """
        img:  (C, D, H, W) float32
        mask: (D, H, W)    int64

        When self.gpu_aug is True, zoom and gaussian blur are skipped here —
        the train loop runs them on GPU after .to(device).
        """
        C, D, H, W = img.shape

        # 1. Zoom — scale up then center-crop back to original size.
        if not self.gpu_aug and np.random.rand() < 0.15:
            scale = np.random.uniform(1.0, 1.4)
            nd, nh, nw = int(D * scale), int(H * scale), int(W * scale)

            img_t = torch.from_numpy(img).unsqueeze(0)  # (1, C, D, H, W)
            img_t = F.interpolate(img_t, size=(nd, nh, nw), mode='trilinear', align_corners=False)

            mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            mask_t = F.interpolate(mask_t, size=(nd, nh, nw), mode='nearest')

            d0 = (nd - D) // 2
            h0 = (nh - H) // 2
            w0 = (nw - W) // 2
            img  = img_t[0, :, d0:d0+D, h0:h0+H, w0:w0+W].numpy()
            mask = mask_t[0, 0, d0:d0+D, h0:h0+H, w0:w0+W].numpy().astype(np.int64)

        # 2. Per-axis flips (p=0.5 each axis, applied independently).
        for ax in range(3):  # D, H, W
            if np.random.rand() < 0.5:
                img  = np.flip(img,  axis=ax + 1)
                mask = np.flip(mask, axis=ax)

        # 3. Gaussian noise.
        if np.random.rand() < 0.15:
            std = np.random.uniform(0, 0.33)
            img = img + np.random.normal(0, std, img.shape).astype(np.float32)

        # 4. Gaussian blur (applied per channel, foreground channel excluded).
        if not self.gpu_aug and np.random.rand() < 0.15:
            sigma = np.random.uniform(0.5, 1.5)
            for c in range(C - 1):  # skip the foreground channel
                img[c] = gaussian_filter(img[c], sigma=sigma)

        # 5. Brightness scaling.
        if np.random.rand() < 0.15:
            factor = np.random.uniform(0.7, 1.3)
            img[:C-1] = img[:C-1] * factor

        # 6. Contrast scaling (clipped to original value range per channel).
        if np.random.rand() < 0.15:
            factor = np.random.uniform(0.65, 1.5)
            for c in range(C - 1):
                vmin, vmax = img[c].min(), img[c].max()
                img[c] = np.clip(img[c] * factor, vmin, vmax)

        return img.astype(np.float32), mask.astype(np.int64)

    # ------------------------------------------------------------------
    # Crop helpers
    # ------------------------------------------------------------------

    def get_crop_coords(self, shape):
        h, w, d = shape
        ph, pw, pd = self.patch_size
        x = np.random.randint(0, max(1, h - ph))
        y = np.random.randint(0, max(1, w - pw))
        z = np.random.randint(0, max(1, d - pd))
        return x, y, z

    def get_tumor_centered_coords(self, mask_mm, shape, tumor_coords=None):
        """Return a top-left patch corner whose patch is centered on a random
        tumor voxel. Uses the precomputed `tumor_coords.npy` sidecar when
        available; otherwise falls back to scanning the mask.
        """
        if tumor_coords is not None and len(tumor_coords) > 0:
            cx, cy, cz = tumor_coords[np.random.randint(len(tumor_coords))]
        else:
            mask_full = np.asarray(mask_mm)
            tumor_indices = np.argwhere(mask_full > 0)
            if len(tumor_indices) == 0:
                return self.get_crop_coords(shape)
            cx, cy, cz = tumor_indices[np.random.randint(len(tumor_indices))]

        ph, pw, pd = self.patch_size
        x = max(0, min(int(cx) - ph // 2, shape[0] - ph))
        y = max(0, min(int(cy) - pw // 2, shape[1] - pw))
        z = max(0, min(int(cz) - pd // 2, shape[2] - pd))
        return x, y, z
