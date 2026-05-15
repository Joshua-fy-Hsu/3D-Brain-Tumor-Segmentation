import os
import sys
import json
import numpy as np
import nibabel as nib
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from configs import config

# Raw source path (optional — only needed for the full optimize-from-raw pipeline).
RAW_PATH = r"D:/University/Projects/Brain_Tumor_Segmentation/data/BraTS2021_TrainingData"

# Destination = the path the training pipeline reads from.
OPT_PATH = config.TRAIN_DATA_PATH

MODALITIES = config.MODALITIES


def _save_tumor_coords(pat_path: str, mask: np.ndarray, max_coords: int = 8192) -> None:
    """Write tumor_coords.npy as (N, 3) int16. If tumor has more than
    `max_coords` voxels, uniformly subsample — tumor-centered cropping picks
    one row per call, so subsample density doesn't affect statistics.
    """
    coords = np.argwhere(mask > 0)
    if len(coords) > max_coords:
        sel = np.random.choice(len(coords), size=max_coords, replace=False)
        coords = coords[sel]
    np.save(os.path.join(pat_path, "tumor_coords.npy"), coords.astype(np.int16))


# image.npy is stored as float16 to halve disk-bandwidth pressure during
# training. dataset.py and validate.py both cast back to float32 at the
# consumption boundary. Z-scored brain MRI values fit comfortably in fp16
# (verified range [-4.5, 12.3], well inside +-65504; fp16 mantissa precision
# is ~4 decimal digits, far above what the model needs).
IMAGE_NPY_DTYPE = np.float16


def build_npy_for_patient(pat_path: str, pat_id: str) -> bool:
    """
    Build image.npy (5, 240, 240, 155) float16, mask.npy (240, 240, 155) uint8,
    and tumor_coords.npy (N, 3) int16 from the per-patient .nii files and
    stats.json in `pat_path`.

    Idempotent: if image.npy and mask.npy already exist, are newer than
    stats.json, AND image.npy is already at the target dtype, only the (cheap)
    tumor_coords.npy is regenerated. A dtype mismatch (e.g. legacy float32
    files) forces a rewrite. This makes rerunning preprocessing safe and fast
    when the heavy caches are already current AND lets a one-time dtype change
    propagate cleanly when this constant is modified.

    Normalization matches BratsDataset semantics exactly:
      - z-score per modality using stats.json
      - background voxels (non-brain) forced back to 0 after normalization
      - 5th channel = foreground mask built from any non-zero raw voxel
    """
    stats_path = os.path.join(pat_path, "stats.json")
    if not os.path.exists(stats_path):
        return False

    image_path = os.path.join(pat_path, "image.npy")
    mask_npy_path = os.path.join(pat_path, "mask.npy")
    coords_path = os.path.join(pat_path, "tumor_coords.npy")

    cache_current = (
        os.path.exists(image_path)
        and os.path.exists(mask_npy_path)
        and os.path.getmtime(image_path) >= os.path.getmtime(stats_path)
        and os.path.getmtime(mask_npy_path) >= os.path.getmtime(stats_path)
    )

    if cache_current:
        # Verify dtype before declaring the cache current. mmap-load just the
        # header so this is essentially free.
        try:
            existing_dtype = np.load(image_path, mmap_mode="r").dtype
        except Exception:
            existing_dtype = None
        if existing_dtype == IMAGE_NPY_DTYPE:
            if (not os.path.exists(coords_path) or
                    os.path.getmtime(coords_path) < os.path.getmtime(mask_npy_path)):
                mask = np.load(mask_npy_path)
                _save_tumor_coords(pat_path, mask)
            return True
        # Dtype mismatch -> fall through to full rebuild.

    with open(stats_path, "r") as f:
        stats = json.load(f)

    raw_channels = []
    for mod in MODALITIES:
        mod_path = os.path.join(pat_path, f"{pat_id}_{mod}.nii")
        if not os.path.exists(mod_path):
            return False
        raw_channels.append(np.asarray(nib.load(mod_path).dataobj).astype(np.float32))

    foreground = np.any([c != 0 for c in raw_channels], axis=0).astype(np.float32)

    normed = []
    for i, mod in enumerate(MODALITIES):
        ch = (raw_channels[i] - stats[mod]["mean"]) / (stats[mod]["std"] + 1e-8)
        ch[foreground == 0] = 0.0
        normed.append(ch)
    normed.append(foreground)

    # Compute in fp32 (numerical stability of the per-modality normalization),
    # store in IMAGE_NPY_DTYPE.
    image = np.stack(normed, axis=0).astype(IMAGE_NPY_DTYPE)
    np.save(image_path, image)

    mask_src_path = os.path.join(pat_path, f"{pat_id}_seg.nii")
    if os.path.exists(mask_src_path):
        mask = np.asarray(nib.load(mask_src_path).dataobj).astype(np.uint8)
        np.save(mask_npy_path, mask)
        _save_tumor_coords(pat_path, mask)

    return True


def optimize_from_raw():
    """Raw .nii.gz → standardized .nii + stats.json + image.npy + mask.npy."""
    print(f"--> Source: {RAW_PATH}")
    print(f"--> Destination: {OPT_PATH}")
    os.makedirs(OPT_PATH, exist_ok=True)

    patients = sorted([
        f for f in os.listdir(RAW_PATH)
        if os.path.isdir(os.path.join(RAW_PATH, f))
    ])
    print(f"Found {len(patients)} patients. Running full optimize...")

    for pat_id in tqdm(patients, desc="Optimizing"):
        src = os.path.join(RAW_PATH, pat_id)
        dst = os.path.join(OPT_PATH, pat_id)
        os.makedirs(dst, exist_ok=True)

        stats = {}
        for mod in MODALITIES:
            src_file = os.path.join(src, f"{pat_id}_{mod}.nii.gz")
            if not os.path.exists(src_file):
                src_file = os.path.join(src, f"{pat_id}_{mod}.nii")
            if not os.path.exists(src_file):
                continue

            img = nib.load(src_file)
            data = img.get_fdata().astype(np.float32)

            # Per-modality stats computed on brain tissue only.
            m = data > 0
            if m.any():
                stats[mod] = {"mean": float(data[m].mean()), "std": float(data[m].std())}
            else:
                stats[mod] = {"mean": 0.0, "std": 1.0}

            nib.save(nib.Nifti1Image(data, img.affine, img.header),
                     os.path.join(dst, f"{pat_id}_{mod}.nii"))

        src_mask = os.path.join(src, f"{pat_id}_seg.nii.gz")
        if not os.path.exists(src_mask):
            src_mask = os.path.join(src, f"{pat_id}_seg.nii")
        if os.path.exists(src_mask):
            mask_img = nib.load(src_mask)
            mask_data = mask_img.get_fdata().astype(np.uint8)
            mask_data[mask_data == 4] = 3  # BraTS label 4 (ET) → 3
            nib.save(nib.Nifti1Image(mask_data, mask_img.affine, mask_img.header),
                     os.path.join(dst, f"{pat_id}_seg.nii"))

        with open(os.path.join(dst, "stats.json"), "w") as f:
            json.dump(stats, f)

        build_npy_for_patient(dst, pat_id)


def rebuild_npy_cache():
    """Rebuild only image.npy / mask.npy in an existing OPT_PATH."""
    print(f"--> Rebuilding .npy cache in: {OPT_PATH}")
    patients = sorted([
        f for f in os.listdir(OPT_PATH)
        if os.path.isdir(os.path.join(OPT_PATH, f))
    ])
    print(f"Found {len(patients)} patient folders.")

    skipped = 0
    for pat_id in tqdm(patients, desc="Rebuilding .npy"):
        pat_path = os.path.join(OPT_PATH, pat_id)
        if not build_npy_for_patient(pat_path, pat_id):
            skipped += 1
    if skipped:
        print(f"Skipped {skipped} patients (missing .nii or stats.json).")


if __name__ == "__main__":
    if os.path.exists(RAW_PATH):
        optimize_from_raw()
    elif os.path.exists(OPT_PATH):
        # Raw data no longer present — just refresh the .npy cache in-place.
        rebuild_npy_cache()
    else:
        print(f"ERROR: neither RAW_PATH ({RAW_PATH}) nor OPT_PATH ({OPT_PATH}) exists.")
