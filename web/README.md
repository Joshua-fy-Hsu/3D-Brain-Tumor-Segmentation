# Web GUI for Brain Tumor Segmentation

A FastAPI + [Niivue](https://niivue.github.io) workstation that runs the pinned `full` variant on user-uploaded 4-modality MRI volumes (T1, T1CE, T2, FLAIR) and produces an integrated report: 3D viewer, synced axial/coronal/sagittal MPR, per-region volumes, anatomical involvement, auto-summary, confidence, risk badge, and a one-click `.zip` export.

## Quick start

```bash
# 1) Install web deps (assumes torch/monai/nibabel etc. are already installed)
pip install -r src/requirements.txt

# 2) (One-time) build population stats from your val split.
#    Atlas is optional — add --aal-source path/to/AAL3v1.nii.gz to enable
#    anatomical-region overlap. Without it the Anatomy card stays empty.
python scripts/prepare_webapp_assets.py \
    --aal-source path/to/AAL3v1.nii.gz \
    --aal-labels path/to/AAL3v1.nii.txt

# 3) Run the app (defaults to http://localhost:8000)
python -m uvicorn web.server:app --host 0.0.0.0 --port 8000
```

The app loads the newest `logs/run_*/best_model.pth` that state_dict-matches the `full` variant. If none exist for that variant, startup fails loudly — train `full` first or copy in a checkpoint.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET`  | `/` | Single-page UI |
| `GET`  | `/api/meta` | Variant + run name + device, for the top-bar badge |
| `POST` | `/api/predict` | Multipart upload (`t1`, `t1ce`, `t2`, `flair`); returns JSON + NIfTI URLs |
| `GET`  | `/api/session/{sid}/{name}.nii.gz` | NIfTI files Niivue fetches |
| `POST` | `/api/session/{sid}/screenshot` | Client posts the 3D canvas PNG before report download |
| `GET`  | `/api/session/{sid}/report` | ZIP with `seg.nii.gz`, `metrics.json`, `summary.txt`, optional `screenshot.png` |

## Notes

- The web pipeline runs the same `sw_predict` → `_logits_to_4ch_probs` → `decode_labels` → `postprocess_et` path used by `evaluate_variant.py --variant full`. Volumes match the `baseline_post` mode in `summary.csv`.
- We do **not** run TTA or MC Dropout from the UI (too slow). Confidence is the mean per-region softmax probability over the predicted region voxels.
- Atlas overlap assumes BraTS-space input (240×240×155). Non-BraTS uploads silently disable anatomy.
- Sessions live in `web/_sessions/` and are swept after 24 h on startup.
- Niivue is loaded from the unpkg CDN. To vendor (offline use), save `https://unpkg.com/@niivue/niivue/dist/index.js` to `web/static/lib/niivue.esm.js` and change the import in `app.js`.
