# AURA — 3D Brain Tumor Segmentation (BraTS 2021)

**National Taipei University, Department of Electrical Engineering**

Joshua Hsu (許豐有) · Jason Wu (吳東霖)

A 3D deep-learning pipeline for multi-class brain-tumor segmentation on
multi-modal MRI — with built-in uncertainty estimation, a rigorous evaluation
suite (Dice / HD95 / NSD + calibration + risk–coverage + paired statistics),
and an interactive web workstation for clinical-style inspection.

The headline model, **AURA**, reaches **0.839 mean Dice** (ET / TC / WT) on the
held-out BraTS-2021 validation split (251 cases) — the best of the three models
here, with the strongest **tumor-core** score, **~5× faster** inference and
**lower peak memory** than the all-components "Complex" model.

> Academic capstone project. Trained and evaluated on the public
> [BraTS 2021](http://braintumorsegmentation.org/) dataset.

**Project documents:**
[Report](docs/PROJECT%20REPORT.pdf) ·
[Presentation](docs/PROJECT%20PRESENTATION.pdf) ·
[Poster](docs/PROJECT%20POSTER.pdf) ·
[Proposal](docs/PROJECT%20PROPOSAL.pdf)

---

## Contents

- [Highlights](#highlights)
- [Results](#results)
- [The three models](#the-three-models)
- [Quickstart](#quickstart)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Pretrained weights](#pretrained-weights)
- [Data & preprocessing](#data--preprocessing)
- [Training](#training) · [Evaluation](#evaluation) · [Web workstation](#web-workstation)
- [License & acknowledgements](#license--acknowledgements)

---

## Highlights

- **Three comparable models, one pipeline** — a CNN baseline, an
  all-components "Complex" model, and the clean-slate **AURA** model, all
  trained and evaluated through the same variant-aware harness for a fair,
  apples-to-apples comparison.
- **Uncertainty-aware** — MC-Dropout / test-time-augmentation predictive
  variance, with calibration (brain- and tumor-restricted ECE/ACE) and
  risk–coverage / AURC diagnostics, not just Dice.
- **Honest evaluation** — per-case metrics, sliding-window inference, ET
  post-processing, V_min sweeps, and paired Wilcoxon + bootstrap 95% CIs for
  model comparison.
- **Deployable** — an interactive FastAPI + [Niivue](https://niivue.github.io)
  workstation runs the deployed AURA model end-to-end and exports a PDF report,
  including per-inference GPU energy / CO₂ estimates.

---

## Results

BraTS-2021 validation split (251 cases), test-time augmentation + ET
post-processing (the `tta_post` mode). Dice is per region — **ET** (enhancing
tumor), **TC** (tumor core), **WT** (whole tumor). Best per column in **bold**.

| Model | Dice ET | Dice TC | Dice WT | **Mean Dice** | HD95 ↓ | NSD ↑ |
|---|---|---|---|---|---|---|
| **baseline** (`base_cnn`) | 0.765 | 0.788 | 0.923 | 0.825 | 6.75 | 0.787 |
| **Complex** (`full`)      | **0.789** | 0.781 | **0.929** | 0.833 | **5.95** | **0.803** |
| **AURA** (`hybrid`)       | 0.786 | **0.804** | 0.928 | **0.839** | 5.96 | 0.801 |

**Efficiency** (single 128³ patch, RTX 4060 Laptop, fp16/bf16):

| Model | Params | GFLOPs | Peak mem | Latency |
|---|---|---|---|---|
| baseline | 22.9 M | 436 | 2096 MB | 165 ms |
| Complex  | 37.1 M | 586 | 2448 MB | 1179 ms |
| AURA     | 45.1 M | 613 | **1929 MB** | **232 ms** |

**Takeaways**

- **AURA** has the best mean Dice and the best **tumor-core** score (the
  hardest, most clinically relevant region), runs ~5× faster than Complex, and
  uses the least peak memory — so it is the model deployed in the web app.
- **Complex** turns on *every* component of the ablation backbone (cross-modal
  attention, frequency branch, spectral-Swin stage, predictive-variance head,
  boundary head, multi-scale fusion). It edges ahead on ET / WT / HD95 but, for
  all that machinery, does **not** beat the simpler AURA on the overall metric —
  a useful negative result on stacking complexity.
- **baseline** is a standard 3D Residual U-Net, the comparison anchor.

> Numbers are reproducible from the committed CSVs under [`results/`](results/).

---

## The three models

| Registry key | Name | Architecture |
|---|---|---|
| `base_cnn` | baseline | 3D Residual U-Net — 4 encoder stages (32→64→128→256, ×16 bottleneck), Instance Norm + LeakyReLU, deep supervision. No attention. |
| `full` | Complex | `TransResUNet3D` with **all** components on: modality stems + cross-modal attention + frequency branch + spectral-Swin transformer stage + predictive-variance (uncertainty) head + boundary head + deeper Swin / extra encoder depth + multi-scale fusion head. |
| `hybrid` | AURA | Clean-slate, standalone: residual CNN encoder → transformer at the 8³ bottleneck → CNN decoder, 4-class softmax head, 3-level deep supervision, MC-Dropout uncertainty. |

Input is **5-channel** (T1, T1CE, T2, FLAIR + a binary foreground mask) at a
128³ patch size; output is 4 classes {background, NCR, ED, ET}, scored over the
nested BraTS regions ET ⊂ TC ⊂ WT.

The names are a presentation layer only — the registry keys
(`base_cnn` / `full` / `hybrid`) are the load-bearing identifiers used for
checkpoints (`logs/run_<key>_*`) and results (`results/<key>/`). The single
source of truth is [`src/model/registry.py`](src/model/registry.py).

---

## Quickstart

```bash
git clone <this-repo> && cd Brain_Tumor_Segmentation
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r src/requirements.txt

# Grab the AURA weights from the GitHub Releases page (see "Pretrained weights")
# into logs/run_hybrid_<id>/best_model.pth, then launch the web workstation:
python scripts/prepare_webapp_assets.py
python -m uvicorn web.server:app --port 8000      # open http://localhost:8000
```

---

## Repository layout

```
src/
  configs/config.py          # all paths + hyperparameters
  model/
    registry.py              # the 3 variants — single source of truth
    model.py                 # ResUnet3D            (baseline)
    trans_resunet.py         # TransResUNet3D       (Complex)
    hybrid.py                # HybridUNet3D         (AURA)
    model_transformer.py     # windowed-attention blocks reused by Complex
    blocks/                  # per-component building blocks
  training/
    train_variant.py         # variant-aware trainer (recipes in TRAINING_PRESETS)
    losses.py                # region-wise Dice+Focal, uncertainty/boundary losses
  evaluation/
    evaluate_variant.py      # variant-aware evaluator
    _core.py                 # shared inference / calibration / summary core
    metrics.py stats.py calibration.py uncertainty*.py postprocess.py ...
  preprocessing/
    optimizing.py            # raw .nii.gz -> normalized .npy cache
    dataset.py gpu_augment.py
scripts/                     # helper scripts (see scripts/README.md)
web/                         # FastAPI + Niivue workstation (pinned to AURA)
results/                     # committed validation metrics (CSV) + plots
```

---

## Installation

```bash
# 1. Install PyTorch from the CUDA-matched index first, e.g. CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Install the rest:
pip install -r src/requirements.txt
```

Python 3.10+ and an NVIDIA GPU are recommended for training (Ampere+ for the
bf16 path used by the Complex model). `monai` is a hard dependency of the
evaluation pipeline; `torchio` (robustness perturbations) and the
`fastapi`/`uvicorn` stack (web app) are optional.

---

## Pretrained weights

Model checkpoints are too large for the git repo, so they are published as
**GitHub Release** assets. Download the `best_model.pth` (and, for the
snapshot-ensemble variants, the `snapshot_top*.pth`) for the model you want and
place it under a matching run directory, e.g.:

```
logs/run_hybrid_<id>/best_model.pth        # AURA   (used by the web app)
logs/run_full_<id>/best_model.pth          # Complex
logs/run_base_cnn_<id>/best_model.pth      # baseline
```

The evaluator and web app auto-discover the newest checkpoint whose weights
match the requested variant; pass `--checkpoint <path>` to override. The
validation metrics behind the results tables are committed under
[`results/`](results/) (CSV summaries + plots), and per-model training curves
under `logs/*/training_log.csv`.

---

## Data & preprocessing

This project uses **BraTS 2021** (4 modalities per case: T1, T1CE, T2, FLAIR,
plus expert segmentation). Download it from the
[official challenge page](http://braintumorsegmentation.org/) — it is **not**
included in this repository and has its own data-use terms.

Point the pipeline at your data and run preprocessing:

```bash
export BRATS_DATA_PATH=/path/to/BraTS2021_Optimized   # or set in config.py
python src/preprocessing/optimizing.py
```

Preprocessing produces, per case, a 5-channel `image.npy` (T1, T1CE, T2,
FLAIR, foreground mask; pre-normalized, foreground-masked), a `mask.npy`
(labels {0 background, 1 NCR, 2 ED, 3 ET}; raw BraTS label 4 is remapped to 3),
and tumor coordinates for tumor-centered patch sampling. The first 1,000 cases
are used for training, the remainder for validation.

---

## Training

Recipes live in the per-variant presets in
[`src/training/train_variant.py`](src/training/train_variant.py); the commands
just select a variant.

```bash
python src/training/train_variant.py --variant base_cnn
python src/training/train_variant.py --variant full   --epochs 300 --warmup 10
python src/training/train_variant.py --variant hybrid --epochs 300
```

Checkpoints are written to `logs/run_<variant>_*/` (`best_model.pth`, plus
`snapshot_top*.pth` for the snapshot-ensemble variants). See
[`scripts/README.md`](scripts/README.md) for the full command reference.

---

## Evaluation

```bash
# single best checkpoint
python src/evaluation/evaluate_variant.py --variant base_cnn --vmin-sweep

# snapshot ensemble + 32-view extended TTA (full / hybrid)
python src/evaluation/evaluate_variant.py --variant hybrid \
    --ensemble-ckpts "logs/run_hybrid_*/snapshot_top*.pth" \
    --tta-extended --vmin-sweep --overlap 0.625 --run-name eval_ensemble
```

Results land in `results/<variant>/eval_*/`: `per_case_metrics.csv`,
`summary.csv` (Dice / HD95 / NSD / calibration / AURC per inference mode), plus
plots and uncertainty diagnostics. Compare two runs with a paired Wilcoxon test
+ bootstrap 95% CIs:

```bash
python -m evaluation.stats compare \
    results/base_cnn/eval_X/per_case_metrics.csv \
    results/hybrid/eval_Y/per_case_metrics.csv \
    --mode tta_post --label-a baseline --label-b AURA
```

---

## Web workstation

An interactive FastAPI + [Niivue](https://niivue.github.io) viewer pinned to
the AURA model: upload a case, run inference, inspect the segmentation and a
per-region confidence / uncertainty overlay, and download a PDF report. It also
tracks per-inference GPU energy / CO₂ estimates.

```bash
python scripts/prepare_webapp_assets.py        # one-time: build population stats
python -m uvicorn web.server:app --host 0.0.0.0 --port 8000
```

It loads the newest `logs/run_*/best_model.pth` that matches the `hybrid`
architecture and runs the same inference path as
`evaluate_variant.py --variant hybrid` (no TTA / MC-Dropout, for latency). See
[`web/README.md`](web/README.md) for the API.

---

## License & acknowledgements

Code is released under the [MIT License](LICENSE).

- Data: [BraTS 2021](http://braintumorsegmentation.org/) (RSNA-ASNR-MICCAI) —
  subject to the dataset's own license; not redistributed here. Pretrained
  weights derived from it remain subject to the dataset's terms.
- Built with [PyTorch](https://pytorch.org), [MONAI](https://monai.io), and
  [Niivue](https://niivue.github.io).
