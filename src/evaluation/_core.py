"""Shared evaluation core used by `evaluate_cnn.py` and `evaluate_transformer.py`.

Contains all the per-case inference, calibration, post-processing and summary
logic. The two entry-point scripts call `run_evaluation()` with their own
model + checkpoint after parsing CLI args via `add_common_args()`.
"""
import argparse
import datetime
import functools
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="monai")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import nibabel as nib

CURR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(CURR)
ROOT = os.path.dirname(SRC)
if SRC not in sys.path:
    sys.path.append(SRC)

from configs import config
from evaluation import metrics as M
from evaluation import uncertainty as U
from evaluation import uncertainty_eval as UE
from evaluation import calibration as C
from evaluation import robustness as R
from evaluation import visualize as V
from evaluation import postprocess as PP


# Default post-processing config (overridable via CLI: --et-vmin / --tc-vmin)
ET_VMIN_DEFAULT = 1000   # if total ET volume < ET_VMIN, demote ET->NCR
TC_VMIN_DEFAULT = 0      # 0 disables per-component TC cleanup
ET_TAU = 0.5             # ET probability threshold (0.5 = argmax behaviour)

# V_min sweep grid (used when --vmin-sweep is set). Cheap because it's
# label-space only — no extra inference, just postprocess + Dice per grid pt.
VMIN_SWEEP_ET = (500, 750, 1000, 1500, 2000)
VMIN_SWEEP_TC = (0, 250, 500, 1000, 1500)


# ----------------------------------------------------------------------
# Output-dict adapter
# ----------------------------------------------------------------------
class DictToSegAdapter(nn.Module):
    """Wraps a model whose forward may return ``{"seg": tensor, ...}`` (Phase 4
    onwards: TransResUNet3D with use_uncertainty / use_boundary) so the rest
    of the eval pipeline (sliding-window, TTA, MC dropout) sees a plain tensor.

    Auxiliary outputs (``variance``, ``boundary``) from the most recent forward
    are exposed via ``self.last_aux`` for any caller that wants them. Most of
    the metric pipeline ignores them — they're emitted for diagnostics only,
    and predictive entropy / TTA-variance / MC-dropout already cover the
    uncertainty signal that downstream code consumes.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.last_aux: dict = {}

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict):
            self.last_aux = {k: v for k, v in out.items() if k != "seg"}
            return out["seg"]
        self.last_aux = {}
        return out


def wrap_for_eval(model: nn.Module) -> nn.Module:
    """Idempotently wrap a model so its forward returns just the seg tensor."""
    if isinstance(model, DictToSegAdapter):
        return model
    return DictToSegAdapter(model)


# ----------------------------------------------------------------------
# Checkpoint / model helpers
# ----------------------------------------------------------------------
def detect_output_mode(ckpt_path):
    """Inspect a checkpoint and return 'softmax' (4-channel head) or
    'sigmoid' (3-channel region head) based on the terminal Conv3d's
    out-channels. Phase 6 `full` replaces ``final_conv`` with
    ``fusion_head.final_conv`` so we check both keys."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "final_conv.weight" in sd:
        out_ch = sd["final_conv.weight"].shape[0]
    elif "fusion_head.final_conv.weight" in sd:
        out_ch = sd["fusion_head.final_conv.weight"].shape[0]
    else:
        raise KeyError(
            f"{ckpt_path}: cannot detect output mode — neither "
            "'final_conv.weight' nor 'fusion_head.final_conv.weight' present"
        )
    return "sigmoid" if out_ch == 3 else "softmax"


def state_dict_matches(state, model):
    """True iff state_dict keys match model exactly (no missing/unexpected)."""
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())
    return model_keys == ckpt_keys


def find_latest_checkpoint(logs_dir, model=None, arch_label="model"):
    """Newest run_*/best_model.pth. If `model` is given, skip checkpoints
    whose state_dict doesn't match the model's keys."""
    if not os.path.isdir(logs_dir):
        return None
    runs = [os.path.join(logs_dir, d) for d in os.listdir(logs_dir)
            if d.startswith("run_") and os.path.isdir(os.path.join(logs_dir, d))]
    runs = [r for r in runs if os.path.exists(os.path.join(r, "best_model.pth"))]
    runs.sort(key=os.path.getmtime, reverse=True)
    for r in runs:
        path = os.path.join(r, "best_model.pth")
        if model is None:
            return path
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            if state_dict_matches(state, model):
                return path
            else:
                print(f"[evaluate:{arch_label}] skipping incompatible checkpoint: {path}")
        except Exception as e:
            print(f"[evaluate:{arch_label}] skipping unreadable checkpoint {path}: {e}")
    return None


def list_val_patients(data_root, train_count):
    folders = sorted([f for f in os.listdir(data_root)
                      if os.path.isdir(os.path.join(data_root, f))])
    valid = [p for p in folders
             if os.path.exists(os.path.join(data_root, p, "image.npy"))
             and os.path.exists(os.path.join(data_root, p, "mask.npy"))]
    split = train_count if train_count <= len(valid) else int(len(valid) * 0.8)
    return valid[split:]


def load_case(data_root, pid):
    """Returns (image (5,D,H,W) float32, mask (D,H,W) uint8)."""
    p = os.path.join(data_root, pid)
    img = np.load(os.path.join(p, "image.npy")).astype(np.float32)
    msk = np.load(os.path.join(p, "mask.npy")).astype(np.uint8)
    return img, msk


def save_nifti(arr, path, ref_affine=np.eye(4)):
    nib.save(nib.Nifti1Image(arr.astype(np.float32), ref_affine), path)


def decode_labels(probs_4ch, output_mode="softmax"):
    """Convert 4-channel probability volume to a label map (D,H,W) uint8.

    For softmax: plain argmax.
    For sigmoid: hierarchical decode at tau=0.5 over the synthetic 4-ch rep
        produced by uncertainty._logits_to_4ch_probs.
    """
    probs = np.asarray(probs_4ch)
    if output_mode == "softmax":
        return probs.argmax(0).astype(np.uint8)
    p_wt = 1.0 - probs[0]
    p_tc = probs[1] + probs[3]
    p_et = probs[3]
    label = np.zeros(p_wt.shape, dtype=np.uint8)
    label[p_wt >= 0.5] = 2
    label[p_tc >= 0.5] = 1
    label[p_et >= 0.5] = 3
    return label


# ----------------------------------------------------------------------
# Per-mode inference
# ----------------------------------------------------------------------
@torch.no_grad()
def infer_modes(model, x, roi, overlap, tta=True, mc_T=20, output_mode="softmax",
                tta_extended=False):
    """Returns dict mode -> (probs (4,D,H,W), unc (D,H,W) or None, logits (C,D,H,W) or None).

    When ``tta_extended`` is True the ``tta`` mode runs the 32-view (8 flips x
    4 in-plane rotations) extended TTA from ``uncertainty.tta_predict_extended``.
    """
    out = {}
    logits = U.sw_predict(model, x, roi=roi, overlap=overlap)
    probs = U._logits_to_4ch_probs(logits, output_mode=output_mode)
    eps = 1e-8
    ent = -(probs * (probs + eps).log()).sum(dim=1, keepdim=True)
    out["baseline"] = (probs[0].cpu().numpy(),
                       ent[0, 0].cpu().numpy(),
                       logits[0].cpu().numpy())

    mp, mu, mv = U.mc_dropout_predict(model, x, T=mc_T, roi=roi, overlap=overlap,
                                       output_mode=output_mode)
    if mp is not None:
        out["mc_dropout"] = (mp[0].cpu().numpy(), mu[0, 0].cpu().numpy(), None)

    if tta:
        if tta_extended:
            tp, tu, tv = U.tta_predict_extended(model, x, roi=roi, overlap=overlap,
                                                 output_mode=output_mode)
        else:
            tp, tu, tv = U.tta_predict(model, x, roi=roi, overlap=overlap,
                                        output_mode=output_mode)
        out["tta"] = (tp[0].cpu().numpy(), tu[0, 0].cpu().numpy(), None)

    return out


def apply_temperature(logits_np, T):
    return torch.softmax(torch.from_numpy(logits_np) / T, dim=0).numpy()


@torch.no_grad()
def run_robustness(model, x, gt, roi, overlap, output_mode="softmax"):
    rows = []
    for name, fn in R.perturbation_suite():
        try:
            xp = fn(x)
        except Exception as e:
            rows.append(dict(perturbation=name, error=str(e)))
            continue
        logits = U.sw_predict(model, xp, roi=roi, overlap=overlap)
        probs = U._logits_to_4ch_probs(logits, output_mode=output_mode)[0].cpu().numpy()
        pred = decode_labels(probs, output_mode=output_mode)
        m = M.all_metrics(pred, probs, gt)
        m["perturbation"] = name
        rows.append(m)
    return rows


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def add_common_args(ap):
    ap.add_argument("--checkpoint", default=None,
                    help="Path to best_model.pth. Auto-detected from logs/run_* if omitted.")
    ap.add_argument("--max-cases", type=int, default=None,
                    help="Limit number of validation cases (debug).")
    ap.add_argument("--mc-T", type=int, default=20)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--skip-robustness", action="store_true", default=True,
                    help="Skip the perturbation sweep (default: on).")
    ap.add_argument("--with-robustness", dest="skip_robustness", action="store_false",
                    help="Run the perturbation sweep (default: first 3 cases).")
    ap.add_argument("--robustness-cases", type=int, default=3,
                    help="Number of val cases to run perturbations on. Use a "
                         "large number (e.g. 200) to sweep the full val set.")
    ap.add_argument("--results-dir", default=None,
                    help="Override the output directory. Defaults to the arch-"
                         "specific results folder. A timestamped subfolder "
                         "'eval_YYYYMMDD-HHMMSS/' is created inside, unless "
                         "--run-name is given or --no-subfolder is set.")
    ap.add_argument("--run-name", default=None,
                    help="Custom subfolder name under --results-dir. Defaults to "
                         "'eval_YYYYMMDD-HHMMSS'. Pass --no-subfolder to disable.")
    ap.add_argument("--no-subfolder", action="store_true",
                    help="Write directly into --results-dir without creating a "
                         "per-run subfolder (legacy — overwrites existing files).")
    ap.add_argument("--save-uncertainty-niftis", action="store_true",
                    help="Write per-case uncertainty maps as NIfTI (large).")
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--sw-batch", type=int, default=4,
                    help="Sliding-window batch size. Increase if GPU has spare VRAM.")
    ap.add_argument("--amp-dtype", choices=["fp16", "bf16"], default=None,
                    help="AMP dtype for sliding-window inference. Default: bf16 "
                         "for transformer, fp16 for CNN.")
    ap.add_argument("--et-vmin", type=int, default=ET_VMIN_DEFAULT,
                    help=f"ET volume rescue threshold (default {ET_VMIN_DEFAULT}). "
                         "If total predicted ET < et_vmin, demote ET->NCR.")
    ap.add_argument("--tc-vmin", type=int, default=TC_VMIN_DEFAULT,
                    help="Per-component TC small-component threshold. Demotes "
                         "small isolated TC components -> ED, preserving WT. "
                         "0 disables (default). Try 500 / 1000 to start.")
    ap.add_argument("--vmin-sweep", action="store_true",
                    help="Also evaluate a (et,tc)-vmin grid in label space and "
                         "write vmin_sweep.csv. Cheap — no extra inference.")
    ap.add_argument("--ensemble-ckpts", default=None,
                    help="Glob pattern (e.g. 'logs/run_full_*/snapshot_top*.pth'). "
                         "When set, build_variant is instantiated N times, each "
                         "loaded with one matching checkpoint, and predictions "
                         "are averaged via EnsemblePredictor. Overrides the "
                         "single --checkpoint auto-discovery.")
    ap.add_argument("--tta-extended", action="store_true",
                    help="Use the 32-view extended TTA (8 flips x 4 in-plane "
                         "rotations) instead of the default 8-flip TTA.")


# ----------------------------------------------------------------------
# Main evaluation loop
# ----------------------------------------------------------------------
def run_evaluation(args, arch, model, ckpt, output_mode, results_base_dir):
    """Shared evaluation loop. `arch` is either 'cnn' or 'transformer'.
    `results_base_dir` is the arch-specific default base (e.g. results_cnn/)."""
    device = torch.device(config.DEVICE)
    roi = config.PATCH_SIZE
    tag = f"evaluate:{arch}"

    print(f"[{tag}] arch={arch}  output_mode={output_mode}  checkpoint={ckpt}")
    # Wrap dict-returning models (Phase 4+: variants with use_uncertainty /
    # use_boundary) so sliding-window inference, TTA, MC dropout get a plain
    # seg tensor. Existing variants pass through untouched (their forward
    # already returns a tensor).
    model = wrap_for_eval(model)
    model.eval()

    # ---- Loud device sanity check ----
    cuda_avail = torch.cuda.is_available()
    model_device = next(model.parameters()).device
    print(f"[{tag}] torch={torch.__version__}  cuda_available={cuda_avail}  "
          f"config.DEVICE={config.DEVICE}  model.device={model_device}")
    if cuda_avail:
        print(f"[{tag}] cuda device: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
    if model_device.type != "cuda":
        print(f"[{tag}] !!! WARNING: model is on {model_device.type}, NOT cuda. "
              "Eval will be ~50x slower. Check your torch install:")
        print(f"[{tag}] !!!   python -c \"import torch; print(torch.cuda.is_available(), torch.version.cuda)\"")

    # AMP dtype — bf16 for transformer (matches training), fp16 for CNN.
    if args.amp_dtype == "fp16":
        amp_dtype = torch.float16
    elif args.amp_dtype == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.bfloat16 if arch == "transformer" else torch.float16
    print(f"[{tag}] sw_predict amp_dtype={amp_dtype}  sw_batch={args.sw_batch}")
    _orig_sw = U.sw_predict
    U.sw_predict = functools.partial(
        _orig_sw, sw_batch_size=args.sw_batch, amp_dtype=amp_dtype,
    )

    has_dropout = U.model_has_dropout(model)
    print(f"[{tag}] model has dropout: {has_dropout} — MC Dropout {'enabled' if has_dropout else 'SKIPPED'}")

    # Resolve post-processing thresholds for this run.
    et_vmin = int(args.et_vmin)
    tc_vmin = int(args.tc_vmin)
    do_sweep = bool(args.vmin_sweep)
    print(f"[{tag}] post-process: et_vmin={et_vmin}  tc_vmin={tc_vmin}  "
          f"sweep={'on' if do_sweep else 'off'}")

    val_pids = list_val_patients(config.TRAIN_DATA_PATH, config.TRAIN_COUNT)
    if args.max_cases:
        val_pids = val_pids[:args.max_cases]
    print(f"[{tag}] val cases: {len(val_pids)}")

    base_dir = args.results_dir if args.results_dir is not None else results_base_dir
    if args.no_subfolder:
        out_dir = base_dir
    else:
        run_name = args.run_name or (
            "eval_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        out_dir = os.path.join(base_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{tag}] results_dir={out_dir}")
    unc_dir = os.path.join(out_dir, "uncertainty")
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(unc_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    TS_VOXELS_PER_CASE = 8000
    CAL_VOXELS_PER_CASE = 4000

    # ---------- Resumable on-disk cache ----------
    cache_dir = os.path.join(out_dir, "_resume_cache")
    os.makedirs(cache_dir, exist_ok=True)
    pc_csv = os.path.join(out_dir, "per_case_metrics.csv")
    unc_csv = os.path.join(cache_dir, "per_case_uncertainty.csv")

    if os.path.exists(pc_csv):
        df_existing = pd.read_csv(pc_csv)
        per_case = df_existing.to_dict("records")
        done_pids = set(df_existing["patient_id"].astype(str).tolist())
        print(f"[{tag}] resuming — {len(done_pids)} patients already complete")
    else:
        per_case = []
        done_pids = set()

    if os.path.exists(unc_csv):
        df_unc = pd.read_csv(unc_csv)
    else:
        df_unc = pd.DataFrame(columns=["patient_id", "mode", "u_mean", "risk", "spear"])

    per_case_unc = {"baseline": [], "tta": [], "mc_dropout": []}
    risk_for_aurc = {"baseline": [], "tta": [], "mc_dropout": []}
    spear_corr = {"baseline": [], "tta": [], "mc_dropout": []}
    for _, r in df_unc.iterrows():
        m = r["mode"]
        if m in per_case_unc:
            per_case_unc[m].append(float(r["u_mean"]))
            risk_for_aurc[m].append(float(r["risk"]))
            spear_corr[m].append(float(r["spear"]))

    ts_logits_samples = []
    ts_target_samples = []
    for f in sorted(os.listdir(cache_dir)):
        if f.endswith("_tslog.npy"):
            ts_logits_samples.append(np.load(os.path.join(cache_dir, f)))
        elif f.endswith("_tstgt.npy"):
            ts_target_samples.append(np.load(os.path.join(cache_dir, f)))

    # Two parallel buckets: brain-restricted (standard) and positive-only
    # (calibration on tumor voxels). The pre-fix code sampled over the entire
    # volume which was dominated by trivial non-brain voxels and gave
    # implausibly low ECE. v2 cache filename below ensures stale v1 caches
    # from the pre-fix code are not silently reused.
    calib_samples = {m: {r: {"conf": [], "correct": []} for r in M.REGIONS}
                     for m in ["baseline", "ts", "tta", "mc_dropout"]}
    calib_samples_pos = {m: {r: {"conf": [], "correct": []} for r in M.REGIONS}
                         for m in ["baseline", "ts", "tta", "mc_dropout"]}
    calib_cache = os.path.join(cache_dir, "calib_samples_v2.npz")
    calib_cache_v1 = os.path.join(cache_dir, "calib_samples.npz")
    if os.path.exists(calib_cache):
        data = np.load(calib_cache, allow_pickle=True)
        calib_samples = data["payload"].item()
        calib_samples_pos = data["payload_pos"].item()
    elif os.path.exists(calib_cache_v1) and done_pids:
        print(f"[{tag}] !!! WARNING: stale v1 calibration cache detected at "
              f"{calib_cache_v1}. Pre-fix calibration is unreliable (sampled "
              f"non-brain voxels). To regenerate, delete {out_dir} and re-run, "
              f"OR delete per_case_metrics.csv to force recomputation.")

    ts_logits_path = os.path.join(out_dir, "_ts_logits_cache")
    os.makedirs(ts_logits_path, exist_ok=True)

    robustness_rows = []
    rb_csv = os.path.join(out_dir, "robustness.csv")
    if os.path.exists(rb_csv):
        robustness_rows = pd.read_csv(rb_csv).to_dict("records")

    # V_min sweep: per-case Dice grid for (et_vmin, tc_vmin). Resumable.
    sweep_rows = []
    sweep_csv = os.path.join(out_dir, "vmin_sweep_per_case.csv")
    sweep_done = set()  # (pid, mode, et_v, tc_v) tuples already computed
    if do_sweep and os.path.exists(sweep_csv):
        df_sw = pd.read_csv(sweep_csv)
        sweep_rows = df_sw.to_dict("records")
        for r in sweep_rows:
            sweep_done.add((str(r["patient_id"]), r["mode"],
                            int(r["et_vmin"]), int(r["tc_vmin"])))

    rng = np.random.RandomState(0)

    def _sample_voxels(arr_4d, n, mask=None):
        D, H, W = arr_4d.shape[1:]
        total = D * H * W
        if mask is not None and mask.any():
            flat_mask = mask.ravel()
            pool = np.where(flat_mask)[0]
        else:
            pool = np.arange(total)
        n = min(n, len(pool))
        idx = rng.choice(pool, n, replace=False) if len(pool) > n else pool
        flat = arr_4d.reshape(arr_4d.shape[0], -1)[:, idx].T
        return idx, flat

    unc_rows_buf = list(df_unc.to_dict("records"))

    case_times: list[float] = []  # wall-clock seconds per non-cached case
    loop_start = time.time()
    for i, pid in enumerate(val_pids):
        if pid in done_pids:
            print(f"[{i+1}/{len(val_pids)}] {pid} — skipped (cached)")
            continue
        case_t0 = time.time()
        print(f"[{i+1}/{len(val_pids)}] {pid}", flush=True)
        img, gt = load_case(config.TRAIN_DATA_PATH, pid)
        x = torch.from_numpy(img).unsqueeze(0).to(device)
        fg = img[4] > 0.5
        gt_flat = gt.reshape(-1)

        modes = infer_modes(model, x, roi=roi, overlap=args.overlap,
                            tta=not args.no_tta, mc_T=args.mc_T,
                            output_mode=output_mode,
                            tta_extended=getattr(args, "tta_extended", False))

        if output_mode == "softmax":
            baseline_logits = modes["baseline"][2]
            idx_ts, log_samp = _sample_voxels(baseline_logits, TS_VOXELS_PER_CASE, mask=fg)
            ts_logits_samples.append(log_samp)
            ts_target_samples.append(gt_flat[idx_ts])
            np.save(os.path.join(ts_logits_path, f"{pid}_logits.npy"),
                    baseline_logits.astype(np.float16))
        else:
            baseline_logits = modes["baseline"][2]

        for mode_name, (probs, unc, _logits) in modes.items():
            pred = decode_labels(probs, output_mode=output_mode)
            m = M.all_metrics(pred, probs, gt)
            row = dict(patient_id=pid, mode=mode_name, **m)
            per_case.append(row)

            if mode_name in ("baseline", "tta", "mc_dropout"):
                pred_pp = PP.postprocess_full(
                    probs, tau_et=ET_TAU, et_vmin=et_vmin, tc_vmin=tc_vmin,
                )
                m_pp = M.all_metrics(pred_pp, probs, gt)
                per_case.append(dict(patient_id=pid, mode=f"{mode_name}_post", **m_pp))

                if do_sweep:
                    for ev in VMIN_SWEEP_ET:
                        for tv in VMIN_SWEEP_TC:
                            key = (pid, mode_name, ev, tv)
                            if key in sweep_done:
                                continue
                            pp = PP.postprocess_full(
                                probs, tau_et=ET_TAU, et_vmin=ev, tc_vmin=tv,
                            )
                            mm = M.dice_only_metrics(pp, gt)
                            sweep_rows.append({
                                "patient_id": pid, "mode": mode_name,
                                "et_vmin": ev, "tc_vmin": tv, **mm,
                            })

            for region in M.REGIONS:
                pr = M.probs_to_region_probs(probs)[region]
                gr = M.labels_to_regions(gt)[region]
                # Brain-restricted ECE: sample only inside brain foreground
                conf, correct = C.region_conf_correct(
                    pr, gr, sample=CAL_VOXELS_PER_CASE, mask=fg,
                )
                calib_samples[mode_name][region]["conf"].append(conf.astype(np.float32))
                calib_samples[mode_name][region]["correct"].append(correct.astype(np.uint8))
                # Positive-only ECE: sample only inside this region's tumor voxels
                pos_mask = fg & gr.astype(bool)
                conf_p, correct_p = C.region_conf_correct(
                    pr, gr, sample=CAL_VOXELS_PER_CASE, mask=pos_mask,
                )
                calib_samples_pos[mode_name][region]["conf"].append(conf_p.astype(np.float32))
                calib_samples_pos[mode_name][region]["correct"].append(correct_p.astype(np.uint8))

            if unc is not None:
                u_mean = float(unc[fg].mean()) if fg.any() else float(unc.mean())
                per_case_unc[mode_name].append(u_mean)
                d = [m[f"dice_{r}"] for r in M.REGIONS]
                d = [v for v in d if np.isfinite(v)]
                risk = 1.0 - float(np.mean(d)) if d else float("nan")
                risk_for_aurc[mode_name].append(risk)
                err = (pred != gt).astype(np.uint8)
                spear_corr[mode_name].append(UE.spearman_unc_error(unc, err))
                if args.save_uncertainty_niftis:
                    save_nifti(unc, os.path.join(unc_dir, f"{pid}_{mode_name}_unc.nii.gz"))

        viz_mode = "tta" if "tta" in modes else "baseline"
        v_probs, v_unc, _ = modes[viz_mode]
        v_pred = decode_labels(v_probs, output_mode=output_mode)
        if i < 5 and v_unc is not None:
            V.overlay_uncertainty(img[1], v_pred, v_unc,
                                  os.path.join(plot_dir, f"{pid}_overlay.png"),
                                  title=f"{pid} ({viz_mode})")
            V.error_vs_uncertainty((v_pred != gt).astype(np.uint8), v_unc,
                                   os.path.join(plot_dir, f"{pid}_error_vs_unc.png"),
                                   title=pid)

        if (not args.skip_robustness) and i < args.robustness_cases:
            rb = run_robustness(model, x, gt, roi=roi, overlap=args.overlap,
                                output_mode=output_mode)
            for r in rb:
                r["patient_id"] = pid
            robustness_rows.extend(rb)
            pd.DataFrame(robustness_rows).to_csv(rb_csv, index=False)

        pd.DataFrame(per_case).to_csv(pc_csv, index=False)
        if do_sweep and sweep_rows:
            pd.DataFrame(sweep_rows).to_csv(sweep_csv, index=False)

        for m_name in ["baseline", "tta", "mc_dropout"]:
            if m_name in modes and modes[m_name][1] is not None:
                unc_rows_buf.append({
                    "patient_id": pid, "mode": m_name,
                    "u_mean": per_case_unc[m_name][-1],
                    "risk": risk_for_aurc[m_name][-1],
                    "spear": spear_corr[m_name][-1],
                })
        pd.DataFrame(unc_rows_buf).to_csv(unc_csv, index=False)

        if output_mode == "softmax":
            np.save(os.path.join(cache_dir, f"{pid}_tslog.npy"), log_samp.astype(np.float32))
            np.save(os.path.join(cache_dir, f"{pid}_tstgt.npy"), gt_flat[idx_ts].astype(np.int64))
        np.savez(calib_cache,
                 payload=np.array(calib_samples, dtype=object),
                 payload_pos=np.array(calib_samples_pos, dtype=object))

        done_pids.add(pid)

        del modes, baseline_logits, x, img
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        case_elapsed = time.time() - case_t0
        case_times.append(case_elapsed)
        # Running average over the cases processed in this run (resumed runs
        # only see the cases done after the resume).
        avg_s = sum(case_times) / len(case_times)
        remaining = len(val_pids) - (i + 1)
        eta_s = avg_s * remaining
        eta_h, eta_m = divmod(int(eta_s) // 60, 60)
        print(f"    case took {case_elapsed:6.1f}s  |  avg {avg_s:5.1f}s/case  |  "
              f"ETA {eta_h:d}h{eta_m:02d}m  ({remaining} left)", flush=True)

    # -------- Temperature scaling --------
    scaler = None
    if output_mode == "softmax":
        # If everything was resumed (no fresh cases processed this run), the
        # TS sample buffers are empty. In that case the prior run already
        # produced temperature_scale rows in per_case_metrics.csv and the
        # logits cache was deleted at the end of that run — skip the TS pass.
        if not ts_logits_samples:
            ts_already_done = any(r.get("mode") == "temperature_scale" for r in per_case)
            if ts_already_done:
                print(f"[{tag}] TS pass skipped — already done in prior run, "
                      f"using cached temperature_scale rows from per_case_metrics.csv")
                scaler = None
            else:
                raise RuntimeError(
                    "TS sample buffer is empty AND no prior temperature_scale "
                    "rows exist. Did the per-case loop run at all?"
                )
        else:
            print(f"[{tag}] fitting temperature scaling on sampled voxels...")
            L = torch.from_numpy(np.concatenate(ts_logits_samples).astype(np.float32)).to(device)
            T = torch.from_numpy(np.concatenate(ts_target_samples).astype(np.int64)).to(device)
            scaler = C.TemperatureScaler()
            scaler.fit(L, T)
            print(f"[{tag}] T = {scaler.T:.4f}")
            del L, T

        # If scaler is None we already established that prior-run TS rows
        # exist for every pid; the per-case loop has nothing to do.
        if scaler is None:
            print(f"[{tag}] no fresh TS work — per_case_metrics.csv already "
                  f"contains the temperature_scale rows from the prior run.")
            existing_ts_pids = set()  # short-circuit the loop
        else:
            print(f"[{tag}] applying temperature scaling per case...")
            # On a resumed run, the TS pass from a prior completed run already
            # populated `temperature_scale` / `temperature_scale_post` rows in
            # per_case_metrics.csv AND deleted the per-case logits cache. Skip
            # any pid whose cache file is gone but whose TS row is already saved.
            existing_ts_pids = {
                r["patient_id"] for r in per_case
                if r.get("mode") == "temperature_scale"
            }
        for pid in (val_pids if scaler is not None else []):
            logits_npy = os.path.join(ts_logits_path, f"{pid}_logits.npy")
            if not os.path.exists(logits_npy):
                if pid in existing_ts_pids:
                    # Prior run already produced the TS metrics for this case;
                    # nothing to redo. (Calibration sampling for this pid is
                    # also lost from this run's in-memory buffer, but the
                    # summary numbers in per_case_metrics.csv are preserved.)
                    continue
                raise FileNotFoundError(
                    f"TS logits cache missing for {pid} AND no prior "
                    f"temperature_scale row in per_case_metrics.csv. "
                    f"Run a fresh eval into a new --run-name."
                )
            logits = np.load(logits_npy).astype(np.float32)
            probs = apply_temperature(logits, scaler.T)
            gt = np.load(os.path.join(config.TRAIN_DATA_PATH, pid, "mask.npy")).astype(np.uint8)
            pred = probs.argmax(0).astype(np.uint8)
            m = M.all_metrics(pred, probs, gt)
            per_case.append(dict(patient_id=pid, mode="temperature_scale", **m))
            pred_pp = PP.postprocess_full(
                probs, tau_et=ET_TAU, et_vmin=et_vmin, tc_vmin=tc_vmin,
            )
            m_pp = M.all_metrics(pred_pp, probs, gt)
            per_case.append(dict(patient_id=pid, mode="temperature_scale_post", **m_pp))

            if do_sweep:
                for ev in VMIN_SWEEP_ET:
                    for tv in VMIN_SWEEP_TC:
                        key = (pid, "temperature_scale", ev, tv)
                        if key in sweep_done:
                            continue
                        pp = PP.postprocess_full(
                            probs, tau_et=ET_TAU, et_vmin=ev, tc_vmin=tv,
                        )
                        mm = M.dice_only_metrics(pp, gt)
                        sweep_rows.append({
                            "patient_id": pid, "mode": "temperature_scale",
                            "et_vmin": ev, "tc_vmin": tv, **mm,
                        })
            # Reload brain foreground for this case (image[4] is the binary FG channel)
            fg_ts = np.load(
                os.path.join(config.TRAIN_DATA_PATH, pid, "image.npy"), mmap_mode="r",
            )[4].astype(np.float32) > 0.5
            for region in M.REGIONS:
                pr = M.probs_to_region_probs(probs)[region]
                gr = M.labels_to_regions(gt)[region]
                conf, correct = C.region_conf_correct(
                    pr, gr, sample=CAL_VOXELS_PER_CASE, mask=fg_ts,
                )
                calib_samples["ts"][region]["conf"].append(conf.astype(np.float32))
                calib_samples["ts"][region]["correct"].append(correct.astype(np.uint8))
                pos_mask = fg_ts & gr.astype(bool)
                conf_p, correct_p = C.region_conf_correct(
                    pr, gr, sample=CAL_VOXELS_PER_CASE, mask=pos_mask,
                )
                calib_samples_pos["ts"][region]["conf"].append(conf_p.astype(np.float32))
                calib_samples_pos["ts"][region]["correct"].append(correct_p.astype(np.uint8))
            del logits, probs, gt, fg_ts
            os.remove(os.path.join(ts_logits_path, f"{pid}_logits.npy"))
        try:
            os.rmdir(ts_logits_path)
        except OSError:
            pass
    else:
        print(f"[{tag}] output_mode='sigmoid' — skipping temperature scaling "
              "(scalar TS over CrossEntropy is undefined for region-wise sigmoid heads).")
        try:
            for f in os.listdir(ts_logits_path):
                os.remove(os.path.join(ts_logits_path, f))
            os.rmdir(ts_logits_path)
        except OSError:
            pass

    # -------- Calibration from pooled samples --------
    # Two views: brain-restricted (standard) and positive-only (tumor voxels).
    # Reliability diagrams use the brain-restricted version (the standard one).
    calib_summary = {}      # brain-restricted
    calib_summary_pos = {}  # positive-only (tumor voxels)
    for mode_name in ["baseline", "ts", "tta", "mc_dropout"]:
        if not calib_samples[mode_name]["ET"]["conf"]:
            continue
        calib_summary[mode_name] = {}
        calib_summary_pos[mode_name] = {}
        for region in M.REGIONS:
            conf = np.concatenate(calib_samples[mode_name][region]["conf"])
            correct = np.concatenate(calib_samples[mode_name][region]["correct"])
            ece, mce_u, ba_u, bc_u, bn_u = C.ece_uniform(conf, correct, n_bins=15)
            ace, mce_a, ba_a, bc_a, bn_a = C.ace_quantile(conf, correct, n_bins=15)
            calib_summary[mode_name][region] = dict(ece=ece, ace=ace, mce=max(mce_u, mce_a))
            V.reliability_diagram(bc_u, ba_u, bn_u, ece,
                                  os.path.join(plot_dir, f"reliability_{mode_name}_{region}.png"),
                                  title=f"{mode_name} {region} (brain)")

            # Positive-only ECE: tumor voxels only. May be empty for cases
            # with no tumor of this region — concatenation handles that.
            conf_p_list = calib_samples_pos[mode_name][region]["conf"]
            corr_p_list = calib_samples_pos[mode_name][region]["correct"]
            conf_p = np.concatenate(conf_p_list) if conf_p_list else np.empty(0, dtype=np.float32)
            corr_p = np.concatenate(corr_p_list) if corr_p_list else np.empty(0, dtype=np.uint8)
            if conf_p.size > 0:
                ece_p, _, _, _, _ = C.ece_uniform(conf_p, corr_p, n_bins=15)
                ace_p, _, _, _, _ = C.ace_quantile(conf_p, corr_p, n_bins=15)
            else:
                ece_p, ace_p = float("nan"), float("nan")
            calib_summary_pos[mode_name][region] = dict(
                ece=ece_p, ace=ace_p, n_voxels=int(conf_p.size),
            )

    # -------- AURC + Spearman + Dice@coverage (clinical framing) --------
    aurc_summary, spear_summary, dice_at_cov_summary = {}, {}, {}
    COVERAGE_LEVELS = (0.7, 0.8, 0.9)
    for mode_name in ["baseline", "tta", "mc_dropout"]:
        if not per_case_unc[mode_name]:
            continue
        cov, risks, aurc = UE.risk_coverage_curve(per_case_unc[mode_name],
                                                  risk_for_aurc[mode_name])
        aurc_summary[mode_name] = aurc
        V.risk_coverage_plot(cov, risks, aurc,
                             os.path.join(plot_dir, f"risk_coverage_{mode_name}.png"),
                             title=f"Risk-Coverage ({mode_name})")
        s = [v for v in spear_corr[mode_name] if np.isfinite(v)]
        spear_summary[mode_name] = float(np.mean(s)) if s else float("nan")
        dice_per_case = [1.0 - r for r in risk_for_aurc[mode_name]]
        dice_at_cov_summary[mode_name] = UE.dice_at_coverage(
            per_case_unc[mode_name], dice_per_case, coverages=COVERAGE_LEVELS,
        )

    df_pc = pd.DataFrame(per_case)
    df_pc.to_csv(os.path.join(out_dir, "per_case_metrics.csv"), index=False)

    if "baseline" in df_pc["mode"].values:
        V.dice_boxplot(df_pc[df_pc["mode"] == "baseline"],
                       save_path=os.path.join(plot_dir, "dice_boxplot.png"))

    if robustness_rows:
        pd.DataFrame(robustness_rows).to_csv(
            os.path.join(out_dir, "robustness.csv"), index=False)

    # -------- Summary table --------
    summary_rows = []
    base_label = "Transformer baseline" if arch == "transformer" else "3D UNet baseline"
    pp_suffix = f"(et_vmin={et_vmin}" + (f", tc_vmin={tc_vmin})" if tc_vmin > 0 else ")")
    mode_label = {
        "baseline": base_label,
        "temperature_scale": "+ Temperature Scale",
        "mc_dropout": "+ MC Dropout",
        "tta": "+ TTA",
        "ts": "+ Temperature Scale",
        "baseline_post":         f"+ Post-process {pp_suffix}",
        "tta_post":              f"+ TTA + Post-process {pp_suffix}",
        "mc_dropout_post":       f"+ MC Dropout + Post-process {pp_suffix}",
        "temperature_scale_post": f"+ Temp Scale + Post-process {pp_suffix}",
    }
    for mode_name in ["baseline", "ts", "mc_dropout", "tta",
                      "baseline_post", "tta_post", "mc_dropout_post",
                      "temperature_scale_post"]:
        if mode_name == "ts":
            sub = df_pc[df_pc["mode"] == "temperature_scale"]
        else:
            sub = df_pc[df_pc["mode"] == mode_name]
        if sub.empty:
            continue
        cal = calib_summary.get(mode_name, {})
        eces = [cal[r]["ece"] for r in cal]
        aces = [cal[r]["ace"] for r in cal]
        cal_pos = calib_summary_pos.get(mode_name, {})
        eces_pos = [cal_pos[r]["ece"] for r in cal_pos
                    if np.isfinite(cal_pos[r]["ece"])]
        aces_pos = [cal_pos[r]["ace"] for r in cal_pos
                    if np.isfinite(cal_pos[r]["ace"])]
        row = {
            "Method":   mode_label[mode_name],
            "Dice ET":  sub["dice_ET"].mean(),
            "Dice TC":  sub["dice_TC"].mean(),
            "Dice WT":  sub["dice_WT"].mean(),
            "ECE_brain": float(np.mean(eces)) if eces else float("nan"),
            "ACE_brain": float(np.mean(aces)) if aces else float("nan"),
            "ECE_pos":   float(np.mean(eces_pos)) if eces_pos else float("nan"),
            "ACE_pos":   float(np.mean(aces_pos)) if aces_pos else float("nan"),
            "HD95":     sub[["hd95_ET", "hd95_TC", "hd95_WT"]].stack().mean(),
            "NSD":      sub[["nsd_ET", "nsd_TC", "nsd_WT"]].stack().mean(),
            "AUC":      sub[["auc_ET", "auc_TC", "auc_WT"]].stack().mean(),
            "AURC":     aurc_summary.get(mode_name, float("nan")),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    pd.options.display.float_format = "{:.4f}".format
    print(f"\n========== EVALUATION SUMMARY ({arch}) ==========")
    print(summary_df.to_string(index=False))
    print(f"\nSpearman(unc, error): {spear_summary}")
    if dice_at_cov_summary:
        print("\n---- Dice @ coverage (clinical framing) ----")
        print("If a clinician reviews the most uncertain (1-c) of cases,")
        print("the auto-segmented remainder achieves this mean Dice:")
        for mode_name, d_at_c in dice_at_cov_summary.items():
            base = d_at_c.get(1.0, float("nan"))
            parts = [f"c={c:.0%}: {d_at_c[c]:.4f}" for c in COVERAGE_LEVELS]
            print(f"  {mode_name:>11s} | full(c=100%): {base:.4f}  |  "
                  + "  ".join(parts))
    if scaler is not None:
        print(f"\nTemperature T: {scaler.T:.4f}")
    print(f"Outputs: {out_dir}")

    # Stash Dice@coverage as its own CSV for easy reference in the report
    if dice_at_cov_summary:
        cov_rows = []
        for mode_name, d_at_c in dice_at_cov_summary.items():
            row = {"mode": mode_name, "dice_full": d_at_c.get(1.0, float("nan"))}
            for c in COVERAGE_LEVELS:
                row[f"dice_at_cov_{int(c*100)}"] = d_at_c[c]
            cov_rows.append(row)
        pd.DataFrame(cov_rows).to_csv(
            os.path.join(out_dir, "dice_at_coverage.csv"), index=False)

    # -------- V_min sweep aggregation --------
    sweep_summary = None
    if do_sweep and sweep_rows:
        df_sw = pd.DataFrame(sweep_rows)
        df_sw.to_csv(sweep_csv, index=False)
        agg = df_sw.groupby(["mode", "et_vmin", "tc_vmin"]).agg(
            dice_ET=("dice_ET", "mean"),
            dice_TC=("dice_TC", "mean"),
            dice_WT=("dice_WT", "mean"),
            n_cases=("dice_ET", "size"),
        ).reset_index()
        agg["dice_mean"] = agg[["dice_ET", "dice_TC", "dice_WT"]].mean(axis=1)
        agg = agg.sort_values(["mode", "dice_mean"], ascending=[True, False])
        agg.to_csv(os.path.join(out_dir, "vmin_sweep.csv"), index=False)

        # Per-mode best (by mean Dice). Useful for picking thresholds.
        best = agg.loc[agg.groupby("mode")["dice_mean"].idxmax()].reset_index(drop=True)
        sweep_summary = best.to_dict("records")
        print(f"\n---- V_min sweep best per mode (by mean Dice) ----")
        print(best.to_string(index=False))

    # Phase 6 — record ensemble + extended-TTA + sw_overlap so the eval folder
    # is self-describing for the SOTA-push numbers.
    ensemble_members = None
    if isinstance(ckpt, str) and ";" in ckpt:
        ensemble_members = [os.path.basename(p) for p in ckpt.split(";")]
    with open(os.path.join(out_dir, "evaluation_meta.json"), "w") as f:
        json.dump({
            "checkpoint": ckpt,
            "arch": arch,
            "output_mode": output_mode,
            "n_val_cases": len(val_pids),
            "temperature": (scaler.T if scaler is not None else None),
            "calibration_brain": calib_summary,
            "calibration_pos":   calib_summary_pos,
            "calibration_note":  ("calibration_brain restricts ECE/ACE to brain "
                                  "foreground (fg=image[4]>0.5). calibration_pos "
                                  "restricts to tumor voxels of each region "
                                  "(fg AND gt_region). The pre-fix code sampled "
                                  "the full volume which produced trivially low "
                                  "ECE — see ECE audit notes."),
            "spearman_unc_error": spear_summary,
            "aurc": aurc_summary,
            "dice_at_coverage": dice_at_cov_summary,
            "has_dropout": has_dropout,
            "postprocess": {"et_vmin": et_vmin, "tc_vmin": tc_vmin},
            "vmin_sweep_best": sweep_summary,
            "ensemble_members": ensemble_members,
            "tta_extended": bool(getattr(args, "tta_extended", False)),
            "sw_overlap": float(args.overlap),
        }, f, indent=2, default=float)
