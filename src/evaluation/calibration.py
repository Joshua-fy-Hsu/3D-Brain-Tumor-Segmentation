"""Calibration: ECE, ACE, MCE, Temperature Scaling, reliability diagrams."""
import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Binning
# ----------------------------------------------------------------------
def _bin_ece(conf, correct, edges):
    """Generic weighted-bin calibration error given bin edges."""
    n = len(conf)
    bin_acc, bin_conf, bin_count = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            bin_acc.append(0.0); bin_conf.append(0.0); bin_count.append(0); continue
        bin_acc.append(float(correct[m].mean()))
        bin_conf.append(float(conf[m].mean()))
        bin_count.append(int(m.sum()))
    bin_acc = np.array(bin_acc); bin_conf = np.array(bin_conf); bin_count = np.array(bin_count)
    w = bin_count / max(n, 1)
    diffs = np.abs(bin_acc - bin_conf)
    ece = float((w * diffs).sum())
    mce = float(diffs[bin_count > 0].max()) if (bin_count > 0).any() else float("nan")
    return ece, mce, bin_acc, bin_conf, bin_count


def ece_uniform(conf, correct, n_bins=15):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return _bin_ece(conf, correct, edges)


def ace_quantile(conf, correct, n_bins=15):
    """Adaptive: equal-frequency bins."""
    if len(conf) == 0:
        return float("nan"), float("nan"), np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(conf, qs)
    edges[0], edges[-1] = 0.0, 1.0
    edges = np.unique(edges)
    return _bin_ece(conf, correct, edges)


# ----------------------------------------------------------------------
# Per-region calibration: confidence = sum of class probs in region,
# correctness = ground-truth voxel belongs to region.
# ----------------------------------------------------------------------
def region_conf_correct(prob_region, gt_region, sample=500_000, seed=0, mask=None):
    """Binary calibration samples for one region.

    `mask` (optional bool array, same shape as prob_region): if given, voxels
    where mask==False are excluded BEFORE sampling. Pass the brain foreground
    here for brain-restricted ECE; pass `brain & gt_region` for positive-only
    ECE on tumor voxels. Without a mask, non-brain background voxels (which
    are trivially conf≈1 / correct=1) dominate the sample and ECE collapses
    to ~0 — see ECE audit notes.
    """
    p = np.asarray(prob_region).ravel().astype(np.float32)
    g = np.asarray(gt_region).ravel().astype(np.uint8)
    if mask is not None:
        m = np.asarray(mask).ravel().astype(bool)
        p, g = p[m], g[m]
    if p.size == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.uint8)
    pred = (p >= 0.5).astype(np.uint8)
    conf = np.where(pred == 1, p, 1.0 - p)
    correct = (pred == g).astype(np.uint8)
    if conf.size > sample:
        rng = np.random.RandomState(seed)
        idx = rng.choice(conf.size, sample, replace=False)
        conf, correct = conf[idx], correct[idx]
    return conf, correct


# ----------------------------------------------------------------------
# Temperature Scaling — single scalar T, optimised by NLL on stored logits.
# Operates on aggregated (N, C) logits + (N,) class targets sampled from val.
# ----------------------------------------------------------------------
class TemperatureScaler:
    def __init__(self):
        self.T = 1.0

    def fit(self, logits, targets, lr=0.01, max_iter=200):
        """logits: (N,C) float tensor, targets: (N,) long tensor."""
        device = logits.device
        T = torch.nn.Parameter(torch.ones(1, device=device))
        opt = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter)
        nll = torch.nn.CrossEntropyLoss()

        def closure():
            opt.zero_grad()
            loss = nll(logits / T.clamp(min=1e-3), targets)
            loss.backward()
            return loss

        opt.step(closure)
        self.T = float(T.detach().clamp(min=1e-3).item())
        return self.T

    def apply(self, logits):
        return logits / self.T
