"""Metric depth evaluation (handoff 10). Evaluator-only; GT never leaves this side."""
from __future__ import annotations

import numpy as np


def _valid(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None):
    m = np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    if mask is not None:
        m &= np.asarray(mask).astype(bool)
    return m


def abs_rel(pred, gt, mask=None):
    m = _valid(pred, gt, mask)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(pred[m] - gt[m]) / gt[m]))


def rmse_m(pred, gt, mask=None):
    m = _valid(pred, gt, mask)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((pred[m] - gt[m]) ** 2)))


def delta(pred, gt, thr=1.25, mask=None):
    m = _valid(pred, gt, mask)
    if not m.any():
        return float("nan")
    ratio = np.maximum(pred[m] / gt[m], gt[m] / pred[m])
    return float(np.mean(ratio < thr))


def evaluate_depth(pred, gt, mask=None, delta_thresholds=(1.25,)) -> dict:
    out = {"abs_rel": abs_rel(pred, gt, mask),
           "rmse_m": rmse_m(pred, gt, mask),
           "n_pixels": int(_valid(pred, gt, mask).sum())}
    for t in delta_thresholds:
        key = "delta1" if abs(t - 1.25) < 1e-9 else f"delta_{t}"
        out[key] = delta(pred, gt, t, mask)
    return out


def paired_bootstrap(values: np.ndarray, n_samples: int, seed: int,
                     stat=np.mean) -> dict:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": 0}
    boots = [stat(rng.choice(v, size=v.size, replace=True)) for _ in range(n_samples)]
    return {"mean": float(np.mean(v)), "median": float(np.median(v)),
            "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
            "n": int(v.size)}
