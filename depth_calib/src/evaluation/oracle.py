"""Oracle diagnostics (handoff 9.3): evaluator-only, never fed to the agent."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def optimal_global_scale_rmse(pred: np.ndarray, gt: np.ndarray,
                              mask: np.ndarray) -> float:
    """Least-squares scale s minimizing sum (s*pred - gt)^2 over valid pixels."""
    m = np.isfinite(pred) & np.isfinite(gt) & (gt > 0) & (pred > 0) & mask
    p, g = pred[m], gt[m]
    denom = float(np.sum(p * p))
    return float(np.sum(p * g) / denom) if denom > 0 else 1.0


def optimal_global_scale_absrel(pred: np.ndarray, gt: np.ndarray,
                                mask: np.ndarray) -> float:
    m = np.isfinite(pred) & np.isfinite(gt) & (gt > 0) & (pred > 0) & mask
    p, g = pred[m], gt[m]
    if p.size == 0:
        return 1.0
    r = np.log(g / p)
    return float(np.exp(np.median(r))) if np.all(np.isfinite(r)) else 1.0


def optimal_depth_affine(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                         max_samples: int | None = None, seed: int = 0) -> dict:
    """Robust (median-abs-log) depth-affine a,b fitted on GT; analysis only.

    Subsamples deterministically for bounded cost; returns a, b and the achieved
    median |log error|.
    """
    m = np.isfinite(pred) & np.isfinite(gt) & (gt > 0) & (pred > 0) & mask
    p, g = pred[m], gt[m]
    if max_samples is not None and max_samples > 0 and p.size > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(p.size, max_samples, replace=False)
        p, g = p[idx], g[idx]
    if p.size == 0:
        return {"a": 1.0, "b": 0.0, "median_abs_log_err": float("nan")}

    med = float(np.median(p))
    def loss(theta):
        a = np.exp(theta[0]); b = theta[1] * med
        z = a * p + b
        if np.any(z <= 1e-9):
            return 10.0
        return float(np.median(np.abs(np.log(z) - np.log(g))))

    best = None
    for la in (np.log(0.2), 0.0, np.log(5.0)):
        for bh in (-1.0, 0.0, 1.0):
            r = minimize_scalar(lambda t: loss(np.array([t, bh])),
                                bounds=(np.log(0.05), np.log(20.0)),
                                method="bounded",
                                options={"xatol": 1e-3})
            cand = (r.fun, float(np.exp(r.x)), float(bh * med))
            if best is None or cand[0] < best[0]:
                best = cand
    return {"a": best[1], "b": best[2], "median_abs_log_err": best[0]}
