"""Shared robust-loss and diagnostics helpers for the three calibration solvers."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def soft_huber(r: np.ndarray, delta: float) -> np.ndarray:
    """Smooth Huber (pseudo-Huber); quadratic for |r|<<delta, linear far out."""
    return delta ** 2 * (np.sqrt(1.0 + (r / delta) ** 2) - 1.0)


def huber_weights(r: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-12))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        raise ValueError("weights sum to zero")
    return float(v[np.searchsorted(cw, 0.5 * cw[-1])])


def normalized_rank(J: np.ndarray, rank_tol: float) -> dict:
    """Effective rank of the residual Jacobian after column normalization."""
    J = np.asarray(J, dtype=float)
    if J.size == 0 or not np.all(np.isfinite(J)):
        return {"rank": 0, "cond": float("inf"), "singular_values": [0.0] * J.shape[1]}
    col = np.linalg.norm(J, axis=0)
    col[col < 1e-15] = 1.0
    Jn = J / col
    sv = np.linalg.svd(Jn, compute_uv=False)
    if sv.size == 0 or sv[0] <= 0:
        return {"rank": 0, "cond": float("inf"), "singular_values": sv.tolist()}
    rank = int(np.sum(sv / sv[0] > rank_tol))
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    return {"rank": rank, "cond": cond, "singular_values": sv.tolist()}


def numeric_jacobian(fun, theta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of a vector-valued residual fun(theta)."""
    theta = np.asarray(theta, dtype=float)
    r0 = np.asarray(fun(theta), dtype=float)
    J = np.zeros((r0.size, theta.size))
    for k in range(theta.size):
        h = eps * max(1.0, abs(theta[k]))
        tp, tm = theta.copy(), theta.copy()
        tp[k] += h
        tm[k] -= h
        J[:, k] = (np.asarray(fun(tp), dtype=float) - np.asarray(fun(tm), dtype=float)) / (2 * h)
    return J


def multistart_minimize(fun, starts: list[np.ndarray], bounds: list[tuple],
                        max_iter: int) -> list[dict]:
    sols = []
    for x0 in starts:
        x0 = np.clip(np.asarray(x0, dtype=float),
                     [b[0] for b in bounds], [b[1] for b in bounds])
        res = minimize(fun, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": max_iter})
        sols.append({"x": res.x.copy(), "loss": float(res.fun),
                     "success": bool(res.success), "message": str(res.message)})
    return sols


def dedupe_solutions(sols: list[dict], rel_tol: float = 0.05) -> list[dict]:
    """Merge solutions within rel_tol (per coordinate scale) keeping the lowest loss."""
    kept: list[dict] = []
    for s in sorted(sols, key=lambda q: q["loss"]):
        dup = False
        for k in kept:
            scale = np.maximum(1.0, np.maximum(np.abs(s["x"]), np.abs(k["x"])))
            if np.all(np.abs(s["x"] - k["x"]) / scale <= rel_tol):
                dup = True
                break
        if not dup:
            kept.append(s)
    return kept
