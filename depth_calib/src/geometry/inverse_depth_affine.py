"""Inverse-depth affine calibration for relative outputs (handoff 6.4).

z'(p) = 1 / (a * r(p) + b), a > 0, denominator positive on the valid region.
Length: X(p) = q(p) / (a r(p) + b). Same robust objective, multi-start,
Jacobian-rank diagnostics and explicit fallback as the depth-affine solver.
"""
from __future__ import annotations

import numpy as np

from .solver_common import (dedupe_solutions, multistart_minimize,
                            normalized_rank, numeric_jacobian, soft_huber)
from .types import AnchorGeom, group_weights


def _prepare(anchors: list[AnchorGeom], cfg: dict):
    q = cfg["quality"]
    rows, Q1, Q2, R1, R2, L, wids = [], [], [], [], [], [], []
    for a in anchors:
        if a.r1 is None or a.r2 is None or a.quality_flags:
            continue
        if not (q["min_length_m"] <= a.length_m <= q["max_length_m"]):
            continue
        if not (np.isfinite(a.r1) and np.isfinite(a.r2)):
            continue
        Q1.append(np.asarray(a.q1)); Q2.append(np.asarray(a.q2))
        R1.append(a.r1); R2.append(a.r2); L.append(a.length_m)
        wids.append(a.knowledge_group_id)
        rows.append(a.anchor_id)
    return (np.array(Q1, float), np.array(Q2, float),
            np.array(R1, float), np.array(R2, float),
            np.array(L, float), group_weights(wids), rows)


def fit_inverse_depth_affine(anchors: list[AnchorGeom], cfg: dict,
                             r_valid_min: float | None = None) -> dict:
    c = cfg["solver"]["inverse_depth_affine"]
    diag = cfg["solver"]["diagnostics"]
    Q1, Q2, R1, R2, L, W, ids = _prepare(anchors, cfg)
    n = len(ids)
    if n < c["min_anchors"]:
        return {"status": "insufficient", "method": "inverse_depth_affine",
                "n_used": n, "a": 1.0, "b": 0.0, "fallback": True,
                "notes": f"{n} usable anchor(s) < {c['min_anchors']} required",
                "per_anchor": []}

    b_scale = float(np.median(np.abs(np.concatenate([R1, R2]))))
    if b_scale <= 0:
        b_scale = 1.0
    delta = c["huber_delta_log"]
    logL = np.log(L)
    denom_min = c["denom_min"]
    probe = 0.0 if r_valid_min is None else float(r_valid_min)

    def denoms(theta, which=(1, 2)):
        a = np.exp(theta[0]); b = theta[1] * b_scale
        return a * (R1 if which == 1 else R2) + b

    def residuals(theta):
        a = np.exp(theta[0]); b = theta[1] * b_scale
        d1 = a * R1 + b
        d2 = a * R2 + b
        pen = np.maximum(0.0, denom_min - d1) + np.maximum(0.0, denom_min - d2)
        # enforce positivity where the whole valid region lives
        pen = pen + max(0.0, denom_min - (a * probe + b)) * np.ones_like(d1)
        X1 = Q1 / d1[:, None]
        X2 = Q2 / d2[:, None]
        Lh = np.maximum(np.linalg.norm(X1 - X2, axis=1), 1e-12)
        r = np.log(Lh) - logL
        return np.where(pen > 0, r + 1e3 * pen, r)

    def loss(theta):
        r = residuals(theta)
        if not np.all(np.isfinite(r)):
            return 1e12
        return float(np.sum(W * soft_huber(r, delta)))

    lo_a, hi_a = np.log(1e-8), np.log(1e8)
    la0 = [max(lo_a, np.log(0.05)), 0.0, min(hi_a, np.log(20.0))]
    bh0 = [-2.0, 0.0, 2.0]
    starts = [np.array([la, bh]) for la in la0 for bh in bh0]
    bounds = [(lo_a, hi_a), (-10.0, 10.0)]
    sols = dedupe_solutions(multistart_minimize(loss, starts, bounds, c["max_iter"]))

    best = sols[0]
    a = float(np.exp(best["x"][0])); b = float(best["x"][1] * b_scale)
    J = numeric_jacobian(residuals, best["x"])
    rk = normalized_rank(J, diag["rank_tol"])
    res = residuals(best["x"])
    denom_ok = bool(np.all(a * np.concatenate([R1, R2]) + b > denom_min)
                    and (r_valid_min is None or a * probe + b > denom_min))

    ok = (rk["rank"] >= 2 and denom_ok and np.all(np.isfinite(res)))
    status = "ok"
    notes = ""
    if not denom_ok:
        status = "invalid_denominator"
        notes = "a*r+b <= denom_min inside the valid region"
    elif rk["rank"] < 2:
        status = "degenerate"
        notes = f"effective rank {rk['rank']} < 2"
    elif rk["cond"] > diag["cond_warn"]:
        status = "ill_conditioned"
        notes = f"condition number {rk['cond']:.3g} > {diag['cond_warn']:.3g}"

    return {
        "status": status, "method": "inverse_depth_affine", "n_used": n,
        "a": a, "b": b, "loss": best["loss"],
        "median_abs_log_resid": float(np.median(np.abs(res))),
        "rank": rk["rank"], "cond": rk["cond"],
        "singular_values": rk["singular_values"],
        "n_distinct_minima": len(sols),
        "solutions": [{"a": float(np.exp(s["x"][0])),
                       "b": float(s["x"][1] * b_scale),
                       "loss": s["loss"]} for s in sols[:5]],
        "fallback": status != "ok",
        "notes": notes,
        "per_anchor": [{"anchor_id": i, "log_resid": float(r)}
                       for i, r in zip(ids, res)],
    }


def apply_inverse_depth_affine(r: np.ndarray, a: float, b: float,
                               denom_min: float = 1e-8) -> np.ndarray:
    d = float(a) * np.asarray(r, dtype=np.float64) + float(b)
    z = np.full_like(d, np.nan)
    ok = d > denom_min
    z[ok] = 1.0 / d[ok]
    return z
