"""Depth-affine calibration for relative outputs in depth space (handoff 6.3).

z'(p) = a * d(p) + b, a > 0. For an endpoint pair:
  U = d(p1) q(p1) - d(p2) q(p2),  V = q(p1) - q(p2)
  Lhat(a,b) = || a U + b V ||
Objective: weighted soft-Huber on log-length residuals. Multi-start deterministic
optimization, residual-Jacobian rank check, degeneracy detection and explicit
fallback. No use of ground-truth masks anywhere.
"""
from __future__ import annotations

import numpy as np

from .solver_common import (dedupe_solutions, multistart_minimize,
                            normalized_rank, numeric_jacobian, soft_huber)
from .types import AnchorGeom, group_weights


def _prepare(anchors: list[AnchorGeom], cfg: dict):
    q = cfg["quality"]
    rows, U, V, L, wids = [], [], [], [], []
    for a in anchors:
        if a.d1 is None or a.d2 is None or a.quality_flags:
            continue
        if not (q["min_length_m"] <= a.length_m <= q["max_length_m"]):
            continue
        if not (np.isfinite(a.d1) and np.isfinite(a.d2)
                and a.d1 > q["depth_valid_eps"] and a.d2 > q["depth_valid_eps"]):
            continue
        u = np.asarray(a.q1) * a.d1 - np.asarray(a.q2) * a.d2
        v = np.asarray(a.q1) - np.asarray(a.q2)
        U.append(u); V.append(v); L.append(a.length_m)
        wids.append(a.knowledge_group_id)
        rows.append(a.anchor_id)
    return (np.array(U, float), np.array(V, float), np.array(L, float),
            group_weights(wids), rows)


def fit_depth_affine(anchors: list[AnchorGeom], cfg: dict) -> dict:
    c = cfg["solver"]["depth_affine"]
    diag = cfg["solver"]["diagnostics"]
    U, V, L, W, ids = _prepare(anchors, cfg)
    n = len(ids)
    if n < c["min_anchors"]:
        return {"status": "insufficient", "method": "depth_affine",
                "n_used": n, "a": 1.0, "b": 0.0, "fallback": True,
                "notes": f"{n} usable anchor(s) < {c['min_anchors']} required",
                "per_anchor": []}

    depths = np.concatenate([np.linalg.norm(U, axis=1) / np.maximum(
        np.linalg.norm(V, axis=1), 1e-12)])
    b_scale = float(np.median(depths)) if np.median(depths) > 0 else 1.0
    delta = c["huber_delta_log"]
    logL = np.log(L)

    def lengths(theta):
        a = np.exp(theta[0]); b = theta[1] * b_scale
        Wv = a * U + b * V
        return np.linalg.norm(Wv, axis=1)

    def residuals(theta):
        Lh = np.maximum(lengths(theta), 1e-12)
        return np.log(Lh) - logL

    def loss(theta):
        r = residuals(theta)
        if not np.all(np.isfinite(r)):
            return 1e12
        return float(np.sum(W * soft_huber(r, delta)))

    lo_a, hi_a = np.log(c["a_bounds"][0]), np.log(c["a_bounds"][1])
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

    ok = (rk["rank"] >= 2 and np.isfinite(best["loss"])
          and np.all(np.isfinite(res)))
    status = "ok"
    notes = ""
    if not ok:
        status = "degenerate"
        notes = f"effective rank {rk['rank']} < 2; parameters not identifiable"
    elif rk["cond"] > diag["cond_warn"]:
        status = "ill_conditioned"
        notes = f"condition number {rk['cond']:.3g} > {diag['cond_warn']:.3g}"
    elif len(sols) > 1 and sols[1]["loss"] < best["loss"] * 1.05 + 1e-9:
        status = "multi_solution"
        notes = f"{len(sols)} distinct near-equal minima"

    fallback = False
    if status != "ok" and c["shrink_on_conflict"]:
        fallback = True

    return {
        "status": status, "method": "depth_affine", "n_used": n,
        "a": 1.0 if fallback else a, "b": 0.0 if fallback else b,
        "a_fit": a, "b_fit": b,
        "loss": best["loss"],
        "median_abs_log_resid": float(np.median(np.abs(res))),
        "rank": rk["rank"], "cond": rk["cond"],
        "singular_values": rk["singular_values"],
        "n_distinct_minima": len(sols),
        "solutions": [{"a": float(np.exp(s["x"][0])),
                       "b": float(s["x"][1] * b_scale),
                       "loss": s["loss"]} for s in sols[:5]],
        "fallback": fallback, "notes": notes,
        "per_anchor": [{"anchor_id": i, "log_resid": float(r)}
                       for i, r in zip(ids, res)],
    }


def apply_depth_affine(depth: np.ndarray, a: float, b: float,
                       eps: float = 1e-8) -> np.ndarray:
    z = float(a) * np.asarray(depth, dtype=np.float64) + float(b)
    return np.maximum(z, eps)
