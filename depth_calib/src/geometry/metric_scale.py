"""Metric-backend single-scale calibration (handoff 6.2).

z'(p) = s * z_metric(p), s > 0. Per-anchor s_i = L_i / l_i with l_i the metric
backend's predicted 3-D length. Robust aggregation in log space, group-weighted,
with an optional geometric-consistency acceptance variant. Fallback s=1 when the
evidence is insufficient or conflicting.
"""
from __future__ import annotations

import numpy as np

from .solver_common import weighted_median
from .types import AnchorGeom, group_weights


def anchor_scale(a: AnchorGeom) -> tuple[float | None, float]:
    """Return (s_i, predicted_length_l_i)."""
    if a.d1 is None or a.d2 is None:
        return None, float("nan")
    x1 = np.asarray(a.q1) * a.d1
    x2 = np.asarray(a.q2) * a.d2
    l_i = float(np.linalg.norm(x1 - x2))
    if not np.isfinite(l_i) or l_i <= 1e-9:
        return None, l_i
    return float(a.length_m / l_i), l_i


def fit_metric_scale(anchors: list[AnchorGeom], cfg: dict) -> dict:
    c = cfg["solver"]["metric_scale"]
    q = cfg["quality"]
    rows = []
    for a in anchors:
        s_i, l_i = anchor_scale(a)
        ok = (s_i is not None and np.isfinite(s_i) and s_i > 0
              and q["min_length_m"] <= l_i <= q["max_length_m"] * 10
              and not a.quality_flags)
        rows.append({"anchor_id": a.anchor_id,
                     "knowledge_group_id": a.knowledge_group_id,
                     "s_i": s_i, "l_i": l_i, "ok": bool(ok)})

    good = [r for r in rows if r["ok"]]
    logs = np.array([np.log(r["s_i"]) for r in good], dtype=float)
    clip_lo, clip_hi = c["log_scale_clip"]
    keep = (logs >= clip_lo) & (logs <= clip_hi)
    good = [g for g, k in zip(good, keep) if k]
    logs = logs[keep]
    if len(good) == 0:
        return {"status": "insufficient", "s": 1.0, "fallback": True,
                "n_used": 0, "per_anchor": rows,
                "notes": "no usable anchors"}

    w = group_weights([g["knowledge_group_id"] for g in good],
                      cap=c["group_weight_cap"])
    s_plain = float(np.exp(weighted_median(logs, w)))
    spread = float(np.percentile(logs, 90) - np.percentile(logs, 10)) if len(logs) > 1 else 0.0

    result = {
        "status": "ok",
        "s": s_plain,
        "s_plain": s_plain,
        "fallback": False,
        "n_used": len(good),
        "n_groups": len({g["knowledge_group_id"] for g in good}),
        "log_spread_p10_p90": spread,
        "per_anchor": rows,
        "notes": "",
    }

    # single-anchor: a scale exists but no consistency check is possible
    if len(good) < c["min_anchors_consistency"]:
        result["status"] = "single_anchor"
        result["notes"] = "single anchor: scale unverifiable, report separately"
        result["s_consistent"] = s_plain
        result["consistent_set"] = [g["anchor_id"] for g in good]
        return result

    # geometric-consistency variant: agreement must come from >=2 distinct knowledge groups
    tol = c["consistency_rel_tol"]
    acc = [g for g, l in zip(good, logs) if abs(l - np.log(s_plain)) <= tol]
    acc_groups = len({g["knowledge_group_id"] for g in acc})
    if len(acc) >= c["min_anchors_consistency"] and \
            acc_groups >= c["min_groups_consistency"]:
        wa = group_weights([g["knowledge_group_id"] for g in acc],
                           cap=c["group_weight_cap"])
        s_cons = float(np.exp(weighted_median([np.log(g["s_i"]) for g in acc], wa)))
    else:
        s_cons = 1.0
        result["status"] = "conflict"
        result["notes"] = (f"only {len(acc)} anchor(s)/{acc_groups} knowledge group(s) "
                           f"within consistency tol {tol}; conservative fallback")
        result["fallback"] = True
    result["s_consistent"] = s_cons
    result["consistent_set"] = [g["anchor_id"] for g in acc]
    return result


def apply_metric_scale(depth: np.ndarray, s: float) -> np.ndarray:
    return np.asarray(depth, dtype=np.float64) * float(s)
