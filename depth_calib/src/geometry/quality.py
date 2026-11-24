"""Endpoint validity and sensitivity checks (handoff 6.5).

Checks use only the image geometry, the backend's raw depth and its validity
mask. Ground truth is never consulted here.
"""
from __future__ import annotations

import numpy as np

from .camera import Camera, sample_depth_bilinear


def evaluate_endpoints(cam: Camera, depth: np.ndarray, p1, p2, cfg: dict,
                       valid_mask: np.ndarray | None = None) -> dict:
    q = cfg["quality"]
    flags: list[str] = []
    p1 = np.asarray(p1, dtype=float).reshape(2)
    p2 = np.asarray(p2, dtype=float).reshape(2)

    for p in (p1, p2):
        if not (0 <= p[0] <= cam.width - 1 and 0 <= p[1] <= cam.height - 1):
            flags.append("endpoint_out_of_bounds")
    sep = float(np.linalg.norm(p1 - p2))
    if sep < q["min_endpoint_separation_px"]:
        flags.append("endpoints_too_close")

    d, ok = sample_depth_bilinear(depth, np.stack([p1, p2]),
                                  valid_mask, eps=q["depth_valid_eps"])
    d1, d2 = float(d[0]), float(d[1])
    if not ok[0] or not ok[1]:
        flags.append("invalid_depth_endpoint")

    out = {"flags": flags, "d1": d1 if ok[0] else None,
           "d2": d2 if ok[1] else None,
           "sep_px": sep, "L0": None, "perturb_max_rel": None}
    if "invalid_depth_endpoint" in flags:
        return out

    L0 = float(np.linalg.norm(d1 * cam.q(p1)[0] - d2 * cam.q(p2)[0]))
    out["L0"] = L0
    if L0 <= 0:
        flags.append("zero_length")
        return out

    r = q["endpoint_perturb_px"]
    rel_depth = []
    for dx, dy in ((r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r)):
        for which in (0, 1):
            pp = [p1.copy(), p2.copy()]
            pp[which] += np.array([dx, dy])
            dd, okp = sample_depth_bilinear(depth, np.stack(pp), valid_mask,
                                            eps=q["depth_valid_eps"])
            if not okp[which] or dd[which] <= 0:
                continue
            d0 = d1 if which == 0 else d2
            rel_depth.append(abs(float(dd[which]) - d0) / d0)
    out["perturb_max_rel"] = float(max(rel_depth)) if rel_depth else None
    if out["perturb_max_rel"] is not None and \
            out["perturb_max_rel"] > q["perturb_depth_rel_change_warn"]:
        flags.append("endpoint_sensitive")
    out["flags"] = flags
    return out


def check_length_range(length_m: float, cfg: dict) -> bool:
    q = cfg["quality"]
    return bool(np.isfinite(length_m)
                and q["min_length_m"] <= length_m <= q["max_length_m"])
