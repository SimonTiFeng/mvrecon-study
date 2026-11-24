"""Geometry pipeline: anchors + backend depth -> calibrated depth (evaluator side).

Only the isolated evaluator imports this module. It never runs inside the VLM
agent's process, and nothing here is fed back to the measurement agent.
"""
from __future__ import annotations

import numpy as np

from .camera import Camera, sample_depth_bilinear
from .depth_affine import apply_depth_affine, fit_depth_affine
from .inverse_depth_affine import (apply_inverse_depth_affine,
                                   fit_inverse_depth_affine)
from .metric_scale import apply_metric_scale, fit_metric_scale
from .quality import evaluate_endpoints
from .types import AnchorGeom


def _anchors_from_norm(norm: dict, cam: Camera, depth: np.ndarray,
                       cfg: dict, use_disparity: bool) -> tuple[list[AnchorGeom], list[dict]]:
    out, diag = [], []
    for a in norm.get("anchors", []):
        p1 = np.asarray(a["p1_xy"], dtype=float)
        p2 = np.asarray(a["p2_xy"], dtype=float)
        qa = evaluate_endpoints(cam, depth, p1, p2, cfg)
        vals, ok = sample_depth_bilinear(depth, np.stack([p1, p2]), eps=cfg["quality"]["depth_valid_eps"])
        geom = AnchorGeom(
            anchor_id=a["anchor_id"], object_group_id=a["object_group_id"],
            knowledge_group_id=a["knowledge_group_id"],
            p1=p1, p2=p2, q1=cam.q(p1)[0], q2=cam.q(p2)[0],
            length_m=float(a["length_m"]),
            length_interval_m=(tuple(a["length_interval_m"])
                               if a.get("length_interval_m") else None),
            quality_flags=tuple(qa["flags"]))
        if ok[0] and ok[1]:
            if use_disparity:
                geom.r1, geom.r2 = float(vals[0]), float(vals[1])
            else:
                geom.d1, geom.d2 = float(vals[0]), float(vals[1])
        out.append(geom)
        diag.append({"anchor_id": a["anchor_id"], "flags": qa["flags"],
                     "perturb_max_rel": qa["perturb_max_rel"]})
    return out, diag


def metric_focal_factor(K: np.ndarray, processed_shape: tuple[int, int],
                        original_size: tuple[int, int], factor: float = 300.0) -> float:
    Hp, Wp = processed_shape
    W, H = original_size
    fx = K[0, 0] * (Wp / W)
    fy = K[1, 1] * (Hp / H)
    return float(((fx + fy) / 2.0) / factor)


def calibrate(*, norm: dict, depth: np.ndarray, cam: Camera,
              cfg: dict, backend_spec: dict,
              processed_shape: tuple[int, int] | None = None) -> dict:
    """Return calibrated depth plus full diagnostics; never touches GT."""
    W = norm["image_width"]
    H = norm["image_height"]
    if (cam.width, cam.height) != (W, H):
        raise ValueError(f"camera {cam.width}x{cam.height} != image {W}x{H}")
    K = cam.K
    conv = backend_spec["depth_convention"]

    d = np.asarray(depth, dtype=np.float64)
    factor = 1.0
    if backend_spec.get("metric") and conv == "z_depth_metric":
        if cfg["backend_inference"]["metric_conversion"] == "focal_over_300_once":
            if processed_shape is None:
                raise ValueError("processed_shape required for metric focal factor")
            factor = metric_focal_factor(K, processed_shape, (W, H))
            d = d * factor

    result = {"backend_metric_conversion_factor": factor,
              "depth_convention": conv}
    if norm.get("status") != "ok" or not norm.get("anchors"):
        result.update({"mode": "abstain", "coverage": 0.0, "fallback": True,
                       "pred_depth": d})
        return result

    use_disparity = conv in ("disparity", "inverse_depth")
    anchors, adiag = _anchors_from_norm(norm, cam, d, cfg, use_disparity)
    result["anchor_diagnostics"] = adiag

    if not use_disparity and backend_spec.get("metric"):
        sol = fit_metric_scale(anchors, cfg)
        result.update({"mode": "metric_scale", "solver": sol})
        result["pred_depth_plain"] = apply_metric_scale(d, sol["s"])
        s_cons = sol.get("s_consistent", sol["s"])
        result["pred_depth_consistent"] = apply_metric_scale(d, s_cons)
        accept = sol["status"] == "ok" if \
            cfg["solver"]["metric_scale"]["conservative_requires_ok"] else True
        result["pred_depth_conservative"] = apply_metric_scale(
            d, s_cons if accept else 1.0)
        result["pred_depth"] = (result["pred_depth_consistent"]
                                if cfg["solver"]["metric_scale"]["use_consistent_variant"]
                                else result["pred_depth_plain"])
        result["fallback"] = bool(sol.get("fallback"))
        result["coverage"] = 0.0 if sol.get("fallback") else 1.0
    elif not use_disparity:
        sol = fit_depth_affine(anchors, cfg)
        result.update({"mode": "depth_affine", "solver": sol})
        result["pred_depth"] = apply_depth_affine(d, sol["a"], sol["b"])
        result["fallback"] = bool(sol["fallback"])
        result["coverage"] = 0.0 if sol["fallback"] else 1.0
    else:
        r_valid_min = float(np.nanmin(d)) if np.isfinite(d).any() else None
        sol = fit_inverse_depth_affine(anchors, cfg, r_valid_min=r_valid_min)
        result.update({"mode": "inverse_depth_affine", "solver": sol})
        result["pred_depth"] = apply_inverse_depth_affine(
            d, sol["a"], sol["b"],
            cfg["solver"]["inverse_depth_affine"]["denom_min"])
        result["fallback"] = bool(sol["fallback"])
        result["coverage"] = 0.0 if sol["fallback"] else 1.0
    return result
