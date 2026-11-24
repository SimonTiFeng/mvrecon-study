"""Synthetic calibration recovery and identifiability tests (handoff 8.1).

Covers: metric single-scale recovery, depth-affine recovery, inverse-depth-affine
recovery, same-plane degeneracy (rank < 2), insufficient-anchor states, and outlier
robustness of the metric-scale median.
"""
import numpy as np
import pytest
import yaml
import pathlib

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.geometry.camera import Camera
from src.geometry.types import AnchorGeom
from src.geometry.metric_scale import fit_metric_scale, apply_metric_scale
from src.geometry.depth_affine import fit_depth_affine, apply_depth_affine
from src.geometry.inverse_depth_affine import (fit_inverse_depth_affine,
                                               apply_inverse_depth_affine)

CFG = yaml.safe_load((pathlib.Path(__file__).resolve().parents[1]
                      / "configs" / "pilot.yaml").read_text(encoding="utf-8"))
CAM = Camera(500.0, 500.0, 320.0, 240.0, 640, 480)


def make_scene(points_pairs):
    """pairs: list of ((X1,Y1,Z1),(X2,Y2,Z2)) in camera coords. Returns anchors
    carrying q vectors and true lengths; depth fields filled by the caller."""
    out = []
    for i, (P1, P2) in enumerate(points_pairs):
        P1, P2 = np.array(P1, float), np.array(P2, float)
        uv1 = CAM.project(P1)[0]
        uv2 = CAM.project(P2)[0]
        out.append(dict(
            anchor_id=f"a{i}", object_group_id=f"o{i}",
            knowledge_group_id=f"g{i}", p1=uv1, p2=uv2,
            q1=CAM.q(uv1)[0], q2=CAM.q(uv2)[0],
            length_m=float(np.linalg.norm(P1 - P2)),
            true_z1=float(P1[2]), true_z2=float(P2[2])))
    return out


PAIRS = [
    ((0.5, 0.1, 2.0), (-0.4, 0.2, 2.4)),
    ((0.2, -0.3, 3.5), (0.9, -0.1, 3.9)),
    ((-0.7, 0.4, 5.0), (-0.1, 0.5, 5.6)),
    ((0.6, 0.5, 7.0), (0.1, 0.2, 7.8)),
]


def anchors_with(rows, d1_key, d2_key):
    out = []
    for r in rows:
        out.append(AnchorGeom(anchor_id=r["anchor_id"],
                              object_group_id=r["object_group_id"],
                              knowledge_group_id=r["knowledge_group_id"],
                              p1=r["p1"], p2=r["p2"], q1=r["q1"], q2=r["q2"],
                              length_m=r["length_m"],
                              d1=r.get(d1_key), d2=r.get(d2_key)))
    return out


# ---------------- metric scale ----------------
def test_metric_scale_recovery():
    rows = make_scene(PAIRS)
    alpha = 0.8
    for r in rows:
        r["d1"] = alpha * r["true_z1"]
        r["d2"] = alpha * r["true_z2"]
    anchors = anchors_with(rows, "d1", "d2")
    res = fit_metric_scale(anchors, CFG)
    assert res["status"] == "ok"
    # l_i = alpha * L_i  =>  s_i = 1/alpha
    assert res["s_plain"] == pytest.approx(1.0 / alpha, rel=2e-6)
    assert res["n_used"] == len(PAIRS)


def test_metric_scale_single_anchor_flagged():
    rows = make_scene(PAIRS[:1])
    for r in rows:
        r["d1"] = 0.8 * r["true_z1"]; r["d2"] = 0.8 * r["true_z2"]
    res = fit_metric_scale(anchors_with(rows, "d1", "d2"), CFG)
    assert res["status"] == "single_anchor"
    assert res["s"] == pytest.approx(1.25, rel=1e-6)


def test_metric_scale_outlier_robust():
    rows = make_scene(PAIRS)
    for r in rows:
        r["d1"] = 0.8 * r["true_z1"]; r["d2"] = 0.8 * r["true_z2"]
    rows[0]["length_m"] *= 10.0          # one grossly wrong anchor
    res = fit_metric_scale(anchors_with(rows, "d1", "d2"), CFG)
    assert abs(res["s_plain"] - 1.25) < 0.02


# ---------------- depth affine ----------------
def test_depth_affine_recovery():
    rows = make_scene(PAIRS)
    a_true, b_true = 2.0, 0.5
    for r in rows:
        r["d1"] = (r["true_z1"] - b_true) / a_true
        r["d2"] = (r["true_z2"] - b_true) / a_true
    res = fit_depth_affine(anchors_with(rows, "d1", "d2"), CFG)
    assert res["status"] == "ok", res["notes"]
    assert res["a"] == pytest.approx(a_true, rel=1e-3)
    assert res["b"] == pytest.approx(b_true, rel=1e-2, abs=1e-2)
    assert res["rank"] == 2


def test_depth_affine_same_plane_degenerate():
    same = [((0.5, 0.1, 3.0), (-0.4, 0.2, 3.0)),
            ((0.2, -0.3, 3.0), (0.9, -0.1, 3.0)),
            ((-0.7, 0.4, 3.0), (-0.1, 0.5, 3.0))]
    rows = make_scene(same)
    for r in rows:
        r["d1"] = (r["true_z1"] - 0.5) / 2.0
        r["d2"] = (r["true_z2"] - 0.5) / 2.0
    res = fit_depth_affine(anchors_with(rows, "d1", "d2"), CFG)
    assert res["status"] in ("degenerate", "multi_solution", "ill_conditioned")
    assert res["rank"] < 2
    assert res["fallback"] is True


def test_depth_affine_insufficient():
    rows = make_scene(PAIRS[:2])
    for r in rows:
        r["d1"] = r["true_z1"]; r["d2"] = r["true_z2"]
    res = fit_depth_affine(anchors_with(rows, "d1", "d2"), CFG)
    assert res["status"] == "insufficient"
    assert res["fallback"] is True


# ---------------- inverse-depth affine ----------------
def test_inverse_affine_recovery():
    rows = make_scene(PAIRS)
    a_true, b_true = 1.5, 0.1
    for r in rows:
        r["r1"] = (1.0 / r["true_z1"] - b_true) / a_true
        r["r2"] = (1.0 / r["true_z2"] - b_true) / a_true
    anchors = [AnchorGeom(anchor_id=r["anchor_id"],
                          object_group_id=r["object_group_id"],
                          knowledge_group_id=r["knowledge_group_id"],
                          p1=r["p1"], p2=r["p2"], q1=r["q1"], q2=r["q2"],
                          length_m=r["length_m"], r1=r["r1"], r2=r["r2"])
               for r in rows]
    res = fit_inverse_depth_affine(anchors, CFG)
    assert res["status"] == "ok", res["notes"]
    assert res["a"] == pytest.approx(a_true, rel=5e-3)
    assert res["b"] == pytest.approx(b_true, rel=5e-2, abs=5e-3)
    assert res["rank"] == 2


def test_apply_functions_roundtrip():
    d = np.array([1.0, 2.0, 4.0])
    assert np.allclose(apply_metric_scale(d, 2.0), d * 2.0)
    z = apply_depth_affine(d, 2.0, 0.5)
    assert np.allclose(z, 2.0 * d + 0.5)
    r = np.array([0.1, 0.2, 0.3])
    zz = apply_inverse_depth_affine(r, 1.5, 0.1)
    assert np.allclose(zz, 1.0 / (1.5 * r + 0.1))
