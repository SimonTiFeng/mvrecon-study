"""Offline end-to-end pipeline test: synthetic depth -> anchors -> calibration -> metrics."""
import sys
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.geometry.camera import Camera
from src.geometry.pipeline import calibrate
from src.evaluation.metrics import evaluate_depth
from src.evaluation.oracle import optimal_global_scale_rmse

CFG = load_config(ROOT / "configs" / "pilot.yaml")
W, H = 160, 120
CAM = Camera(200.0, 200.0, W / 2, H / 2, W, H)


def synth_gt():
    """GT z-depth: three flat billboards at different depths (no discontinuities
    under the chosen endpoints, so the quality gate keeps all anchors)."""
    z = np.full((H, W), 8.0)                 # far background
    z[10:50, 10:55] = 1.5
    z[10:50, 85:150] = 2.5
    z[70:110, 10:150] = 4.0
    return z


def endpoint_pair(depth, uv1, uv2):
    d1 = depth[int(round(uv1[1])), int(round(uv1[0]))]
    d2 = depth[int(round(uv2[1])), int(round(uv2[0]))]
    p1 = CAM.backproject(np.array(uv1), np.array([d1]))[0]
    p2 = CAM.backproject(np.array(uv2), np.array([d2]))[0]
    return float(np.linalg.norm(p1 - p2))


def make_norm(gt):
    pairs = [((20, 30), (45, 30)), ((100, 30), (140, 30)), ((30, 90), (120, 90))]
    anchors = []
    for i, (a, b) in enumerate(pairs):
        anchors.append({"anchor_id": f"a{i}", "object_group_id": f"o{i}",
                        "knowledge_group_id": f"g{i}", "description": "d",
                        "p1_xy": list(map(float, a)), "p2_xy": list(map(float, b)),
                        "length_m": endpoint_pair(gt, a, b),
                        "length_interval_m": None,
                        "prior_kind": "identified_standard",
                        "visible_evidence": "v", "identity_uncertainty": "u"})
    return {"image_id": "t", "image_width": W, "image_height": H,
            "status": "ok", "anchors": anchors, "abstain_reason": None}


def test_metric_calibration_and_eval():
    gt = synth_gt()
    norm = make_norm(gt)
    pred = gt * 0.7                      # scale error
    cal = calibrate(norm=norm, depth=pred, cam=CAM, cfg=CFG,
                    backend_spec={"metric": True, "depth_convention": "z_depth_metric"},
                    processed_shape=(H, W))
    # metric_conversion focal/300 applied by design; use effective factor
    factor = cal["backend_metric_conversion_factor"]
    s = cal["solver"]["s"]
    assert s * factor == pytest.approx(1 / 0.7, rel=5e-4)
    m_before = evaluate_depth(pred * factor, gt)
    m_after = evaluate_depth(cal["pred_depth"], gt)
    assert m_after["abs_rel"] < 1e-6
    assert m_before["abs_rel"] > 0.2


def test_affine_calibration():
    gt = synth_gt()
    norm = make_norm(gt)
    pred = 0.5 * gt + 0.3               # depth-space affine error
    cal = calibrate(norm=norm, depth=pred, cam=CAM, cfg=CFG,
                    backend_spec={"metric": False, "depth_convention": "z_depth"},
                    processed_shape=(H, W))
    assert cal["mode"] == "depth_affine"
    assert cal["solver"]["status"] == "ok"
    m = evaluate_depth(cal["pred_depth"], gt)
    assert m["abs_rel"] < 5e-3


def test_oracle_and_metrics():
    gt = synth_gt()
    pred = gt * 0.6 + 0.1
    mask = np.ones_like(gt, dtype=bool)
    s = optimal_global_scale_rmse(pred, gt, mask)
    assert 1.0 < s < 3.0
    m0 = evaluate_depth(pred, gt)
    m1 = evaluate_depth(pred * s, gt)
    assert m1["abs_rel"] < m0["abs_rel"]
    assert evaluate_depth(gt, gt)["delta1"] == 1.0


def test_abstain_marks_coverage_zero():
    gt = synth_gt()
    cal = calibrate(norm={"status": "abstain", "anchors": [],
                          "image_width": W, "image_height": H},
                    depth=gt, cam=CAM, cfg=CFG,
                    backend_spec={"metric": True, "depth_convention": "z_depth_metric"},
                    processed_shape=(H, W))
    assert cal["mode"] == "abstain" and cal["coverage"] == 0.0
    assert cal["fallback"] is True
