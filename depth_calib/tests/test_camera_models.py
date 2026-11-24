"""Camera model tests: pinhole equivalence and fisheye (pycolmap) rays."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.geometry.camera import Camera


def test_pinhole_q_matches_formula():
    cam = Camera(500, 520, 320, 240, 640, 480)
    uv = np.array([[100.0, 80.0], [320.0, 240.0]])
    q = cam.q(uv)
    assert np.allclose(q[:, 0], (uv[:, 0] - 320) / 500)
    assert np.allclose(q[:, 1], (uv[:, 1] - 240) / 520)
    assert np.allclose(q[:, 2], 1.0)


def test_thin_prism_center_ray():
    params = [3411.42, 3410.02, 3024.0, 2016.0,
              0.21047, 0.21102, -5.36231e-06, 0.00051541,
              -0.158023, 0.406856, -8.46499e-05, 0.000861313]
    cam = Camera(params[0], params[1], params[2], params[3], 6048, 4032,
                 model="THIN_PRISM_FISHEYE", params=params)
    q = cam.q(np.array([[3024.0, 2016.0]]))
    assert np.allclose(q[0, :2], 0.0, atol=5e-3)
    assert q[0, 2] == 1.0
    qc = cam.q(np.array([[0.0, 0.0]]))
    assert np.linalg.norm(qc[0, :2]) > 0.5     # corner ray is far off-axis


def test_thin_prism_roundtrip():
    params = [3411.42, 3410.02, 3024.0, 2016.0,
              0.21047, 0.21102, -5.36231e-06, 0.00051541,
              -0.158023, 0.406856, -8.46499e-05, 0.000861313]
    cam = Camera(params[0], params[1], params[2], params[3], 6048, 4032,
                 model="THIN_PRISM_FISHEYE", params=params)
    uv = np.array([[3024.0, 2016.0], [100.0, 200.0], [5900.0, 3900.0]])
    q = cam.q(uv)
    X = q * np.array([2.0, 3.0, 5.0])[:, None]
    uv2 = cam.project(X)
    assert np.allclose(uv2, uv, atol=1e-6)
