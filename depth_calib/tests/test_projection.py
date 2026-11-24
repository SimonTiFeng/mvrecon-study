"""Projection, coordinate-transform and depth-sampling tests (handoff 8.1)."""
import numpy as np
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.geometry.camera import (Camera, apply_h, crop_transform,
                                 sample_depth_bilinear, scale_matrix)


def make_cam():
    return Camera(fx=500.0, fy=520.0, cx=320.0, cy=240.0, width=640, height=480)


def test_ray_roundtrip():
    cam = make_cam()
    uv = np.array([[100.0, 80.0], [320.0, 240.0], [639.0, 479.0]])
    q = cam.q(uv)
    assert np.allclose(q[:, 2], 1.0)
    z = np.array([2.0, 3.0, 5.0])
    X = cam.backproject(uv, z)
    assert np.allclose(cam.project(X), uv, atol=1e-9)


def test_crop_transform_preserves_projection():
    cam = make_cam()
    P = np.array([0.4, -0.2, 3.0])
    uv = cam.project(P)[0]
    x0, y0 = 90, 60
    uv_c = apply_h(crop_transform(x0, y0), uv)[0]
    assert np.allclose(uv_c, uv - np.array([x0, y0]))
    cam_c = cam.transformed(crop_transform(x0, y0))
    assert np.allclose(cam_c.project(P)[0], uv_c, atol=1e-9)


def test_resize_preserves_projection():
    cam = make_cam()
    P = np.array([-0.5, 0.3, 4.0])
    uv = cam.project(P)[0]
    A = scale_matrix(0.5, 0.5)
    uv_s = apply_h(A, uv)[0]
    cam_s = cam.transformed(A)
    assert np.allclose(cam_s.project(P)[0], uv_s, atol=1e-9)


def test_depth_sampling():
    d = np.arange(1, 21, dtype=float).reshape(4, 5)
    v, ok = sample_depth_bilinear(d, np.array([[0, 0], [2, 1], [1.5, 1.5]]))
    assert ok.all()
    assert v[0] == pytest.approx(d[0, 0])
    assert v[1] == pytest.approx(d[1, 2])
    assert v[2] == pytest.approx((d[1, 1] + d[1, 2] + d[2, 1] + d[2, 2]) / 4)


def test_depth_sampling_out_of_bounds_and_invalid():
    d = np.ones((4, 5))
    d[0, 0] = 0.0
    v, ok = sample_depth_bilinear(d, np.array([[10, 10], [0, 0], [-1, 2]]), eps=1e-6)
    assert not ok.any()
