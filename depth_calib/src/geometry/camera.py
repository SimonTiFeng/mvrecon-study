"""Pinhole camera, depth sampling and view<->original coordinate transforms.

Conventions (handoff 4.3 / 6.1):
  - pixel centers at integer coordinates, origin top-left, x right, y down;
  - K = [[fx,0,cx],[0,fy,cy],[0,0,1]];
  - depth is optical-axis z-depth unless the adapter declares range/disparity;
  - q(p) = normalized ray with q_z = 1, X(p) = z(p) * q(p).
"""
from __future__ import annotations

import numpy as np


class Camera:
    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 width: int, height: int, model: str = "PINHOLE",
                 params: list | None = None):
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.width, self.height = int(width), int(height)
        self.model = model
        self.params = list(params) if params is not None else None
        self.K = np.array([[self.fx, 0.0, self.cx],
                           [0.0, self.fy, self.cy],
                           [0.0, 0.0, 1.0]])
        self._pcam = None
        if model not in ("PINHOLE", None) and self.params is not None:
            import pycolmap
            self._pcam = pycolmap.Camera(model=model, width=self.width,
                                         height=self.height, params=self.params)

    # -- rays -------------------------------------------------------------
    def q(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        if self._pcam is not None:
            xy = np.asarray(self._pcam.cam_from_img(uv), dtype=float)
            return np.stack([xy[:, 0], xy[:, 1], np.ones(len(xy))], axis=-1)
        x = (uv[:, 0] - self.cx) / self.fx
        y = (uv[:, 1] - self.cy) / self.fy
        return np.stack([x, y, np.ones_like(x)], axis=-1)

    def backproject(self, uv: np.ndarray, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).reshape(-1)
        return self.q(uv) * z[:, None]

    def project(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float).reshape(-1, 3)
        if self._pcam is not None:
            return np.asarray(self._pcam.img_from_cam(X), dtype=float)
        u = self.fx * X[:, 0] / X[:, 2] + self.cx
        v = self.fy * X[:, 1] / X[:, 2] + self.cy
        return np.stack([u, v], axis=-1)

    # -- transforms -------------------------------------------------------
    def transformed(self, uv_to_uv: np.ndarray) -> "Camera":
        """Apply a 3x3 pixel-space projective transform; K' = A @ K."""
        A = np.asarray(uv_to_uv, dtype=float).reshape(3, 3)
        Kp = A @ self.K
        if abs(Kp[2, 2]) < 1e-12:
            raise ValueError("projective transform sends the plane at infinity to zero")
        return Camera(Kp[0, 0], Kp[1, 1], Kp[0, 2], Kp[1, 2],
                      self.width, self.height)


def translate_matrix(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def scale_matrix(sx: float, sy: float) -> np.ndarray:
    """Pixel-center-preserving resize by factor (sx, sy) applied to coordinates."""
    return np.diag([sx, sy, 1.0])


def crop_transform(x0: int, y0: int) -> np.ndarray:
    """bbox_xyxy=[x0,y0,x1,y1] half-open. Crop pixel (i,j) center == original (x0+i, y0+j)."""
    return translate_matrix(-float(x0), -float(y0))


def apply_h(M: np.ndarray, uv: np.ndarray) -> np.ndarray:
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    h = np.concatenate([uv, np.ones((len(uv), 1))], axis=1)
    out = (np.asarray(M, dtype=float).reshape(3, 3) @ h.T).T
    return out[:, :2] / out[:, 2:3]


def sample_depth_bilinear(depth: np.ndarray, uv: np.ndarray,
                          valid_mask: np.ndarray | None = None,
                          eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear sample at float pixel coords. Returns (values, valid)."""
    d = np.asarray(depth, dtype=np.float64)
    H, W = d.shape
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    x, y = uv[:, 0], uv[:, 1]
    inside = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
    x0 = np.clip(np.floor(x).astype(int), 0, W - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.clip(np.floor(y).astype(int), 0, H - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    wx = np.clip(x - x0, 0.0, 1.0)
    wy = np.clip(y - y0, 0.0, 1.0)
    v = (d[y0, x0] * (1 - wx) * (1 - wy) + d[y0, x1] * wx * (1 - wy)
         + d[y1, x0] * (1 - wx) * wy + d[y1, x1] * wx * wy)
    ok = inside & np.isfinite(v) & (v > eps)
    if valid_mask is not None:
        m = np.asarray(valid_mask).astype(bool)
        ok &= m[y0, x0] & m[y0, x1] & m[y1, x0] & m[y1, x1]
    return v, ok
