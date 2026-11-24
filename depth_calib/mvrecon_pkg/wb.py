"""Interactive 3D workbench: a controlled state interface over the point clouds.

Design rules from the spec:
  - the first cloud is the REFERENCE and never moves
  - each other cloud has a cumulative transform (R, t, s) applied to its own-frame points
  - all state changes go through this object, which keeps an audit log
  - observations (views, close-ups, region picks) are rendered from the CURRENT state
  - picks return real points and their source pixels, never invented ones

Stage 1 constraint: each cloud keeps its internal shape; only rigid + uniform scale.
"""
from __future__ import annotations

import copy
import json
import pathlib

import numpy as np
import open3d as o3d


class Cloud:
    """One view's points in its own camera frame, plus the transform applied to them."""

    def __init__(self, sid, points, colors, K, W, H, depth):
        self.sid = sid
        self.points = np.asarray(points, dtype=np.float64)   # own frame, never modified
        self.colors = np.asarray(colors, dtype=np.float64)
        self.K = np.asarray(K, dtype=np.float64)
        self.W, self.H = W, H
        self.depth = np.asarray(depth, dtype=np.float64)
        self.R = np.eye(3)
        self.t = np.zeros(3)
        self.s = 1.0

    def matrix(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.R * self.s
        M[:3, 3] = self.t
        return M

    def transformed(self) -> np.ndarray:
        return (self.points * self.s) @ self.R.T + self.t

    def as_o3d(self, color=None) -> o3d.geometry.PointCloud:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.transformed())
        c = self.colors if color is None else np.tile(np.asarray(color, float), (len(self.points), 1))
        pc.colors = o3d.utility.Vector3dVector(c)
        return pc


class Workbench:
    def __init__(self, scene_dir: pathlib.Path, stride: int = 24):
        self.scene_dir = pathlib.Path(scene_dir)
        self.meta = json.loads((self.scene_dir / 'meta.json').read_text())
        self.clouds: dict[str, Cloud] = {}
        self.ref: str | None = None
        self.log: list[dict] = []
        self._history: list[dict] = []
        self._load(stride)

    def _load(self, stride):
        for im in self.meta['images']:
            sid = im['sid']
            dep = np.load(self.scene_dir / im['depth_file']).astype(np.float64)
            K = np.asarray(im['K'], float)
            H, W = dep.shape
            ys, xs = np.mgrid[0:H:stride, 0:W:stride]
            z = dep[ys, xs]
            ok = np.isfinite(z) & (z > 0) & (z < 200)
            ys, xs, z = ys[ok], xs[ok], z[ok]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            P = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], 1)
            # per-point colour from the source image so views are distinguishable
            colors = np.tile(_sid_color(sid), (len(P), 1))
            self.clouds[sid] = Cloud(sid, P, colors, K, W, H, dep)
        self.ref = sorted(self.clouds)[0]
        self.log.append({'op': 'load', 'views': sorted(self.clouds), 'ref': self.ref})

    # ---------- state ----------
    def set_reference(self, sid):
        self.ref = sid
        self._snap()
        self.log.append({'op': 'set_reference', 'ref': sid})

    def _snap(self):
        self._history.append({sid: (c.R.copy(), c.t.copy(), c.s)
                              for sid, c in self.clouds.items()})

    def undo(self):
        if not self._history:
            return False
        state = self._history.pop()
        for sid, (R, t, s) in state.items():
            self.clouds[sid].R = R
            self.clouds[sid].t = t
            self.clouds[sid].s = s
        self.log.append({'op': 'undo'})
        return True

    def apply(self, sid, R=None, t=None, s=None):
        """Apply an incremental transform to one cloud (not the reference)."""
        if sid == self.ref:
            raise ValueError('the reference cloud does not move')
        c = self.clouds[sid]
        self._snap()
        if R is not None:
            c.R = np.asarray(R, float) @ c.R
        if t is not None:
            # translation is in the world frame, so it composes additively
            c.t = c.t + np.asarray(t, float)
        if s is not None:
            c.s = float(s) * c.s
            c.t = c.t * float(s)
        self.log.append({'op': 'apply', 'sid': sid, 's': c.s,
                         'R_trace': float(np.trace(c.R)), 't': c.t.tolist()})
        return c.matrix()

    def state(self):
        return {sid: {'R': c.R.tolist(), 't': c.t.tolist(), 's': c.s}
                for sid, c in self.clouds.items()}

    # ---------- observation ----------
    def combined(self, exclude=()):
        pcs, pts = [], []
        for sid, c in self.clouds.items():
            if sid in exclude:
                continue
            pcs.append(c.as_o3d())
            pts.append(c.transformed())
        return pcs, pts

    def overlap_stats(self, sid, radius=0.35):
        """Fraction of `sid`'s transformed points within `radius` of any other cloud."""
        if sid == self.ref:
            raise ValueError('reference has no transform to score')
        others = [c.transformed() for s, c in self.clouds.items() if s != sid]
        if not others:
            return 0.0, 0.0
        A = np.concatenate(others, 0) if len(others) > 1 else others[0]
        if len(A) < 20:
            return 0.0, 0.0
        from scipy.spatial import cKDTree
        d, _ = cKDTree(A).query(self.clouds[sid].transformed(), k=1)
        return float((d < radius).mean()), float(np.median(d))

    def residual(self, sid, radius=0.35):
        """Mean nearest-neighbour distance from `sid` onto the others (inlier-trimmed)."""
        others = [c.transformed() for s, c in self.clouds.items() if s != sid]
        if not others:
            return float('nan')
        A = np.concatenate(others, 0)
        if len(A) < 20:
            return float('nan')
        from scipy.spatial import cKDTree
        d, _ = cKDTree(A).query(self.clouds[sid].transformed(), k=1)
        keep = d < radius
        return float(np.mean(d[keep])) if keep.sum() else float('nan')

    def pick_region(self, sid, uv_box):
        """Return the REAL points whose source pixels fall in a uv box."""
        c = self.clouds[sid]
        H, W = c.depth.shape
        u0, v0, u1, v1 = uv_box
        ys, xs = np.mgrid[0:H, 0:W]
        m = (xs >= u0) & (xs <= u1) & (ys >= v0) & (ys <= v1)
        z = c.depth[m]
        xy = np.stack([xs[m], ys[m]], 1)
        ok = np.isfinite(z) & (z > 0) & (z < 200)
        z, xy = z[ok], xy[ok]
        fx, fy, cx, cy = c.K[0, 0], c.K[1, 1], c.K[0, 2], c.K[1, 2]
        P = np.stack([(xy[:, 0] - cx) / fx * z, (xy[:, 1] - cy) / fy * z, z], 1)
        return {'sid': sid, 'n': int(len(P)),
                'points_own_frame': P[::max(1, len(P)//200)].tolist(),
                'pixels': xy[::max(1, len(P)//200)].tolist()}


def _sid_color(sid):
    h = abs(hash(sid)) % 360
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, 0.65, 0.95)
    return [r, g, b]
