"""Agent-facing tool surface over the Workbench.

Every capability the agent has is a method here. The agent NEVER sees eval_data.
Every observation is rendered from the current state, and every pick returns real
points traceable to source pixels.

Budget and stopping rules are enforced here, not by the agent's goodwill.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import open3d as o3d

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from wb import Workbench


class AgentTools:
    def __init__(self, scene_dir, stride=24, budget=None):
        self.wb = Workbench(scene_dir, stride=stride)
        b = budget or {}
        self.budget = {
            'render': b.get('render', 24),
            'transform': b.get('transform', 20),
            'solve': b.get('solve', 8),
            'pick': b.get('pick', 10),
            'total': b.get('total', 60),
        }
        self.used = {k: 0 for k in self.budget}
        self.trace: list[dict] = []

    # ---------- budget ----------
    def _charge(self, kind, n=1):
        self.used[kind] += n
        self.used['total'] += n
        return self.used[kind] <= self.budget[kind] and self.used['total'] <= self.budget['total']

    def exhausted(self):
        return any(self.used[k] >= self.budget[k] for k in self.budget)

    def report(self):
        return {'used': dict(self.used), 'budget': dict(self.budget),
                'exhausted': self.exhausted()}

    # ---------- observation ----------
    def list_views(self):
        return {'views': sorted(self.wb.clouds), 'reference': self.wb.ref,
                'current_state': self.wb.state()}

    def render(self, view='front', exclude=(), size=(320, 240)):
        """Return an RGB image of the current state from a named viewpoint."""
        if not self._charge('render'):
            return {'error': 'render budget exhausted'}
        pcs = [c.as_o3d() for s, c in self.wb.clouds.items() if s not in exclude]
        pts = np.concatenate([np.asarray(p.points) for p in pcs], 0)
        cen = pts.mean(0)
        rad = float(np.linalg.norm(pts - cen, axis=1).max())
        if view == 'top':
            eye, up = cen + [0, 0, 3 * rad], [0, 1, 0]
        elif view == 'side':
            eye, up = cen + [3 * rad, 0, 0], [0, 0, 1]
        elif view == 'front':
            eye, up = cen + [0, -3 * rad, 0.3 * rad], [0, 0, 1]
        else:
            eye, up = cen + np.asarray(view[:3], float) * rad, np.asarray(view[3:6], float)
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=size[0], height=size[1])
        for p in pcs:
            vis.add_geometry(p)
        vc = vis.get_view_control()
        vc.set_front(eye - cen)
        vc.set_up(up)
        vc.set_zoom(0.6)
        vis.poll_events()
        vis.update_renderer()
        img = np.asarray(vis.capture_screen_float_buffer(False))
        vis.destroy_window()
        self.trace.append({'op': 'render', 'view': view, 'exclude': list(exclude)})
        return {'image': img, 'n_points': int(len(pts))}

    def closeup(self, sid, uv_box, size=(256, 256)):
        """Crop of the ORIGINAL photo plus that region's current 3D stats."""
        if not self._charge('pick'):
            return {'error': 'pick budget exhausted'}
        d = self.wb.clouds[sid]
        u0, v0, u1, v1 = uv_box
        p = self.wb.pick_region(sid, uv_box)
        px = np.array(p['points_own_frame'], float)
        if len(px):
            M = d.matrix()
            world = (px @ M[:3, :3].T) + M[:3, 3]
            stats = {'centroid_world': world.mean(0).tolist(),
                     'extent_world': (world.max(0) - world.min(0)).tolist()}
        else:
            stats = {}
        self.trace.append({'op': 'closeup', 'sid': sid, 'uv': list(uv_box)})
        return {'sid': sid, 'n': p['n'], **stats}

    def measure(self, sid, uv_box):
        """Physical size in metres of a picked region, in the CURRENT state."""
        if not self._charge('pick'):
            return {'error': 'pick budget exhausted'}
        p = self.wb.pick_region(sid, uv_box)
        px = np.array(p['points_own_frame'], float)
        if len(px) < 10:
            return {'error': 'too few points', 'n': len(px)}
        ext = (px.max(0) - px.min(0)) * self.wb.clouds[sid].s
        return {'sid': sid, 'extent_m': ext.tolist(), 'diag_m': float(np.linalg.norm(ext)),
                'n': p['n'], 'scale': self.wb.clouds[sid].s}

    # ---------- action ----------
    def transform(self, sid, R=None, t=None, s=None):
        if not self._charge('transform'):
            return {'error': 'transform budget exhausted'}
        try:
            M = self.wb.apply(sid, R=R, t=t, s=s)
        except ValueError as e:
            return {'error': str(e)}
        ov = None
        if sid != self.wb.ref:
            ov = self.wb.overlap_stats(sid)[0]
            rs = self.wb.residual(sid)
        self.trace.append({'op': 'transform', 'sid': sid, 's': s,
                           'overlap': ov})
        return {'ok': True, 'matrix': M.tolist(), 'overlap': ov,
                'residual': rs if sid != self.wb.ref else None}

    def solve_correspondences(self, sid, src_box, dst_box, dst_sid=None):
        """Fit a transform from two picked regions (coarse correspondences)."""
        if not self._charge('solve'):
            return {'error': 'solve budget exhausted'}
        dst_sid = dst_sid or self.wb.ref
        ps = np.array(self.wb.pick_region(sid, src_box)['points_own_frame'], float)
        pd = np.array(self.wb.pick_region(dst_sid, dst_box)['points_own_frame'], float)
        if len(ps) < 20 or len(pd) < 20:
            return {'error': 'too few points in one region'}
        R, t = _kabsch(ps, pd)
        self.wb._snap()
        M = _compose(self.wb.clouds[sid].matrix(), R, t)
        self.wb.apply(sid, R=R, t=t)
        self.trace.append({'op': 'solve', 'sid': sid, 'n_src': len(ps), 'n_dst': len(pd)})
        return {'ok': True, 'R': R.tolist(), 't': t.tolist(),
                'overlap': self.wb.overlap_stats(sid)[0], 'residual': self.wb.residual(sid)}

    def refine_icp(self, sid, radius=0.6, max_iter=80):
        if not self._charge('solve'):
            return {'error': 'solve budget exhausted'}
        src = self.wb.clouds[sid].as_o3d()
        dst = self.wb.clouds[self.wb.ref].as_o3d()
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=20))
        dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=20))
        r = o3d.pipelines.registration.registration_icp(
            src, dst, radius, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
        self.wb._snap()
        T = r.transformation
        newM = T @ self.wb.clouds[sid].matrix()
        Sc = float(np.cbrt(abs(np.linalg.det(newM[:3, :3]))))
        c = self.wb.clouds[sid]
        c.R, c.t, c.s = newM[:3, :3] / Sc, newM[:3, 3], float(c.s * Sc)
        self.trace.append({'op': 'icp', 'sid': sid, 'fitness': float(r.fitness)})
        return {'ok': True, 'fitness': float(r.fitness), 'rmse': float(r.inlier_rmse),
                'overlap': self.wb.overlap_stats(sid)[0]}

    def stats(self, sid):
        if sid == self.wb.ref:
            return {'sid': sid, 'note': 'reference', 'n': len(self.wb.clouds[sid].points)}
        ov, med = self.wb.overlap_stats(sid)
        return {'sid': sid, 'overlap': ov, 'median_nn_m': med,
                'residual': self.wb.residual(sid), 'scale': self.wb.clouds[sid].s}

    def undo(self):
        ok = self.wb.undo()
        self.trace.append({'op': 'undo', 'ok': ok})
        return {'ok': ok}

    def give_up(self, reason):
        self.trace.append({'op': 'give_up', 'reason': reason})
        return {'status': 'abstain', 'reason': reason}


def _kabsch(src, dst):
    """Least-squares rigid transform mapping src onto dst (no scale)."""
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return R, cd - R @ cs


def _compose(M, R, t):
    out = np.eye(4)
    out[:3, :3] = R @ M[:3, :3]
    out[:3, 3] = R @ M[:3, 3] + t
    return out
