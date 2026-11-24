"""Coverage-aware lateral refinement for planar-degenerate overlaps.

In a plane-dominated overlap the ICP mean-distance cost is almost flat along the plane,
so lateral translation is unobservable. The OVERLAP FOOTPRINT is not flat: sliding the
moving cloud out of the reference footprint reduces the number of points that land near
the reference at all.

This searches the two tangential directions of the overlap plane, maximising the overlap
fraction (points within `tol` of the reference), coarse-to-fine. Rotation is held fixed at
the value the rigid baseline recovered, so this isolates the lateral DOF.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval'))
sys.path.insert(0, str(ROOT / 'workbench'))
from protocol import setup  # noqa: E402
from wb import Workbench  # noqa: E402

TASK = json.loads((ROOT / 'eval_data/task.json').read_text())
BASE = json.loads((ROOT / 'out' / 'baseline_rescored.json').read_text())
DEG = {r['sid'] + r['scene'] + r['tag']: r for r in
       json.loads((ROOT / 'out' / 'degeneracy.json').read_text())}


def plane_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dominant plane normal and the two in-plane axes."""
    X = points - points.mean(0)
    w, V = np.linalg.eigh((X.T @ X) / len(X))
    return V[:, 0], V[:, 1], V[:, 2]   # normal, then two in-plane axes


def coverage(ref_tree, pts, tol):
    d, _ = ref_tree.query(pts, k=1)
    return float((d < tol).mean())


def refine_lateral(ref_pts, mov_pts, R, t0, normal, axes, tol=0.35,
                   span=4.0, coarse=0.5, iters=3):
    """Coarse-to-fine search over offsets in the plane spanned by `axes`."""
    sub = mov_pts[::max(1, len(mov_pts) // 2500)]
    tree = cKDTree(ref_pts)
    e1, e2 = axes
    best = (coverage(tree, sub @ R.T + t0, tol), 0.0, 0.0)
    step = coarse
    centre = (0.0, 0.0)
    for _ in range(iters):
        n = int(span / step)
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                a = centre[0] + i * step
                b = centre[1] + j * step
                if abs(a) > span or abs(b) > span:
                    continue
                t = t0 + a * e1 + b * e2
                c = coverage(tree, sub @ R.T + t, tol)
                if c > best[0]:
                    best = (c, a, b)
        centre = (best[1], best[2])
        step /= 4.0
        span = step * 3
    cov, a, b = best
    return t0 + a * e1 + b * e2, cov, a, b


def main():
    low = [x for x in BASE
           if x['method'] == 'A_icp'
           and DEG.get(x['sid'] + x['scene'] + x['tag'], {}).get('normal_div', 1.0) < 0.09]
    print('planar-degenerate targets: %d' % len(low))
    rows = []
    for x in low:
        scene, tag, sid = x['scene'], x['tag'], x['sid']
        wb = Workbench(ROOT / 'agent_data' / scene, stride=32)
        setup(wb, TASK[scene][tag], TASK[scene][tag]['ref'])
        M = np.asarray(x['matrix_est'], float)
        c = wb.clouds[sid]
        s = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
        c.R, c.t, c.s = M[:3, :3] / s, M[:3, 3], s
        mov = c.transformed()
        ref = wb.clouds[wb.ref].transformed()
        normal, e1, e2 = plane_axes(mov)
        T = np.asarray(x['matrix_gt'], float)
        t_new, cov, a, b = refine_lateral(ref, mov, np.eye(3), M[:3, 3], normal, (e1, e2))
        lat_old = float(np.linalg.norm((M[:3, 3] - T[:3, 3])[:2]))
        lat_new = float(np.linalg.norm((t_new - T[:3, 3])[:2]))
        rows.append({'scene': scene, 'tag': tag, 'sid': sid,
                     'lateral_before': lat_old, 'lateral_after': lat_new,
                     'offset_a': a, 'offset_b': b, 'coverage': cov,
                     'normal_div': DEG[sid + scene + tag]['normal_div']})
        print('%-14s%-4s%-12s lat %6.2f -> %6.2f  (offset %.2f,%.2f cov %.3f)'
              % (scene, tag, sid, lat_old, lat_new, a, b, cov), flush=True)
    p = ROOT / 'out' / 'coverage_refine.json'
    p.write_text(json.dumps(rows, indent=1))
    if rows:
        b4 = np.array([r['lateral_before'] for r in rows])
        af = np.array([r['lateral_after'] for r in rows])
        print('\nmedian lateral %.2f -> %.2f m   improved %d/%d'
              % (np.median(b4), np.median(af), int((af < b4).sum()), len(rows)))
    print('wrote', p)


if __name__ == '__main__':
    main()
