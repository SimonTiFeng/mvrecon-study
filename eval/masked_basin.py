"""With sky masked, how wide is the rotation basin?

masked_rotation.py showed the mask improves playground's achievable rotation precision from
~2 deg to 0.49 deg, but ICP started at the coarse rotation still lands at ~4.6-5.1 deg. Either the
basin is still narrow, or ICP is failing for another reason.

Scan the offset from the true rotation (masked geometry, translation seeded from an anchor) and
record where local refinement converges. That gives the basin directly, and with it whether a
search accurate to that width would suffice.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import open3d as o3d

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval'))
sys.path.insert(0, str(ROOT / 'workbench'))
from protocol import setup, rotdeg  # noqa: E402
from wb import Workbench  # noqa: E402
import importlib.util
_s = importlib.util.spec_from_file_location('mr', ROOT / 'eval' / 'masked_rotation.py')
mr = importlib.util.module_from_spec(_s)
_s.loader.exec_module(mr)

TASK = json.loads((ROOT / 'eval_data/task.json').read_text())
BASE = json.loads((ROOT / 'out' / 'baseline_rescored.json').read_text())
OFFSETS = np.arange(-10, 10.01, 1.0)
TARGETS = [('playground', 'v2', 'eth3d_0286'), ('facade', 'v3', 'eth3d_0209')]


def axis_rot(ax, deg):
    th = np.deg2rad(deg)
    k = np.zeros(3)
    k[ax] = 1.0
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def main():
    rows = []
    for scene, tag, sid in TARGETS:
        x = next(r for r in BASE if r['method'] == 'A_icp' and r['scene'] == scene
                 and r['tag'] == tag and r['sid'] == sid)
        wb = Workbench(ROOT / 'agent_data' / scene, stride=32)
        setup(wb, TASK[scene][tag], TASK[scene][tag]['ref'])
        ref_c, mov_c = wb.clouds[wb.ref], wb.clouds[sid]
        M = np.asarray(x['matrix_est'], float)
        R_coarse = M[:3, :3] / max(np.cbrt(abs(np.linalg.det(M[:3, :3]))), 1e-12)
        T = np.asarray(x['matrix_gt'], float)
        R_true, t_true = T[:3, :3], T[:3, 3]
        pa, pm = mr.anchor_of(ref_c.points, R_true, t_true, mov_c.K, mov_c.depth)
        kr = mr.keep_mask(scene, ref_c.sid, ref_c, None, True)
        km = mr.keep_mask(scene, sid, mov_c, None, True)
        src, dst = mr.prep(mov_c.points[km]), mr.prep(ref_c.points[kr])

        def at(R0):
            t0 = pa - R0 @ pm
            T0 = np.eye(4)
            T0[:3, :3] = R0
            T0[:3, 3] = t0
            r = o3d.pipelines.registration.registration_icp(
                src, dst, 0.30, T0,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80))
            return rotdeg(r.transformation[:3, :3] @ R_true.T)

        floor = at(R_true)
        from_coarse = at(R_coarse)
        curve = {}
        for ax in range(3):
            curve['xyz'[ax]] = [(float(d), float(at(R_true @ axis_rot(ax, d)))) for d in OFFSETS]
        # basin: largest |offset| on each axis that still converges near the floor
        tol = max(0.6, 2.0 * floor)
        widths = []
        for ax in 'xyz':
            ok = [abs(d) for d, r in curve[ax] if r <= tol]
            widths.append(max(ok) if ok else 0.0)
        rows.append({'scene': scene, 'tag': tag, 'sid': sid, 'floor': floor,
                     'from_coarse': from_coarse, 'tol': tol,
                     'basin_widths': widths, 'curve': curve})
        print('%-14s%-4s%-12s floor %.2f  from-coarse %.2f  basin widths %s deg (tol %.2f)'
              % (scene, tag, sid, floor, from_coarse,
                 '/'.join('%.0f' % w for w in widths), tol), flush=True)
        for ax in 'xyz':
            print('    %s: %s' % (ax, ' '.join('%+.0f:%.1f' % (d, r) for d, r in curve[ax])))

    p = ROOT / 'out' / 'masked_basin.json'
    p.write_text(json.dumps(rows, indent=1))
    print('\nwrote', p)


if __name__ == '__main__':
    main()
