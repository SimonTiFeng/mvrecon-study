"""Can systematic multi-start search do the narrow-basin job?

Addendum 4 localised the remaining failures to four targets where the achievable rotation precision
is high (floor 0.65-1.92 deg) but the basin is narrow (2-4 deg) and the coarse estimate is 5-9 deg
out. The interactive loop's claimed job is to supply an initialisation inside the basin.

Before crediting that to interaction or semantics, test the cheap geometric alternative: a
systematic multi-start search over rotations near the coarse estimate. If ICP basins are ~5 deg
median, a grid at 8 deg in the three rotation-vector components should cover the space of
"plausible corrections" without any learning.

Selection is ground-truth free, using the signal validated earlier: the spread of per-correspondence
translations (+0.43 correlation with rotation error). The best candidate by that spread is taken.
Reported alongside the oracle-best candidate over the same set, so selection loss is visible.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import pathlib
import sys

import cv2
import numpy as np
import open3d as o3d

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval'))
sys.path.insert(0, str(ROOT / 'workbench'))
from protocol import setup, rotdeg  # noqa: E402
from wb import Workbench  # noqa: E402

_s = importlib.util.spec_from_file_location('fpi', ROOT / 'eval' / 'frame_plus_identity.py')
fpi = importlib.util.module_from_spec(_s)
_s.loader.exec_module(fpi)
_s2 = importlib.util.spec_from_file_location('pref', ROOT / 'eval' / 'plane_refine.py')
pref = importlib.util.module_from_spec(_s2)
_s2.loader.exec_module(pref)

TASK = json.loads((ROOT / 'eval_data/task.json').read_text())
BASE = json.loads((ROOT / 'out' / 'baseline_rescored.json').read_text())
RNG = np.random.default_rng(0)

TARGETS = [
    ('playground', 'v2', 'eth3d_0286'),
    ('playground', 'v3', 'eth3d_0286'),
    ('playground', 'v4', 'eth3d_0286'),
    ('facade', 'v3', 'eth3d_0209'),
    ('facade', 'v2', 'eth3d_0209'),
    ('office', 'v3', 'eth3d_0149'),
    ('courtyard', 'v3', 'eth3d_0000'),
    ('delivery_area', 'v3', 'eth3d_0057'),
]
STEP = 8.0
LIM = 16.0


def prep(pts, voxel=0.5, max_pts=2500):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc = pc.voxel_down_sample(voxel)
    if len(pc.points) > max_pts:
        pc = pc.select_by_index(RNG.choice(len(pc.points), max_pts, replace=False))
    return pc


def main():
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    grid = []
    vals = np.arange(-LIM, LIM + 1e-9, STEP)
    for wx in vals:
        for wy in vals:
            for wz in vals:
                if np.linalg.norm([wx, wy, wz]) <= LIM + 1e-9:
                    grid.append(np.array([wx, wy, wz]))
    print('rotation grid: %d starts' % len(grid))
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

        pair = fpi.sift_pairs(scene, wb.ref, sid, ref_c, mov_c, sift, bf)
        if pair is None:
            print('%-14s%-4s%-12s no correspondences' % (scene, tag, sid), flush=True)
            continue
        Pr, Pm = pair

        def score(R):
            ti = Pr - Pm @ R.T
            t = np.median(ti, axis=0)
            sp = float(np.median(np.linalg.norm(ti - t, axis=1)))
            return sp, float(rotdeg(R @ R_true.T)), float(np.linalg.norm((t - t_true)[:2]))

        src, dst = prep(mov_c.points), prep(ref_c.points)
        cands = []
        for w in grid:
            R0 = pref.rodrigues(np.deg2rad(w)) @ R_coarse
            _, _, t_seed = score(R0)
            T0 = np.eye(4)
            T0[:3, :3] = R0
            T0[:3, 3] = t_seed
            res = o3d.pipelines.registration.registration_icp(
                src, dst, 0.20, T0,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
            R1 = res.transformation[:3, :3]
            sp, rot, lat = score(R1)
            cands.append((sp, rot, lat, float(res.fitness)))
        sp0, rot0, lat0 = score(R_coarse)
        best = min(cands, key=lambda z: z[0])
        oracle = min(cands, key=lambda z: z[1])
        rows.append({'scene': scene, 'tag': tag, 'sid': sid,
                     'coarse': [rot0, lat0, sp0],
                     'selected': [best[1], best[2], best[0]],
                     'oracle': [oracle[1], oracle[2], oracle[0]],
                     'n_starts': len(grid)})
        r = rows[-1]
        print('%-14s%-4s%-12s coarse %5.2f°/%5.2f m | selected %5.2f°/%5.2f m | oracle %5.2f°/%5.2f m'
              % (scene, tag, sid, rot0, lat0, best[1], best[2], oracle[1], oracle[2]), flush=True)

    p = ROOT / 'out' / 'multistart_rotation.json'
    p.write_text(json.dumps(rows, indent=1))
    if rows:
        g = lambda k, i: np.array([r[k][i] for r in rows])
        print('\nn=%d targets' % len(rows))
        for name in ('coarse', 'selected', 'oracle'):
            print('%-9s rot median %5.2f°  lat median %5.2f m  under 0.5 m %d/%d'
                  % (name, np.median(g(name, 0)), np.median(g(name, 1)),
                     int((g(name, 1) < 0.5).sum()), len(rows)))
    print('wrote', p)


if __name__ == '__main__':
    main()
