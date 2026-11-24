"""How precisely do two sparse monocular clouds constrain rotation?

ICP is not a valid instrument for this question: starting from the EXACT transform it drifts up to
78 deg on some targets (fitness only 0.45 at a 0.5 m threshold, and point-to-plane is unstable on
grazing surfaces). That measures ICP's instability, not the information in the data.

So measure the objective directly and never optimise it. Hold translation at its exact value and
scan a rotation offset about each axis, evaluating a fixed geometric cost. Two questions:

  identified?  is the true rotation the minimum of the cost, and by how much
  sharp?        how far can rotation move before the cost degrades measurably -- i.e. the precision
               the data supports, which is the best any method could achieve

Costs reported:
  overlap  - fraction of moving points landing within 0.20 m of the reference
  nn       - median nearest-neighbour distance
A rotation is well determined only if the cost rises clearly as it moves away from the truth.
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
OFFSETS = np.arange(-20, 20.01, 2.0)
RNG = np.random.default_rng(9)
TOL = 0.20


def axis_rot(axis, deg):
    a = np.zeros(3)
    a[axis] = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    cx, sx = c[0], s[0]
    cy, sy = c[1], s[1]
    cz, sz = c[2], s[2]
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def main():
    rows = []
    for x in BASE:
        if x['method'] != 'A_icp':
            continue
        if x['score_corrected']['rot_err_deg'] >= 20:
            continue
        scene, tag, sid = x['scene'], x['tag'], x['sid']
        wb = Workbench(ROOT / 'agent_data' / scene, stride=32)
        setup(wb, TASK[scene][tag], TASK[scene][tag]['ref'])
        ref_c, mov_c = wb.clouds[wb.ref], wb.clouds[sid]
        T = np.asarray(x['matrix_gt'], float)
        R_true, t_true = T[:3, :3], T[:3, 3]
        # own-frame points: as_o3d() returns transformed(), i.e. already at the injected D
        P_ref = ref_c.points
        P_mov = mov_c.points
        if len(P_mov) > 6000:
            P_mov = P_mov[RNG.choice(len(P_mov), 6000, replace=False)]
        tree = cKDTree(P_ref)

        def cost(R):
            Q = P_mov @ R.T + t_true
            d, _ = tree.query(Q, k=1)
            return float((d < TOL).mean()), float(np.median(d))

        c0 = cost(R_true)
        curve = {}
        for ax in range(3):
            vals = []
            for d in OFFSETS:
                ov, nn = cost(R_true @ axis_rot(ax, d))
                vals.append((float(d), ov, nn))
            curve['xyz'[ax]] = vals

        row = {'scene': scene, 'tag': tag, 'sid': sid,
               'overlap_true': c0[0], 'nn_true': c0[1], 'curve': curve}
        # is the truth the best offset on each axis?
        for ax in 'xyz':
            v = curve[ax]
            best_ov = max(p[1] for p in v)
            row['argmin_%s' % ax] = float(min(v, key=lambda p: p[2])[0])
            row['best_ov_%s' % ax] = best_ov
        rows.append(row)
        print('%-14s%-4s%-12s overlap@true %.2f nn@true %.3f | argmin x%+5.1f y%+5.1f z%+5.1f'
              % (scene, tag, sid, c0[0], c0[1], row['argmin_x'], row['argmin_y'],
                 row['argmin_z']), flush=True)

    p = ROOT / 'out' / 'rotation_profile.json'
    p.write_text(json.dumps(rows, indent=1))
    if rows:
        print('\nn=%d' % len(rows))
        print('overlap at true rotation: median %.2f' % np.median([r['overlap_true'] for r in rows]))
        for ax in 'xyz':
            a = np.array([r['argmin_%s' % ax] for r in rows])
            ok = int((np.abs(a) <= 2.0).sum())
            print('axis %s: truth is the argmin (within 2 deg) in %d/%d, median |offset| %.1f deg'
                  % (ax, ok, len(rows), np.median(np.abs(a))))
        # how much does 2 deg of rotation error cost, on average
        d2 = []
        for r in rows:
            for ax in 'xyz':
                v = {p[0]: p[1] for p in r['curve'][ax]}
                d2.append(v.get(2.0, np.nan) - r['overlap_true'])
        d2 = np.array([v for v in d2 if np.isfinite(v)])
        print('a 2 deg rotation error changes overlap by median %+.3f' % np.median(d2))
    print('wrote', p)


if __name__ == '__main__':
    main()
