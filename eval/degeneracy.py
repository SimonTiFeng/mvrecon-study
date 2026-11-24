"""Is the lateral translation failure explained by surface degeneracy?

ICP constrains translation only through the spread of surface normals in the overlapping
region. If the overlap is dominated by a single plane, sliding within that plane leaves the
nearest-neighbour cost nearly unchanged, so translation is unobservable no matter how well
rotation is recovered.

For each target we compute, over the region of the moving cloud that actually overlaps the
reference in the ESTIMATED pose:
  - normal-diversity  = smallest eigenvalue of the (normalised) normal covariance
                        -> ~0 means all normals parallel (planar), -> 0.33 isotropic
  - planarity         = (l1 - l2) / l1 of the point covariance
and correlate both with the lateral translation error.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval'))
sys.path.insert(0, str(ROOT / 'workbench'))
from protocol import setup  # noqa: E402
from wb import Workbench  # noqa: E402

TASK = json.loads((ROOT / 'eval_data/task.json').read_text())
BASE = json.loads((ROOT / 'out' / 'baseline_rescored.json').read_text())


def observability(points: np.ndarray) -> dict:
    """Normal diversity and planarity of a point set."""
    if len(points) < 30:
        return {'normal_div': float('nan'), 'planarity': float('nan'), 'n': len(points)}
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.6, max_nn=30))
    n = np.asarray(pc.normals)
    # sign-invariant normal covariance: use outer products, which is sign independent
    C = (n.T @ n) / len(n)
    ev = np.sort(np.linalg.eigvalsh(C))
    normal_div = float(ev[0] / max(ev.sum(), 1e-12))
    X = points - points.mean(0)
    pv = np.sort(np.linalg.eigvalsh((X.T @ X) / len(X)))
    planarity = float((pv[2] - pv[1]) / max(pv[2], 1e-12))
    return {'normal_div': normal_div, 'planarity': planarity, 'n': int(len(points))}


def main():
    rows = []
    for x in BASE:
        if x['method'] != 'A_icp':
            continue
        s = x['score_corrected']
        if s['rot_err_deg'] >= 20:
            continue
        scene, tag, sid = x['scene'], x['tag'], x['sid']
        wb = Workbench(ROOT / 'agent_data' / scene, stride=32)
        setup(wb, TASK[scene][tag], TASK[scene][tag]['ref'])
        # reproduce the method's final cumulative transform from the stored matrix
        M = np.asarray(x['matrix_est'], float)
        c = wb.clouds[sid]
        sc = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
        c.R, c.t, c.s = M[:3, :3] / sc, M[:3, 3], sc
        mine = c.transformed()
        ref = wb.clouds[wb.ref].transformed()
        d, _ = cKDTree(ref).query(mine, k=1)
        overlap = mine[d < 0.35]
        obs = observability(overlap)
        T = np.asarray(x['matrix_gt'], float)
        err = M[:3, 3] - T[:3, 3]
        obs.update({'scene': scene, 'tag': tag, 'sid': sid,
                    'rot_err': s['rot_err_deg'],
                    'axial': abs(float(err[2])),
                    'lateral': float(np.linalg.norm(err[:2])),
                    'n_overlap': len(overlap)})
        rows.append(obs)
        print('%-14s%-4s%-12s rot %5.1f lat %6.2f  normal_div %.4f planarity %.3f n=%d'
              % (scene, tag, sid, obs['rot_err'], obs['lateral'],
                 obs['normal_div'], obs['planarity'], obs['n_overlap']), flush=True)

    p = ROOT / 'out' / 'degeneracy.json'
    p.write_text(json.dumps(rows, indent=1))
    lat = np.array([r['lateral'] for r in rows])
    nd = np.array([r['normal_div'] for r in rows])
    pl = np.array([r['planarity'] for r in rows])
    ok = np.isfinite(nd) & np.isfinite(pl)
    print('\nn=%d' % ok.sum())
    print('corr(log lateral, log normal_div) = %+.3f'
          % np.corrcoef(np.log1p(lat[ok]), np.log1p(nd[ok]))[0, 1])
    print('corr(log lateral, planarity)      = %+.3f'
          % np.corrcoef(np.log1p(lat[ok]), pl[ok])[0, 1])
    if ok.sum() > 4:
        med = np.median(nd[ok])
        hi = nd[ok] >= med
        print('low  normal diversity (planar): median lateral %.3f m (n=%d)'
              % (np.median(lat[ok][~hi]), int((~hi).sum())))
        print('high normal diversity        : median lateral %.3f m (n=%d)'
              % (np.median(lat[ok][hi]), int(hi.sum())))
    print('\nwrote', p)


if __name__ == '__main__':
    main()
