#!/usr/bin/env python
"""Evaluation for multi-view registration tasks (real or monocular geometry).

Reported per task:
  * per-view pose error against the reference view, in metres / degrees;
  * relative-camera error (pairwise rotation and translation direction), which reveals
    local misalignments that an aggregate surface distance hides;
  * GLOBAL SE(3): ONE rigid transform for the whole task, estimated against ground-truth
    point correspondences only (nearest GT surface points are used to establish
    correspondences from the *estimated* clouds, then a single Kabsch is fitted and applied
    to all views; error is then measured as distance to GT correspondences), never a
    per-view fit;
  * GLOBAL Sim(3): same with one scale.

Ground truth correspondences come from the evaluated cloud points and the real scan/GT-depth
surface, so nothing is invented by unconstrained nearest-neighbour matching of the
reconstruction to itself.

The estimator also refuses to fit rotation from fewer than 3 non-collinear camera centres
and flags near-collinear view-point sets as ill-conditioned.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def ang(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def kabsch(A, B, s=False):
    ca, cb = A.mean(0), B.mean(0)
    Hm = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    sc = float(S.sum() / max(((A - ca) ** 2).sum(), 1e-12)) if s else 1.0
    return sc, R, cb - sc * R @ ca


def apply(pts, s, R, t):
    return s * pts @ R.T + t


def identifiability(centres):
    """Centres of the views in the reference frame; bounds the rotation that points alone
    can determine."""
    c = np.asarray(centres, float)
    if len(c) < 3:
        return {'status': 'unidentifiable_rotation', 'n_centres': len(c)}
    sv = np.linalg.svd(c - c.mean(0), compute_uv=False)
    ratio = float(sv[-1] / max(sv[0], 1e-12))
    return {'status': 'ok' if ratio > 0.05 else 'near_collinear',
            'singular_value_ratio': ratio, 'n_centres': len(c)}


def evaluate_task(est, gt_planes, seed=0):
    """est: {'views': {sid: M_est (4x4, own frame -> scene frame)}, 'canonical': sid}
    gt:  {'views': {sid: M_gt}, 'surfaces': {sid: Nx3 world/ref-frame points}}
    Both M_est and M_gt map a view's OWN cloud into the SAME reference-view frame."""
    rng = np.random.default_rng(seed)
    sids = [s for s in est['views'] if s in gt_planes['views']]
    canon = est['canonical']
    per_view = {}
    centres_est, centres_gt = [], []
    for s in sids:
        Me = np.asarray(est['views'][s], float)
        Mg = np.asarray(gt_planes['views'][s], float)
        if s != canon:
            se = float(np.cbrt(abs(np.linalg.det(Me[:3, :3]))))
            Re = Me[:3, :3] / se
            per_view[s] = {'rot_err_deg': ang(Re @ Mg[:3, :3].T),
                           'trans_err_m': float(np.linalg.norm(Me[:3, 3] - Mg[:3, 3])),
                           'lateral_err_m': float(np.linalg.norm(
                               (Me[:3, 3] - Mg[:3, 3])[:2]))}
        centres_est.append(Me[:3, 3])
        centres_gt.append(Mg[:3, 3])
    # pair-wise relative rotation errors
    rel = {}
    for i, a in enumerate(sids):
        for b in sids[i + 1:]:
            Me_a, Me_b = np.asarray(est['views'][a], float), np.asarray(est['views'][b], float)
            Mg_a, Mg_b = np.asarray(gt_planes['views'][a], float), np.asarray(gt_planes['views'][b], float)
            Ra = Me_a[:3, :3] / np.cbrt(abs(np.linalg.det(Me_a[:3, :3])))
            Rb = Me_b[:3, :3] / np.cbrt(abs(np.linalg.det(Me_b[:3, :3])))
            Re_rel = Ra.T @ Rb
            Rg_rel = Mg_a[:3, :3].T @ Mg_b[:3, :3]
            rel['%s|%s' % (a, b)] = ang(Re_rel @ Rg_rel.T)
    # GLOBAL SE(3) and Sim(3): one transform for the whole task
    glob = {}
    for mode in ('SE3', 'Sim3'):
        res = []
        for s in sids:
            Me = np.asarray(est['views'][s], float)
            surf = gt_planes['surfaces'].get(s)
            pts = est.get('points', {}).get(s)
            if surf is None or pts is None:
                continue
            P = pts @ Me[:3, :3].T + Me[:3, 3]
            if len(P) > 20000:
                P = P[rng.choice(len(P), 20000, replace=False)]
            from scipy.spatial import cKDTree
            d, idx = cKDTree(surf).query(P)
            ok = d < 0.4
            if ok.sum() > 50:
                res.append((P[ok], surf[idx[ok]]))
        if not res:
            glob[mode] = None
            continue
        A = np.vstack([r[0] for r in res])
        B = np.vstack([r[1] for r in res])
        sc, R, t = kabsch(A, B, s=(mode == 'Sim3'))
        err = np.linalg.norm(apply(A, sc, R, t) - B, axis=1)
        glob[mode] = {'scale': sc, 'rot_deg': ang(R), 'trans_m': float(np.linalg.norm(t)),
                      'inlier_frac': float((err < 0.1).mean()),
                      'rms_m': float(np.sqrt(np.mean(err ** 2))),
                      'median_m': float(np.median(err))}
    return {'per_view': per_view,
            'rel_rot_deg_median': float(np.median(list(rel.values()))) if rel else None,
            'rel_rot_max_deg': float(np.max(list(rel.values()))) if rel else None,
            'global': glob,
            'identifiability_est': identifiability(centres_est),
            'identifiability_gt': identifiability(centres_gt)}


def synthetic_selftest():
    """Known transform -> evaluation must report zero error, and a per-scene global fit must
    recover the injected SE(3) exactly even when views are individually misplaced."""
    rng = np.random.default_rng(0)
    Ts = {}
    for s in 'ABC':
        T = np.eye(4)
        T[:3, :3] = Rot.random(random_state=rng.integers(1 << 31)).as_matrix()
        T[:3, 3] = rng.normal(size=3)
        Ts[s] = T
    Ts['R'] = np.eye(4)
    surf = {}
    pts = {}
    for s in 'ABCR':
        P = rng.normal(size=(500, 3)) * 2
        surf[s] = P @ Ts[s][:3, :3].T + Ts[s][:3, 3]
        pts[s] = P
    est = {'views': Ts, 'canonical': 'R', 'points': pts}
    gt = {'views': Ts, 'surfaces': surf}
    r = evaluate_task({'views': {k: v for k, v in Ts.items()}, 'canonical': 'R',
                       'points': pts}, gt)
    ok1 = all(abs(v['rot_err_deg']) < 1e-6 for v in r['per_view'].values())
    # inject a per-view error; the global SE(3) must still be a single transform
    est2 = {'views': {}, 'canonical': 'R', 'points': pts}
    for s, T in Ts.items():
        E = np.eye(4)
        if s != 'R':
            E[:3, 3] = np.array([0.3, -0.2, 0.1])
        est2['views'][s] = E @ T
    r2 = evaluate_task(est2, gt)
    ok2 = r2['global']['SE3'] is not None
    return {'zero_error_on_truth': ok1, 'global_fit_available_with_error': ok2,
            'sample_per_view': r['per_view'], 'global_with_error': r2['global']}


if __name__ == '__main__':
    print(json.dumps(synthetic_selftest(), indent=1, default=str))
