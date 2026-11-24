"""Multi-view fusion: use every pair in the subset, not just each view against the reference.

The nested v3/v4 subsets contain 3-4 images but every result so far registers each view only to the
reference. For a subset with k views there are k(k-1)/2 pairs, and each gives an independent estimate
of the relative pose. Composing chains through other views and combining them robustly should reduce
error roughly like 1/sqrt(k) and can rescue a pair whose direct estimate fails.

Pipeline per subset:
  1. classical front end (FPFH + RANSAC + ICP) for EVERY pair, to initialise
  2. correspondence refinement for every pair: sky-masked SIFT matches, robust reprojection fit
  3. for each view, collect candidate poses to the reference:
       direct, and via each other view (compose two estimated transforms)
  4. combine robustly: component-wise median of translation, chordal mean of rotation
  5. score the direct estimate and the fused estimate against the reference transform

Conventions: T_ab maps a point in b's frame into a's frame, i.e. p_a = R_ab p_b + t_ab.
Composition through k: T_ak o T_kb.
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
from scipy.optimize import least_squares

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval'))
sys.path.insert(0, str(ROOT / 'workbench'))
from protocol import setup, rotdeg  # noqa: E402
from wb import Workbench  # noqa: E402

_s = importlib.util.spec_from_file_location('si', ROOT / 'eval' / 'sift_identity.py')
si = importlib.util.module_from_spec(_s)
_s.loader.exec_module(si)

TASK = json.loads((ROOT / 'eval_data' / 'task.json').read_text())
RNG = np.random.default_rng(3)


def rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def feat(wb, sid):
    pc = wb.clouds[sid].as_o3d()
    d = pc.voxel_down_sample(0.8)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    f = o3d.pipelines.registration.compute_fpfh_feature(
        d, o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=60))
    return pc, d, f


def front_end(src_pc, dst_pc, fs, Ft, d_src, d_dst):
    coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        d_src, d_dst, fs, Ft, True, 1.0,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [],
        o3d.pipelines.registration.RANSACConvergenceCriteria(12000, 0.95))
    d_src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    d_dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    fine = o3d.pipelines.registration.registration_icp(
        d_src, d_dst, 0.5, coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80))
    return fine.transformation


def pair_pose(scene, wb, a, b, sift, bf, cache):
    """Estimate T_ab (b's frame -> a's frame) with the front end plus correspondence refinement."""
    key = (a, b)
    if key in cache:
        return cache[key]
    ca, cb = wb.clouds[a], wb.clouds[b]
    pa, da, fa = feat(wb, a)
    pb, db, fb = feat(wb, b)
    M0 = front_end(pb, pa, fb, fa, db, da)      # b -> a
    R0 = M0[:3, :3] / max(np.cbrt(abs(np.linalg.det(M0[:3, :3]))), 1e-12)
    t0 = M0[:3, 3]

    gi, gj = si.load_rgb(scene, a), si.load_rgb(scene, b)
    R1, t1 = R0, t0
    if gi is not None and gj is not None:
        k1, d1 = sift.detectAndCompute(gi, None)
        k2, d2 = sift.detectAndCompute(gj, None)
        if d1 is not None and d2 is not None and len(k1) > 8 and len(k2) > 8:
            m = bf.knnMatch(d1, d2, k=2)
            good = [x for x, y in m if x.distance < 0.8 * y.distance]
            m2 = bf.knnMatch(d2, d1, k=2)
            back = {x.queryIdx: x.trainIdx for x, y in m2 if x.distance < 0.8 * y.distance}
            good = [x for x in good if back.get(x.trainIdx) == x.queryIdx]
            if len(good) >= 20:
                uv_a = np.array([k1[x.queryIdx].pt for x in good]) / si.SCALE
                uv_b = np.array([k2[x.trainIdx].pt for x in good]) / si.SCALE
                Pa, oka = si.backproject(uv_a, ca.depth, ca.K)
                Pb, okb = si.backproject(uv_b, cb.depth, cb.K)
                keep = oka & okb
                if keep.sum() >= 20:
                    Pb, uv_a_k = Pb[keep], uv_a[keep]
                    K = ca.K
                    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

                    def resid(p):
                        R = rodrigues(p[:3]) @ R0
                        q = Pb @ R.T + p[3:]
                        z = np.clip(q[:, 2], 1e-3, None)
                        return np.concatenate([fx * q[:, 0] / z + cx - uv_a_k[:, 0],
                                               fy * q[:, 1] / z + cy - uv_a_k[:, 1]])

                    t_init = np.median(Pa[keep] - Pb @ R0.T, axis=0)
                    try:
                        sol = least_squares(resid, np.concatenate([np.zeros(3), t_init]),
                                            loss='soft_l1', f_scale=8.0, max_nfev=120)
                        R1, t1 = rodrigues(sol.x[:3]) @ R0, sol.x[3:]
                    except Exception:                   # noqa: BLE001
                        pass
    cache[key] = (R1, t1)
    cache[(b, a)] = (R1.T, -R1.T @ t1)
    return cache[key]


def chordal_mean(Rs):
    A = np.sum(Rs, axis=0)
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def main():
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    rows = []
    for scene in sorted(TASK):
        wb = Workbench(ROOT / 'agent_data' / scene, stride=32)
        cache = {}
        for tag in ('v2', 'v3', 'v4'):
            if tag not in TASK[scene]:
                continue
            t = TASK[scene][tag]
            ref = t['ref']
            ids = t['ids']
            setup(wb, t, ref)
            for j in ids[1:]:
                T = np.asarray(t['T_true'][j], float)
                R_true, t_true = T[:3, :3], T[:3, 3]
                Rd, td = pair_pose(scene, wb, ref, j, sift, bf, cache)

                cands_R, cands_t = [Rd], [td]
                for k in ids:
                    if k in (ref, j):
                        continue
                    try:
                        Rrk, trk = pair_pose(scene, wb, ref, k, sift, bf, cache)
                        Rkj, tkj = pair_pose(scene, wb, k, j, sift, bf, cache)
                    except Exception:               # noqa: BLE001
                        continue
                    cands_R.append(Rrk @ Rkj)
                    cands_t.append(Rrk @ tkj + trk)
                Rf = chordal_mean(cands_R)
                tf = np.median(np.array(cands_t), axis=0)
                rows.append({'scene': scene, 'tag': tag, 'sid': j, 'n_cands': len(cands_R),
                             'rot_direct': float(rotdeg(Rd @ R_true.T)),
                             'lat_direct': float(np.linalg.norm((td - t_true)[:2])),
                             'rot_fused': float(rotdeg(Rf @ R_true.T)),
                             'lat_fused': float(np.linalg.norm((tf - t_true)[:2]))})
                r = rows[-1]
                print('%-14s%-4s%-12s cands %d | rot %5.2f -> %5.2f | lat %5.2f -> %5.2f'
                      % (scene, tag, j, len(cands_R), r['rot_direct'], r['rot_fused'],
                         r['lat_direct'], r['lat_fused']), flush=True)
    p = ROOT / 'out' / 'multiview_fuse.json'
    p.write_text(json.dumps(rows, indent=1))
    if rows:
        g = lambda k: np.array([r[k] for r in rows])
        print('\nn=%d' % len(rows))
        for nm, a, b in (('rotation', 'rot_direct', 'rot_fused'),
                         ('lateral', 'lat_direct', 'lat_fused')):
            A, B = g(a), g(b)
            print('%-9s median %5.2f -> %5.2f   improved %2d  worsened %2d'
                  % (nm, np.median(A), np.median(B),
                     int((B < A - 0.02).sum()), int((B > A + 0.02).sum())))
        print('under 0.5 m: %d -> %d of %d'
              % (int((g('lat_direct') < 0.5).sum()), int((g('lat_fused') < 0.5).sum()), len(rows)))
    print('wrote', p)


if __name__ == '__main__':
    main()
