#!/usr/bin/env python
"""Check the previous study's GT pose formula against real GT-depth geometry.

The previous study built T_true with  R_ij = Rr Rj^T, t_ij = tr - Rr Rj^T tj, treating the
EMS images.txt poses as world->camera.  This script builds the same transform purely from
ETH3D GT depth (unproject both views to world, fit one rigid transform A with world->cam and
one with the camera->world interpretation) and reports which matches T_true and which makes
the clouds agree.
"""
import json
import pathlib
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = pathlib.Path(__file__).resolve().parents[1]
E = pathlib.Path('E:/depth_calib_data/eth3d_extracted')
H, W = 4032, 6048


def calib(scene):
    pose = {}
    for line in (E / scene / 'dslr_calibration_jpg' / 'images.txt').read_text().splitlines():
        s = line.split()
        if len(s) == 10 and s[0].isdigit() and s[9].endswith('.JPG'):
            q = [float(x) for x in s[1:5]]
            pose[s[9].split('/')[-1]] = dict(R=Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(),
                                             t=np.array([float(x) for x in s[5:8]]))
    return pose


def cloud(scene, name, k):
    d = np.fromfile(E / scene / 'ground_truth_depth' / 'dslr_images' / name, np.float32)
    d = d.reshape(H, W)
    d[~np.isfinite(d)] = 0
    v, u = np.mgrid[0:H:16, 0:W:16]
    z = d[v, u]
    ok = z > 0
    return np.stack([(u[ok] - k[2]) / k[0] * z[ok], (v[ok] - k[3]) / k[1] * z[ok], z[ok]], 1)


def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    Hm = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(Hm)
    dd = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, dd]) @ U.T
    return R, cb - R @ ca


def main():
    res = []
    for scene in ['office', 'playground', 'courtyard', 'terrace']:
        p = calib(scene)
        task = json.loads((ROOT / 'eval_data' / 'task.json').read_text())[scene]['v2']
        ref, tgt = task['ref'], task['ids'][1]
        meta = json.loads((ROOT / 'eval_data' / scene / 'meta.json').read_text())
        src = {m['sid']: pathlib.Path(m['source_image']).name for m in meta['images']}
        Pr, pt = p[src[ref]], p[src[tgt]]
        cam = {}
        for line in (E / scene / 'dslr_calibration_jpg' / 'cameras.txt').read_text().splitlines():
            if line and not line.startswith('#'):
                q = line.split()
                cam[int(q[0])] = [float(q[4]), float(q[5]), float(q[6]), float(q[7])]
        cid = {}
        for line in (E / scene / 'dslr_calibration_jpg' / 'images.txt').read_text().splitlines():
            q = line.split()
            if len(q) == 10 and q[0].isdigit():
                cid[q[9].split('/')[-1]] = int(q[8])
        kk = {sid: cam[cid[src[sid]]] for sid in (ref, tgt)}
        cr = cloud(scene, src[ref], kk[ref])
        ct = cloud(scene, src[tgt], kk[tgt])
        # world->cam interpretation: p_cam = R p_w + t   ;  cam->world: p_w = R p_cam + t
        # Ground truth relative transform target-cam -> ref-cam under world->cam:
        #   T_w2c = [Rr Rj^T | tr - Rr Rj^T tj]
        R1 = Pr['R'] @ pt['R'].T
        t1 = Pr['t'] - Pr['R'] @ pt['R'].T @ pt['t']
        # under camera->world:  p_ref = Rr (Rj^T (p_t - tj)) + tr
        R2 = Pr['R'] @ pt['R'].T
        t2 = Pr['t'] - R2 @ pt['t']
        # data-driven: fit rigid target-cam -> ref-cam using camera->world to make both clouds
        Wr = cr @ Pr['R'].T + Pr['t']
        Wt = ct @ pt['R'].T + pt['t']
        # nearest-point correspondences
        from scipy.spatial import cKDTree
        d, idx = cKDTree(Wr).query(Wt)
        keep = d < 0.05
        if keep.sum() > 20:
            # ref-cam coords of the target cloud points under cam->world interpretation
            Xt_cam = ct[keep]
            Xr_cam = (Wr[idx[keep]] - Pr['t']) @ Pr['R']
            Rf, tf = kabsch(Xt_cam, Xr_cam)
        else:
            Rf, tf = np.eye(3), np.zeros(3)
        def ang(R):
            return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        Tprev = np.array(task['T_true'][tgt])
        res.append(dict(scene=scene, ref=ref, tgt=tgt, n_corr=1.0 * keep.sum(),
                        formula_w2c_rot=ang(R1), formula_w2c_trans=float(np.linalg.norm(t1)),
                        formula_c2w_rot=ang(R2), formula_c2w_trans=float(np.linalg.norm(t2)),
                        data_rot=ang(Rf), data_trans=float(np.linalg.norm(tf)),
                        prev_T_rot=ang(Tprev[:3, :3]), prev_T_trans=float(np.linalg.norm(Tprev[:3, 3])),
                        data_vs_formula_w2c_rot=ang(Rf @ R1.T),
                        data_vs_formula_c2w_rot=ang(Rf @ R2.T),
                        data_vs_prev_rot=ang(Rf @ Tprev[:3, :3].T),
                        data_vs_prev_trans=float(np.linalg.norm(tf - Tprev[:3, 3]))))
    for r in res:
        print(json.dumps(r))
    (ROOT / 'out' / 'real_convention_check.json').write_text(json.dumps(res, indent=1))


def kk_src(scene):
    return json.loads((ROOT / 'eval_data' / scene / 'meta.json').read_text())['images']


if __name__ == '__main__':
    main()
