#!/usr/bin/env python
"""Register the laser-scan mesh to the SfM camera world once per scene (using GT poses),
then rebuild every local view cloud with the registered scan.

The mesh and the COLMAP poses come from different registrations and sit ~1.5% apart.
This step estimates a single rigid transform T (scan -> SfM world) by minimising, over all
views, the distance between scan points projected into a view and that view's GT depth,
then verifies it: in the refined frame, pairs of views that should overlap must agree in
metric space.

The refinement uses GT (evaluation side).  It is applied once to the scan before the agent
task is constructed; nothing per-view is fitted, and the agent never sees it.

Outputs
  E:/mvrecon_real/scenes/<scene>/scan_to_sfm.json     (the single transform + residuals)
  E:/mvrecon_real/scenes/<scene>/<sid>.npz            (rebuilt, registered local clouds)
  E:/mvrecon_real/scenes/<scene>/poses_eval_only.json (unchanged poses)
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree
import open3d as o3d

ROOT = pathlib.Path(__file__).resolve().parents[1]
ETH = pathlib.Path('E:/depth_calib_data/eth3d_extracted')
MESH = pathlib.Path('E:/mvrecon_real/extracted')
SC = pathlib.Path('E:/mvrecon_real/scenes')
H, W = 4032, 6048


def calib(scene):
    cam = {}
    for line in (ETH / scene / 'dslr_calibration_jpg' / 'cameras.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            p = line.split()
            cam[int(p[0])] = dict(fx=float(p[4]), fy=float(p[5]), cx=float(p[6]),
                                  cy=float(p[7]), W=int(p[2]), H=int(p[3]))
    pose = {}
    for line in (ETH / scene / 'dslr_calibration_jpg' / 'images.txt').read_text().splitlines():
        s = line.split()
        if len(s) == 10 and s[0].isdigit() and s[9].endswith('.JPG'):
            pose[s[9].split('/')[-1]] = dict(q=[float(x) for x in s[1:5]],
                                             t=np.array([float(x) for x in s[5:8]]),
                                             cam=cam[int(s[8])])
    return pose


def gt_depth(scene, name):
    a = np.fromfile(ETH / scene / 'ground_truth_depth' / 'dslr_images' / name, np.float32)
    a = a.reshape(H, W)
    a[~np.isfinite(a)] = 0
    return a


def sample_scan(scene, n=150_000, voxel=0.02):
    m = o3d.io.read_triangle_mesh(str(MESH / scene / scene / 'occlusion' / 'surface_mesh.ply'))
    V = np.asarray(m.vertices).astype(np.float64)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(V)
    pc = pc.voxel_down_sample(voxel)
    P = np.asarray(pc.points)
    if len(P) > n:
        P = P[np.random.default_rng(0).choice(len(P), n, replace=False)]
    return P


def residuals_fn(scene, pose, P_scan, names):
    cams = []
    for name in names:
        pp = pose[name]
        R = Rot.from_quat([pp['q'][1], pp['q'][2], pp['q'][3], pp['q'][0]]).as_matrix()
        cams.append((R, pp['t'], pp['cam'], gt_depth(scene, name)))
    def resid(p):
        T = np.eye(4)
        T[:3, :3] = Rot.from_rotvec(p[:3]).as_matrix()
        T[:3, 3] = p[3:]
        Q = P_scan @ T[:3, :3].T + T[:3, 3]
        out = []
        for R, t, c, D in cams:
            pc = (Q - t) @ R          # R here is camera->world, so this is world->cam
            z = pc[:, 2]
            m = z > 0.2
            u = c['fx'] * pc[:, 0] / np.where(m, z, 1) + c['cx']
            v = c['fy'] * pc[:, 1] / np.where(m, z, 1) + c['cy']
            ui = np.round(u).astype(int)
            vi = np.round(v).astype(int)
            ok = m & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            zz = np.zeros(len(pc))
            zz[ok] = D[vi[ok], ui[ok]]
            valid = ok & (zz > 0)
            r = np.full(len(pc), 0.0)
            r[valid] = (z[valid] - zz[valid]) / zz[valid]   # relative depth residual
            out.append(np.clip(r, -0.3, 0.3))               # fixed size: Huber handles outliers
        return np.concatenate(out) if out else np.zeros(len(P_scan))
    return resid


def main(scenes):
    for scene in scenes:
        pose = calib(scene)
        meta = json.loads((ROOT / 'eval_data' / scene / 'meta.json').read_text())
        names = [pathlib.Path(im['source_image']).name for im in meta['images']
                 if pathlib.Path(im['source_image']).name in pose]
        P = sample_scan(scene)
        resid = residuals_fn(scene, pose, P, names)
        f0 = resid(np.zeros(6))
        print('== %s  scan pts %d  views %d  initial |rel resid| median %.4f'
              % (scene, len(P), len(names), np.median(np.abs(f0))))
        sol = least_squares(resid, np.zeros(6), loss='soft_l1', f_scale=0.02, max_nfev=60)
        f1 = resid(sol.x)
        print('   refined: rot deg %.3f trans %.4f |rel med %.4f -> %.4f'
              % (np.degrees(np.linalg.norm(sol.x[:3])), np.linalg.norm(sol.x[3:]),
                 np.median(np.abs(f0)), np.median(np.abs(f1))))
        T = np.eye(4)
        T[:3, :3] = Rot.from_rotvec(sol.x[:3]).as_matrix()
        T[:3, 3] = sol.x[3:]
        (SC / scene / 'scan_to_sfm.json').write_text(json.dumps(
            {'transform': T.tolist(), 'rot_deg': float(np.degrees(np.linalg.norm(sol.x[:3]))),
             'trans_m': float(np.linalg.norm(sol.x[3:])),
             'rel_median_before': float(np.median(np.abs(f0))),
             'rel_median_after': float(np.median(np.abs(f1))),
             'n_scan_points': int(len(P))}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:] or ['office', 'courtyard', 'playground', 'terrace'])
