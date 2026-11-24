"""Calibrate the scan -> SfM-world alignment of a scene, then verify on held-out views.

The laser meshes and the ETH3D SfM poses need not share a frame exactly.  For a scene:
  1. build world clouds from GT depth for every view (ground truth geometry, eval side),
  2. pick a training set of views and fit one rigid T (scan -> world) by minimising the
     trimmed median distance from scan points near each training view to that view's depth,
  3. check T on HELD-OUT views and on the local clouds that will be used as agent input.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import open3d as o3d
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

ROOT = pathlib.Path(__file__).resolve().parents[1]
ETH = pathlib.Path('E:/depth_calib_data/eth3d_extracted')
SC = pathlib.Path('E:/mvrecon_real/scenes')
H, W = 4032, 6048


def scene_calib(scene):
    cam = {}
    for line in (ETH / scene / 'dslr_calibration_jpg' / 'cameras.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            p = line.split()
            cam[int(p[0])] = [float(p[4]), float(p[5]), float(p[6]), float(p[7])]
    pose = {}
    for line in (ETH / scene / 'dslr_calibration_jpg' / 'images.txt').read_text().splitlines():
        s = line.split()
        if len(s) == 10 and s[0].isdigit():
            q = [float(x) for x in s[1:5]]
            pose[s[9].split('/')[-1]] = (Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(),
                                         np.array([float(x) for x in s[5:8]]), cam[int(s[8])])
    return pose


def gt_cloud(scene, pose, name, stride=8):
    R, t, K = pose[name]
    D = np.fromfile(ETH / scene / 'ground_truth_depth' / 'dslr_images' / name, np.float32)
    D = D.reshape(H, W)
    D[~np.isfinite(D)] = 0
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = D[v, u]
    ok = z > 0
    pc = np.stack([(u[ok] - K[2]) / K[0] * z[ok], (v[ok] - K[3]) / K[1] * z[ok], z[ok]], 1)
    return pc, pc @ R.T + t     # own camera, world (camera-to-world interpretation)


def fit(scene, names_train, have_clouds):
    pose = scene_calib(scene)
    m = o3d.io.read_triangle_mesh(str(pathlib.Path('E:/mvrecon_real/extracted') / scene / scene /
                                      'occlusion' / 'surface_mesh.ply'))
    V = np.asarray(m.vertices, float)
    # near-surface scan points: voxel down to 3 cm, cap
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(V)
    V = np.asarray(pc.voxel_down_sample(0.03).points)
    rng = np.random.default_rng(0)
    if len(V) > 120000:
        V = V[rng.choice(len(V), 120000, replace=False)]
    gts = {}
    for n in names_train:
        own, world = gt_cloud(scene, pose, n)
        gts[n] = (own, world)
    def obj(p):
        T = np.eye(4)
        T[:3, :3] = Rot.from_rotvec(p[:3]).as_matrix()
        T[:3, 3] = p[3:]
        Q = V @ T[:3, :3].T + T[:3, 3]
        vals = []
        for n, (own, world) in gts.items():
            # scan points near this view (within the GT cloud's bounding sphere + 1 m)
            c = world.mean(0)
            rad = np.percentile(np.linalg.norm(world - c, axis=1), 95) + 1.0
            sel = np.linalg.norm(Q - c, axis=1) < rad
            if sel.sum() < 100:
                vals.append(0.5)
                continue
            d, _ = cKDTree(world).query(Q[sel])
            vals.append(float(np.clip(np.median(d), 0, 0.5)))
        return float(np.mean(vals))
    best = None
    for init in [np.zeros(6)] + [np.array([0, 0, 0, dx, dy, dz], float)
                                  for dx in (-0.2, 0.2) for dy in (-0.2, 0.2) for dz in (-0.2, 0.2)]:
        r = minimize(obj, init, method='Nelder-Mead',
                     options={'maxiter': 400, 'xatol': 1e-3, 'fatol': 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    return best, pose, V


def main(scenes):
    out = {}
    for scene in scenes:
        pose = scene_calib(scene)
        meta = json.loads((ROOT / 'eval_data' / scene / 'meta.json').read_text())
        names = [pathlib.Path(m['source_image']).name for m in meta['images']
                 if pathlib.Path(m['source_image']).name in pose]
        train = names[::2]
        test = [n for n in names if n not in train]
        best, pose, V = fit(scene, train, None)
        T = np.eye(4)
        T[:3, :3] = Rot.from_rotvec(best.x[:3]).as_matrix()
        T[:3, 3] = best.x[3:]
        # held-out check
        res = {}
        for n in test:
            own, world = gt_cloud(scene, pose, n)
            c = world.mean(0)
            rad = np.percentile(np.linalg.norm(world - c, axis=1), 95) + 1.0
            Q = V @ T[:3, :3].T + T[:3, 3]
            sel = np.linalg.norm(Q - c, axis=1) < rad
            d, _ = cKDTree(world).query(Q[sel]) if sel.any() else (np.array([np.nan]), None)
            d0, _ = cKDTree(world).query(V[sel]) if sel.any() else (np.array([np.nan]), None)
            res[n] = {'before': float(np.median(d0)), 'after': float(np.median(d))}
        out[scene] = {'transform': T.tolist(),
                      'rot_deg': float(np.degrees(np.linalg.norm(best.x[:3]))),
                      'trans_m': float(np.linalg.norm(best.x[3:])),
                      'train_obj': float(best.fun), 'held_out': res}
        print('\n== %s  rot %.2f deg  trans %.3f m  train %.4f'
              % (scene, out[scene]['rot_deg'], out[scene]['trans_m'], best.fun))
        for n, r in list(res.items())[:6]:
            print('   held-out %s  before %.3f  after %.3f' % (n, r['before'], r['after']))
    (ROOT / 'study' / 'real_meta' / 'scan_calibration.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:] or ['office'])
