#!/usr/bin/env python
"""Build real-geometry local view clouds from registered ETH3D laser scan meshes.

For each study view V of a scene:
  1. subsample scan mesh vertices (voxel thinning) to a manageable density;
  2. keep only points V can actually see: inside V's frustum, in front of V, and not
     occluded by another scan point closer along the ray (z-buffer test per small image
     cell, tolerance ~1.5% of depth);
  3. colour them from the real RGB image at full resolution;
  4. express them in V's OWN camera frame  (p_cam = R^T (p_world - t)), never world.

Visibility uses the scan itself, which is what an agent would have if handed a partial
reconstruction; no monocular depth is involved.

Writes  E:/mvrecon_real/scenes/<scene>/<sid>.npz   (points_cam, colors, K, W, H, pose)
and     E:/mvrecon_real/scenes/<scene>/meta.json   (agent-visible: K, W, H, RGB name)
Poses are also written to  E:/mvrecon_real/scenes/<scene>/poses_eval_only.json  (eval side).

Usage: python study/build_real_local_clouds.py office courtyard playground terrace
"""
from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = pathlib.Path(__file__).resolve().parents[1]
ETH = pathlib.Path('E:/depth_calib_data/eth3d_extracted')
MESH = pathlib.Path('E:/mvrecon_real/extracted')
OUT = pathlib.Path('E:/mvrecon_real/scenes')
VOXEL = 0.015         # m, scan thinning (before visibility)
MAX_POINTS = 120_000  # per local view, keeps pillar geometry dense enough for ICP
ZBUF_TOL = 0.015      # 1.5% relative depth tolerance for the occlusion test
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


def load_scan(scene):
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(str(MESH / scene / scene / 'occlusion' / 'surface_mesh.ply'))
    V = np.asarray(m.vertices).astype(np.float64)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(V)
    pc = pc.voxel_down_sample(VOXEL)
    return np.asarray(pc.points)


def visible(Pw, pose, cam):
    # EMS images.txt stores a CAMERA-TO-WORLD pose: C = [R | t] maps cam -> world,
    # so world -> cam is  p_cam = R^T (p_world - t).
    R = Rot.from_quat([pose['q'][1], pose['q'][2], pose['q'][3], pose['q'][0]]).as_matrix()
    pc = (Pw - pose['t']) @ R
    z = pc[:, 2]
    inz = z > 0.15
    u = np.zeros(len(pc))
    v = np.zeros(len(pc))
    u[inz] = cam['fx'] * pc[inz, 0] / z[inz] + cam['cx']
    v[inz] = cam['fy'] * pc[inz, 1] / z[inz] + cam['cy']
    inf = inz & (u >= 1) & (u < W - 1) & (v >= 1) & (v < H - 1)
    # z-buffer in coarse image cells, vectorised
    cell = 24
    key = (np.round(u / cell).astype(np.int64) * 10000
           + np.round(v / cell).astype(np.int64))
    zinf = np.where(inf, z, np.inf)
    # minimum z per cell (np.minimum.at over unique cells is much faster than a Python loop)
    uniq, inv = np.unique(key, return_inverse=True)
    zmin_cell = np.full(len(uniq), np.inf)
    np.minimum.at(zmin_cell, inv, zinf)
    keep = inf & (zinf <= zmin_cell[inv] * (1 + ZBUF_TOL))
    return pc[keep], keep          # already in this view's camera frame


def rgb_at(img, u, v):
    ui = np.clip(np.round(u).astype(int), 0, W - 1)
    vi = np.clip(np.round(v).astype(int), 0, H - 1)
    return img[vi, ui].astype(np.float32) / 255.0


def main(scenes):
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    for scene in scenes:
        pose = calib(scene)
        scan = load_scan(scene)
        meta = json.loads((ROOT / 'eval_data' / scene / 'meta.json').read_text())
        d = OUT / scene
        d.mkdir(parents=True, exist_ok=True)
        ameta = {'scene': scene, 'views': []}
        poses = {}
        for im in meta['images']:
            sid = im['sid']
            name = pathlib.Path(im['source_image']).name
            if name not in pose:
                continue
            P = pose[name]
            cam = P['cam']
            pc, keep = visible(scan, P, cam)
            if len(pc) < 500:
                report.append((scene, sid, name, 0, 'too_few'))
                continue
            if len(pc) > MAX_POINTS:
                idx = np.random.default_rng(0).choice(len(pc), MAX_POINTS, replace=False)
                pc = pc[idx]
            img = cv2.cvtColor(cv2.imread(str(ETH / scene / 'images' / 'dslr_images' / name)),
                               cv2.COLOR_BGR2RGB)
            if img is None:
                report.append((scene, sid, name, len(pc), 'no_rgb'))
                continue
            K = np.array([[cam['fx'], 0, cam['cx']], [0, cam['fy'], cam['cy']], [0, 0, 1]],
                         float)
            u = cam['fx'] * pc[:, 0] / pc[:, 2] + cam['cx']
            v = cam['fy'] * pc[:, 1] / pc[:, 2] + cam['cy']
            col = rgb_at(img, u, v)
            R_c2w = Rot.from_quat([P['q'][1], P['q'][2], P['q'][3], P['q'][0]]).as_matrix()
            np.savez_compressed(d / (sid + '.npz'), points=pc.astype(np.float32),
                                colors=col.astype(np.float32), K=K, W=W, H=H,
                                rgb=name)
            ameta['views'].append({'sid': sid, 'K': K.tolist(), 'W': W, 'H': H,
                                   'rgb': name})
            poses[sid] = {'R': R_c2w.tolist(), 't': P['t'].tolist(), 'rgb': name,
                          'convention': 'camera_to_world'}
            report.append((scene, sid, name, len(pc), 'ok'))
        (d / 'meta.json').write_text(json.dumps(ameta, indent=1))
        (d / 'poses_eval_only.json').write_text(json.dumps(poses, indent=1))
    (OUT / 'build_report.json').write_text(json.dumps(report, indent=1))
    for r in report:
        print(r)


if __name__ == '__main__':
    main(sys.argv[1:] or ['office', 'courtyard', 'playground', 'terrace'])
