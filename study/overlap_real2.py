#!/usr/bin/env python
"""Overlap statistics from REAL ETH3D laser-scan GT depth (full resolution, metres).

Unprojects view A's GT depth with its real COLMAP pose; a pixel counts as *seen by* view B
if it projects into B's image and B's GT depth at that pixel agrees within TOL (relative),
which also encodes occlusion.  Overlap(A,B) = min over directions of that fraction.

Writes study/real_meta/overlap_<scene>.json and overlap_summary.txt.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = pathlib.Path(__file__).resolve().parents[1]
E = pathlib.Path('E:/depth_calib_data/eth3d_extracted')
OUT = ROOT / 'study' / 'real_meta'
SCENES = ['office', 'playground', 'terrace', 'courtyard',
          'delivery_area', 'electro', 'facade']
TOL = 0.05
STRIDE = 8
H, W = 4032, 6048


def calib(scene):
    cam = {}
    for line in (E / scene / 'dslr_calibration_jpg' / 'cameras.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            p = line.split()
            cam[int(p[0])] = dict(fx=float(p[4]), fy=float(p[5]), cx=float(p[6]),
                                  cy=float(p[7]), W=int(p[2]), H=int(p[3]))
    pose = {}
    for line in (E / scene / 'dslr_calibration_jpg' / 'images.txt').read_text().splitlines():
        s = line.split()
        if len(s) != 10 or not s[0].isdigit() or not s[9].endswith(('.JPG', '.jpg')):
            continue
        pose[s[9].split('/')[-1]] = dict(name=s[9].split('/')[-1], q=[float(x) for x in s[1:5]],
                                         t=np.array([float(x) for x in s[5:8]]),
                                         cam=cam[int(s[8])])
    return pose


def depth(scene, name):
    a = np.fromfile(E / scene / 'ground_truth_depth' / 'dslr_images' / name, np.float32)
    a = a.reshape(H, W)
    a[~np.isfinite(a)] = 0
    return a


def R_of(p):
    q = p['q']
    return Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def world_points(scene, sid, pose, stride=STRIDE):
    d = depth(scene, pose[sid]['name'])
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = d[v, u]
    ok = z > 0
    c = pose[sid]['cam']
    pc = np.stack([(u[ok] - c['cx']) / c['fx'] * z[ok],
                   (v[ok] - c['cy']) / c['fy'] * z[ok], z[ok]], 1)
    R = R_of(pose[sid])
    P = (pc - pose[sid]['t']) @ R
    return P, float(ok.mean())


def seen_fraction(P, scene, sid, pose):
    c = pose[sid]['cam']
    R = R_of(pose[sid])
    pb = P @ R.T + pose[sid]['t']
    z = pb[:, 2]
    u = c['fx'] * pb[:, 0] / np.where(np.abs(z) > 1e-6, z, 1e-6) + c['cx']
    v = c['fy'] * pb[:, 1] / np.where(np.abs(z) > 1e-6, z, 1e-6) + c['cy']
    inb = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if inb.sum() == 0:
        return 0.0
    ui = np.clip(np.round(u[inb]).astype(int), 0, W - 1)
    vi = np.clip(np.round(v[inb]).astype(int), 0, H - 1)
    db = depth(scene, pose[sid]['name'])[vi, ui]
    good = (db > 0) & (np.abs(db - z[inb]) < TOL * db)
    return float(good.sum() / len(P))


def main(scenes=None):
    OUT.mkdir(exist_ok=True)
    out = {}
    for scene in (scenes or SCENES):
        if not (ROOT / 'eval_data' / scene / 'meta.json').exists():
            continue
        meta = json.loads((ROOT / 'eval_data' / scene / 'meta.json').read_text())
        pose = calib(scene)
        sids = [im['sid'] for im in meta['images']
                if pathlib.Path(im['source_image']).name in pose]
        POSE = {im['sid']: pose[pathlib.Path(im['source_image']).name]
                for im in meta['images'] if im['sid'] in sids}
        P = {}
        vf = {}
        for s in sids:
            P[s], vf[s] = world_points(scene, s, POSE)
        ov = {}
        for a in sids:
            ov[a] = {b: (seen_fraction(P[a], scene, b, POSE) if b != a else 1.0)
                     for b in sids}
        sym = {(a, b): min(ov[a][b], ov[b][a]) for a in sids for b in sids if a < b}
        out[scene] = dict(views=sids, valid_frac=vf, directed=ov,
                          symmetric={'%s|%s' % k: v for k, v in sym.items()})
        ranked = sorted(sym.items(), key=lambda kv: -kv[1])
        print('\n== %s  n=%d valid_frac med %.2f' % (scene, len(sids), np.median(list(vf.values()))))
        for (a, b), v in ranked[:10]:
            print('  %s %s  overlap %.3f' % (a, b, v))
        print('  ... min %.3f' % ranked[-1][1])
    (OUT / 'overlap_real_all.json').write_text(json.dumps(out, indent=1))
    print('\nwrote', OUT / 'overlap_real_all.json')


if __name__ == '__main__':
    main(sys.argv[1:] or None)
