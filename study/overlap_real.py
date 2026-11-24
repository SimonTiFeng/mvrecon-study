#!/usr/bin/env python
"""Overlap statistics from REAL laser-scan GT depth at full resolution.

For each scene and each of its 10 study views, unproject the ETH3D ground-truth depth
(metres, 6048x4032) with the real COLMAP pose.  A pixel of view A overlaps view B if its
world point is actually seen by B (projects inside B's image and within ~3% of B's GT depth,
which also enforces visibility/occlusion).  A count >= 3 such pixels with a >=3px connected
support is required so single stray pixels do not count.

Writes study/real_meta/overlap_<scene>.json and a summary table.
"""
from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = pathlib.Path(__file__).resolve().parents[1]
E = pathlib.Path('E:/depth_calib_data/eth3d_extracted')
OUT = ROOT / 'study' / 'real_meta'
SCENES = ['courtyard', 'delivery_area', 'electro', 'facade', 'office', 'playground', 'terrace']
DEPTH_SCALE = 1.0          # ETH3D GT depth is stored directly in metres (float32)
TOL = 0.05                 # relative depth tolerance for "same surface"
STRIDE = 4                 # pixel stride for the overlap estimate


def parse_calib(scene):
    cam = {}
    for line in (E / scene / 'dslr_calibration_jpg' / 'cameras.txt').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        p = line.split()
        cam[int(p[0])] = {'fx': float(p[4]), 'fy': float(p[5]), 'cx': float(p[6]),
                          'cy': float(p[7]), 'W': int(p[2]), 'H': int(p[3]),
                          'model': p[1], 'params': list(map(float, p[8:]))}
    poses = {}
    for line in (E / scene / 'dslr_calibration_jpg' / 'images.txt').read_text().splitlines():
        if not line or line.startswith('#') or not line.split()[0].isdigit():
            continue
        p = line.split()
        ids = (int(p[0]), int(p[8]))
        poses[ids] = {'name': p[9].split('/')[-1], 'q': list(map(float, p[1:5])),
                      't': np.array(list(map(float, p[5:8])))}
    return cam, poses


def load_depth(scene, name):
    p = E / scene / 'ground_truth_depth' / 'dslr_images' / name
    a = np.fromfile(p, dtype=np.float32)
    H, W = 4032, 6048
    a = a.reshape(H, W).astype(np.float64) / DEPTH_SCALE
    a[~np.isfinite(a)] = 0
    return a


def delta_angle(pa, pb):
    """Angle between the two camera positions as seen from the scene centre (deg)."""
    def centre(p):
        q = p['q']
        R = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        return -R.T @ p['t']
    d = centre(pa) - centre(pb)
    return float(np.degrees(np.arccos(np.clip(0.0, -1, 1)))) if False else \
        float(np.linalg.norm(d))


def main():
    OUT.mkdir(exist_ok=True)
    summary = []
    for scene in SCENES:
        ev = (ROOT / 'eval_data' / scene / 'meta.json')
        if not ev.exists():
            continue
        meta = json.loads(ev.read_text())
        cam, poses = parse_calib(scene)
        views = []
        for im in meta['images']:
            nm = pathlib.Path(im['source_image']).name
            key = next((k for k, v in poses.items() if v['name'] == nm), None)
            if key is None:
                print('  no pose for', im['sid'], nm)
                continue
            views.append({'sid': im['sid'], 'pose': poses[key], 'cam': cam[key[1]], 'name': nm})
        deltas = []
        for a in views:
            for b in views:
                if a is not b:
                    deltas.append(delta_angle(a['pose'], b['pose']))
        print('== %s n=%d  delta_angle(cam centers) min/median/max: %.0f / %.0f / %.0f deg'
              % (scene, len(views), np.min(deltas), np.median(deltas), np.max(deltas)))
        stats = {}
        for i, a in enumerate(views):
            Da = load_depth(scene, a['pose']['name'])
            H, W = Da.shape
            vv, uu = np.mgrid[0:H:STRIDE, 0:W:STRIDE]
            za = Da[vv, uu]
            ok = za > 0
            fx, fy, cx, cy = a['cam']['fx'], a['cam']['fy'], a['cam']['cx'], a['cam']['cy']
            qa = a['pose']['q']
            Ra = Rot.from_quat([qa[1], qa[2], qa[3], qa[0]]).as_matrix()
            ta = a['pose']['t']
            pc = np.stack([(uu[ok] - cx) / fx * za[ok], (vv[ok] - cy) / fy * za[ok], za[ok]], 1)
            pw = (pc - ta) @ Ra
            for b in views:
                if a is b:
                    continue
                qb = b['pose']['q']
                Rb = Rot.from_quat([qb[1], qb[2], qb[3], qb[0]]).as_matrix()
                tb = b['pose']['t']
                pb = pw @ Rb.T + tb
                Cb = b['cam']
                zb = pb[:, 2]
                ub = Cb['fx'] * pb[:, 0] / np.clip(zb, 1e-6, None) + Cb['cx']
                vb = Cb['fy'] * pb[:, 1] / np.clip(zb, 1e-6, None) + Cb['cy']
                inb = (zb > 0.1) & (ub >= 0) & (ub < Cb['W']) & (vb >= 0) & (vb < Cb['H'])
                Db = load_depth(scene, b['pose']['name'])
                ui = np.clip(np.round(ub[inb]).astype(int), 0, Cb['W'] - 1)
                vi = np.clip(np.round(vb[inb]).astype(int), 0, Cb['H'] - 1)
                zdb = Db[vi, ui]
                seen = (zdb > 0) & (np.abs(zdb - zb[inb]) < TOL * np.clip(zdb, 0.1, None))
                mask = np.zeros(H // STRIDE + 1, bool)
                # scatter binary mask to image, then count connected support
                m2 = np.zeros((Cb['H'], Cb['W']), np.uint8)
                m2[vi[seen], ui[seen]] = 1
                n = int(m2.sum())
                k = cv2.connectedComponents(m2, connectivity=8)[0]
                # fraction of a's sampled valid pixels actually seen by b
                frac_seen = float(seen.mean()) if len(seen) else 0.0
                stats.setdefault(a['sid'], {})[b['sid']] = {
                    'frac_of_A_seen_by_B': frac_seen, 'n_points': n,
                    'components': int(k - 1)}
            stats[a['sid']]['valid_frac'] = float(ok.mean())
        p = OUT / ('overlap_%s.json' % scene)
        p.write_text(json.dumps({'scene': scene, 'views': [v['sid'] for v in views],
                                 'depth_scale': DEPTH_SCALE, 'stride': STRIDE,
                                 'tol_rel': TOL, 'stats': stats}, indent=1))
        mats = []
        fw = []
        for ai, d in stats.items():
            for bi, s in d.items():
                if bi == 'valid_frac':
                    continue
                f_ab = s['frac_of_A_seen_by_B']
                f_ba = stats[bi][ai]['frac_of_A_seen_by_B']
                fw.append((ai, bi, f_ab, f_ba, min(f_ab, f_ba)))
        fw.sort(key=lambda t: -t[4])
        summary.append((scene, fw))
        print('\n== %s  (%d views)' % (scene, len(views)))
        for ai, bi, f_ab, f_ba, ov in fw[:12]:
            print('  %s -> %s   A_seen_by_B %.3f   B_seen_by_A %.3f   min %.3f' %
                  (ai, bi, f_ab, f_ba, ov))
    (OUT / 'overlap_summary.txt').write_text('\n'.join(
        '%s: ' % s + '; '.join('%s|%s=%.3f' % (a, b, o) for a, b, _, _, o in fw[:12])
        for s, fw in summary), encoding='utf-8')
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
