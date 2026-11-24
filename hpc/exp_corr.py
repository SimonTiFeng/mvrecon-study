"""Correspondence-method sweep for sparse-view monocular registration. Runs on HPC2.

Reads the corrected handoff package layout, so it needs only numpy + OpenCV:

  <root>/agent_input/<scene>/views.json          intrinsics (original resolution) + file names
  <root>/agent_input/<scene>/rgb/<sid>.jpg        half-resolution photographs
  <root>/agent_input/<scene>/model_output/<sid>_depth_metric_q4.npy   metric z-depth, 1/4 scale
  <root>/evaluation_only/task.json               reference transform and injected start pose
  <root>/evaluation_only/baseline_matrices.json  the classical front end's answer, for comparison

Everything is done at 1/4 resolution so the photograph and the depth map are pixel-aligned: the
photograph is resized to the depth shape and the intrinsics are scaled by 0.25.

For every (target, matcher, mask, solve) this records the resulting errors and the cost, so the
Pareto front over accuracy and cost can be read off directly.

  matcher : sift | akaze | orb | sift_akaze
  mask    : none | sky | sky_far   (sky = colour rule; sky_far adds a distance cap at 3x median depth)
  solve   : rot_fixed  (translation only, rotation from the front end)
            corrected  (iterate a small rotation + translation fit over the same correspondences)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import cv2
import numpy as np


def load_view(root, scene, sid):
    vj = json.loads((root / 'agent_input' / scene / 'views.json').read_text())
    v = next(x for x in vj['views'] if x['sid'] == sid)
    dep = np.load(root / 'agent_input' / scene / 'model_output'
                  / (sid + '_depth_metric_q4.npy')).astype(np.float32)
    rgb = cv2.imread(str(root / 'agent_input' / scene / 'rgb' / (sid + '.jpg')))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (dep.shape[1], dep.shape[0]), interpolation=cv2.INTER_AREA)
    K = np.asarray(v['K'], float) * (dep.shape[1] / float(v['W']))
    return {'sid': sid, 'depth': dep, 'rgb': rgb, 'K': K}


def detect(kind, img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    t0 = time.time()
    if kind == 'sift':
        d = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
        kp, des = d.detectAndCompute(g, None)
    elif kind == 'akaze':
        d = cv2.AKAZE_create(threshold=0.0008)
        kp, des = d.detectAndCompute(g, None)
    elif kind == 'orb':
        d = cv2.ORB_create(nfeatures=4000, fastThreshold=12)
        kp, des = d.detectAndCompute(g, None)
    else:
        raise ValueError(kind)
    return kp, des, time.time() - t0


def match(kind, k1, d1, k2, d2, ratio=0.8):
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return [], 0.0
    norm = cv2.NORM_HAMMING if kind in ('akaze', 'orb') else cv2.NORM_L2
    bf = cv2.BFMatcher(norm)
    t0 = time.time()
    m = bf.knnMatch(d1, d2, k=2)
    good = [a for a, b in m if a.distance < ratio * b.distance]
    m2 = bf.knnMatch(d2, d1, k=2)
    back = {a.queryIdx: a.trainIdx for a, b in m2 if a.distance < ratio * b.distance}
    good = [a for a in good if back.get(a.trainIdx) == a.queryIdx]
    return good, time.time() - t0


def backproject(uv, view):
    dep, K = view['depth'], view['K']
    H, W = dep.shape
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    ok = (u >= 1) & (u < W - 1) & (v >= 1) & (v < H - 1)
    z = np.zeros(len(uv), np.float32)
    z[ok] = dep[v[ok], u[ok]]
    ok = ok & np.isfinite(z) & (z > 0.1) & (z < 300)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = np.full((len(uv), 3), np.nan, np.float32)
    P[ok, 0] = (uv[ok, 0] - cx) / fx * z[ok]
    P[ok, 1] = (uv[ok, 1] - cy) / fy * z[ok]
    P[ok, 2] = z[ok]
    return P, ok


def sky_test(rgb, uv):
    col = rgb[np.clip(uv[:, 1].astype(int), 0, rgb.shape[0] - 1),
              np.clip(uv[:, 0].astype(int), 0, rgb.shape[1] - 1)].astype(np.float32)
    r, g, b = col[:, 0], col[:, 1], col[:, 2]
    return (b > r + 25) & (b > 110) & (g > r)


def small_rot(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def fit_correction(Pr, Pm, R0, iters=6):
    w = np.zeros(3)
    t = np.median(Pr - Pm @ R0.T, axis=0)
    for _ in range(iters):
        q = Pm @ R0.T
        n = len(q)
        # rows of [-[q]x | I]; built per point, which is clearer than index gymnastics
        A2 = np.zeros((3 * n, 6))
        for i in range(n):
            A2[3 * i:3 * i + 3, 0:3] = [[0, q[i, 2], -q[i, 1]],
                                        [-q[i, 2], 0, q[i, 0]],
                                        [q[i, 1], -q[i, 0], 0]]
            A2[3 * i:3 * i + 3, 3:6] = np.eye(3)
        b = (Pr - q).ravel()
        res = np.linalg.norm((Pr - q) - (np.cross(w, q) + t), axis=1)
        sc = max(np.median(res), 1e-3)
        wts = 1.0 / (1.0 + (res / (2.0 * sc)) ** 2)
        sol, *_ = np.linalg.lstsq(A2 * np.repeat(wts, 3)[:, None],
                                  b * np.repeat(wts, 3), rcond=None)
        w, t = sol[:3], sol[3:]
    return small_rot(w) @ R0, t


def rotdeg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scenes', default='')
    ap.add_argument('--outliers', default='0,0.1,0.2,0.3',
                    help='comma list: fraction of correspondences replaced by gross outliers')
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    task = json.loads((root / 'evaluation_only' / 'task.json').read_text())
    base = json.loads((root / 'evaluation_only' / 'baseline_matrices.json').read_text())
    scenes = [s for s in a.scenes.split(',') if s] or sorted(task)
    out = open(a.out, 'w')
    n = 0
    for scene in scenes:
        views = {}
        for tag in ('v2', 'v3', 'v4'):
            if tag not in task.get(scene, {}):
                continue
            t = task[scene][tag]
            ref = t['ref']
            for sid in t['ids'][1:]:          # the reference is fixed; others are registered
                if ref not in views:
                    views[ref] = load_view(root, scene, ref)
                if sid not in views:
                    views[sid] = load_view(root, scene, sid)
                vr, vm = views[ref], views[sid]
                T = np.asarray(t['T_true'][sid], float)
                R_true, t_true = T[:3, :3], T[:3, 3]
                key = '%s|%s|%s|%s' % (scene, tag, ref, sid)
                M = np.asarray(base.get(key, np.eye(4).tolist()), float)
                R_base = M[:3, :3] / max(np.cbrt(abs(np.linalg.det(M[:3, :3]))), 1e-12)

                feats = {}
                for kind in ('sift', 'akaze', 'orb'):
                    kp_r, de_r, dtr = detect(kind, vr['rgb'])
                    kp_m, de_m, dtm = detect(kind, vm['rgb'])
                    feats[kind] = (kp_r, de_r, kp_m, de_m, dtr + dtm)

                for mkind in ('sift', 'akaze', 'orb', 'sift_akaze'):
                    if mkind == 'sift_akaze':
                        good = []
                        tmatch = 0.0
                        for kind in ('sift', 'akaze'):
                            kp_r, de_r, kp_m, de_m, _ = feats[kind]
                            g, tm = match(kind, kp_r, de_r, kp_m, de_m)
                            good += [(kind, x) for x in g]
                            tmatch += tm
                        pairs = []
                        for kind, x in good:
                            kp_r, de_r, kp_m, de_m, _ = feats[kind]
                            pairs.append((kp_r[x.queryIdx].pt, kp_m[x.trainIdx].pt))
                        n_feat = sum(len(feats[k][0]) + len(feats[k][2]) for k in ('sift', 'akaze'))
                        tdet = sum(feats[k][4] for k in ('sift', 'akaze'))
                    else:
                        kp_r, de_r, kp_m, de_m, tdet = feats[mkind]
                        g, tmatch = match(mkind, kp_r, de_r, kp_m, de_m)
                        pairs = [(kp_r[x.queryIdx].pt, kp_m[x.trainIdx].pt) for x in g]
                        n_feat = len(kp_r) + len(kp_m)
                    if len(pairs) < 8:
                        continue
                    uv_r = np.array([p[0] for p in pairs], np.float32)
                    uv_m = np.array([p[1] for p in pairs], np.float32)
                    Pr, okr = backproject(uv_r, vr)
                    Pm, okm = backproject(uv_m, vm)
                    keep = okr & okm
                    sky_r = sky_test(vr['rgb'], uv_r)
                    sky_m = sky_test(vm['rgb'], uv_m)
                    for mask in ('none', 'sky', 'sky_far'):
                        k = keep.copy()
                        if mask in ('sky', 'sky_far'):
                            k &= ~sky_r & ~sky_m
                        if mask == 'sky_far':
                            capr = 3.0 * np.median(np.linalg.norm(Pr[keep], axis=1))
                            capm = 3.0 * np.median(np.linalg.norm(Pm[keep], axis=1))
                            k &= (np.linalg.norm(Pr, axis=1) < capr) & \
                                 (np.linalg.norm(Pm, axis=1) < capm)
                        if k.sum() < 8:
                            continue
                        PKr, PKm = Pr[k], Pm[k]
                        rng = np.random.default_rng(12345)
                        for ofrac in [float(z) for z in a.outliers.split(',')]:
                          Pmw = PKm
                          if ofrac > 0:
                            n_out = int(round(ofrac * len(PKm)))
                            if n_out > 0:
                                Pmw = PKm.copy()
                                idx = rng.choice(len(Pmw), n_out, replace=False)
                                scale = max(np.median(np.linalg.norm(PKw := PKm, axis=1)), 1.0)
                                Pmw[idx] += rng.normal(0, 0.5 * scale, size=(n_out, 3))
                          for solve in ('rot_fixed', 'corrected'):
                            t0 = time.time()
                            if solve == 'rot_fixed':
                                R, tt = R_base, np.median(PKr - Pmw @ R_base.T, axis=0)
                            else:
                                R, tt = fit_correction(PKr, Pmw, R_base)
                            ms = (time.time() - t0) * 1000
                            rec = {'scene': scene, 'tag': tag, 'ref': ref, 'sid': sid,
                                   'matcher': mkind, 'mask': mask, 'solve': solve,
                                   'outlier_frac': ofrac,
                                   'n_feat': int(n_feat), 'n_match': int(len(pairs)),
                                   'n_used': int(k.sum()),
                                   'rot_err': rotdeg(R @ R_true.T),
                                   'lat_err': float(np.linalg.norm((tt - t_true)[:2])),
                                   'trans_err': float(np.linalg.norm(tt - t_true)),
                                   'base_rot': rotdeg(R_base @ R_true.T),
                                   'base_trans': float(np.linalg.norm((M[:3, 3] - t_true))),
                                   'ms_detect': tdet * 1000, 'ms_match': tmatch * 1000,
                                   'ms_solve': ms}
                            out.write(json.dumps(rec) + '\n')
                            n += 1
        print('done', scene, n, flush=True)
    out.close()
    print('wrote', a.out, n)


if __name__ == '__main__':
    main()
