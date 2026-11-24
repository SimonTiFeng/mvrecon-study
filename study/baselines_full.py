#!/usr/bin/env python
"""Traditional baselines for the interactive-registration study, ALL 40 instances.

Every method starts from the identical injected state D, sees only agent-visible
inputs (RGB + own-view depth + K), and is scored by eval/protocol.py against T_true.

Methods
-------
identity          no operation
A_fpfh_icp        FPFH + RANSAC + point-to-plane ICP (project baseline A)
B_sift_rigid      SIFT mutual-ratio matches -> 3D RANSAC rigid, no mask
C_sift_rigid_sky  same, sky-coloured matches excluded
D_rotA_sift_t     A's rotation + robust median translation from SIFT matches
E_rotA_sift_t_sky same, sky matches excluded
F_rotA_sift_corr  E + iterated small-rotation correction (accepted if spread improves)
G_rotA_sift_reproj E + robust reprojection-error refinement
H_sift_rigid_reproj C's rigid init + reprojection refinement (SIFT-only pipeline)

No method is restricted to a GT-selected subset.  The 25-target "coarse rotation
< 20 deg" view is produced later by the summary script, as a reporting subset only.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import time

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'workbench'))
sys.path.insert(0, str(ROOT / 'eval'))
from wb import Cloud  # noqa: E402
from protocol import score, rotdeg  # noqa: E402

TASK = json.loads((ROOT / 'eval_data' / 'task.json').read_text())
AG = ROOT / 'agent_data'
SCALE = 0.25
RATIO = 0.80
OUT = ROOT / 'out' / 'agent_study_20261123'


class TwoView:
    """Minimal Workbench-compatible surface that loads only the two views."""

    def __init__(self, scene, ref, sid, stride=32):
        self.scene = scene
        d = AG / scene
        meta = json.loads((d / 'meta.json').read_text())
        self.clouds = {}
        for im in meta['images']:
            if im['sid'] not in (ref, sid):
                continue
            dep = np.load(d / im['depth_file'])
            K = np.asarray(im['K'], float)
            H, W = dep.shape
            ys, xs = np.mgrid[0:H:stride, 0:W:stride]
            z = dep[ys, xs]
            ok = np.isfinite(z) & (z > 0) & (z < 200)
            ys, xs, z = ys[ok], xs[ok], z[ok]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            P = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], 1)
            colors = np.tile([0.7, 0.7, 0.7], (len(P), 1))
            self.clouds[im['sid']] = Cloud(im['sid'], P, colors, K, W, H, dep)
        self.ref = ref
        self._history = []

    def set_reference(self, sid):
        self.ref = sid

    def _snap(self):
        self._history.append({sid: (c.R.copy(), c.t.copy(), c.s)
                              for sid, c in self.clouds.items()})

    def overlap_stats(self, sid, radius=0.35):
        from scipy.spatial import cKDTree
        others = [c.transformed() for s, c in self.clouds.items() if s != sid]
        A = np.concatenate(others, 0)
        d, _ = cKDTree(A).query(self.clouds[sid].transformed(), k=1)
        return float((d < radius).mean()), float(np.median(d))

    def residual(self, sid, radius=0.35):
        return float('nan')


def reset(wb, t, sid):
    wb.set_reference(t['ref'])
    M = np.asarray(t['D_injected'][sid], float)
    sc = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
    c = wb.clouds[sid]
    c.R, c.t, c.s = M[:3, :3] / sc, M[:3, 3].copy(), sc
    wb._history = []
    return wb


def set_absolute(wb, sid, R, t, s=1.0):
    wb._snap()
    c = wb.clouds[sid]
    c.R, c.t, c.s = np.asarray(R, float), np.asarray(t, float), float(s)


def img_rgb(scene, sid):
    p = AG / scene / (sid + '.png')
    im = cv2.imread(str(p))
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def gray_quarter(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)


def backproject(c, uv_full):
    H, W = c.depth.shape
    u = np.clip(np.round(uv_full[:, 0]).astype(int), 0, W - 1)
    v = np.clip(np.round(uv_full[:, 1]).astype(int), 0, H - 1)
    z = c.depth[v, u].astype(float)
    ok = np.isfinite(z) & (z > 0.1) & (z < 200)
    fx, fy, cx, cy = c.K[0, 0], c.K[1, 1], c.K[0, 2], c.K[1, 2]
    P = np.full((len(uv_full), 3), np.nan)
    P[ok, 0] = (uv_full[ok, 0] - cx) / fx * z[ok]
    P[ok, 1] = (uv_full[ok, 1] - cy) / fy * z[ok]
    P[ok, 2] = z[ok]
    return P, ok


def sift_correspondences(scene, ref_sid, sid, cr, cm):
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    ir, im = img_rgb(scene, ref_sid), img_rgb(scene, sid)
    if ir is None or im is None:
        return None
    k1, d1 = sift.detectAndCompute(gray_quarter(ir), None)
    k2, d2 = sift.detectAndCompute(gray_quarter(im), None)
    if d1 is None or d2 is None:
        return None
    m = bf.knnMatch(d1, d2, k=2)
    good = [a for a, b in m if a.distance < RATIO * b.distance]
    m2 = bf.knnMatch(d2, d1, k=2)
    back = {a.queryIdx: a.trainIdx for a, b in m2 if a.distance < RATIO * b.distance}
    good = [a for a in good if back.get(a.trainIdx) == a.queryIdx]
    if len(good) < 6:
        return {'uv_r': np.zeros((0, 2)), 'uv_m': np.zeros((0, 2)),
                'Pr': np.zeros((0, 3)), 'Pm': np.zeros((0, 3)),
                'n_kp': [len(k1), len(k2)], 'n_match': len(good)}
    uv_r = np.array([k1[a.queryIdx].pt for a in good]) / SCALE
    uv_m = np.array([k2[a.trainIdx].pt for a in good]) / SCALE
    Pr, okr = backproject(cr, uv_r)
    Pm, okm = backproject(cm, uv_m)
    ok = okr & okm
    return {'uv_r': uv_r[ok], 'uv_m': uv_m[ok], 'Pr': Pr[ok], 'Pm': Pm[ok],
            'n_kp': [len(k1), len(k2)], 'n_match': len(good)}


def sky_mask(scene, ref_sid, sid, uv_r, uv_m):
    ir, im = img_rgb(scene, ref_sid), img_rgb(scene, sid)

    def at(img, uv):
        H, W = img.shape[:2]
        u = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
        v = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)
        c = img[v, u].astype(float)
        return (c[:, 2] > c[:, 0] + 25) & (c[:, 2] > 110) & (c[:, 1] > c[:, 0])
    return ~(at(ir, uv_r) | at(im, uv_m))


def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    Hm = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cb - R @ ca


def ransac_rigid(Pm, Pr, iters=3000, tol=0.12, seed=0):
    rng = np.random.default_rng(seed)
    n = len(Pm)
    if n < 4:
        return None
    best_k, best_idx = -1, None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            R, t = kabsch(Pm[idx], Pr[idx])
        except np.linalg.LinAlgError:
            continue
        r = np.linalg.norm(Pm @ R.T + t - Pr, axis=1)
        k = int((r < tol).sum())
        if k > best_k:
            best_k, best_idx = k, np.where(r < tol)[0]
    if best_idx is None or len(best_idx) < 4:
        return None
    R, t = kabsch(Pm[best_idx], Pr[best_idx])
    return {'R': R, 't': t, 'inliers': int(len(best_idx)), 'n': n}


def rodrigues(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(w) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def correction(Pr, Pm, R0, iters=8):
    w = np.zeros(3)
    t = np.median(Pr - Pm @ R0.T, axis=0)
    for _ in range(iters):
        q = Pm @ R0.T
        A = np.zeros((3 * len(q), 6))
        b = (Pr - q).ravel()
        for i, (qx, qy, qz) in enumerate(q):
            A[3 * i:3 * i + 3, 0:3] = [[0, qz, -qy], [-qz, 0, qx], [qy, -qx, 0]]
            A[3 * i:3 * i + 3, 3:6] = np.eye(3)
        res0 = np.linalg.norm((Pr - q) - (np.cross(w, q) + t), axis=1)
        sc = max(float(np.median(res0)), 1e-3)
        wts = 1.0 / (1.0 + (res0 / (2.0 * sc)) ** 2)
        sol, *_ = np.linalg.lstsq(A * np.repeat(wts, 3)[:, None],
                                  b * np.repeat(wts, 3), rcond=None)
        w, t = sol[:3], sol[3:]
    return rodrigues(w) @ R0, t, float(np.degrees(np.linalg.norm(w)))


def reproj_refine(Pr, Pm, R0, t0, uv, K, max_nfev=200):
    from scipy.optimize import least_squares
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def resid(p):
        R = rodrigues(p[:3]) @ R0
        q = Pm @ R.T + p[3:6]
        z = np.clip(q[:, 2], 1e-3, None)
        return np.concatenate([fx * q[:, 0] / z + cx - uv[:, 0],
                               fy * q[:, 1] / z + cy - uv[:, 1]])

    x0 = np.concatenate([np.zeros(3), t0])
    sol = least_squares(resid, x0, loss='soft_l1', f_scale=8.0, max_nfev=max_nfev)
    return rodrigues(sol.x[:3]) @ R0, sol.x[3:6], float(np.sqrt(np.mean(resid(sol.x) ** 2)))


def trim_translation(Pr, Pm, R):
    ti = Pr - Pm @ R.T
    med = np.median(ti, 0)
    res = np.linalg.norm(ti - med, axis=1)
    keep = res < max(0.15, np.quantile(res, 0.6))
    return (np.median(ti[keep], 0) if keep.sum() >= 3 else med), float(np.median(
        np.linalg.norm(ti[keep] - med, axis=1))) if keep.sum() else float('nan')


def run_case(scene, tag, sid):
    t = TASK[scene][tag]
    ref = t['ref']
    wb = TwoView(scene, ref, sid)
    rows = []
    cache = {}

    def take(method, fn, meta=None):
        reset(wb, t, sid)
        t0 = time.time()
        info = fn() or {}
        ms = (time.time() - t0) * 1000.0
        s = score(wb, t, sid)
        M = np.asarray(s['matrix_est'])
        lat = float(np.linalg.norm((M[:3, 3] - np.asarray(s['matrix_gt'])[:3, 3])[:2]))
        rows.append({'scene': scene, 'tag': tag, 'sid': sid, 'ref': ref,
                     'method': method, 'rot_err': s['rot_err'],
                     'lat_err': lat, 'trans_err': s['trans_err'],
                     'scale_err': s['scale_err'], 'overlap': s['overlap'],
                     'success': bool(s['rot_err'] < 2 and lat < 0.5),
                     'runtime_ms': ms, 'n_sift': cache.get('n_sift'),
                     'meta': meta or {}, 'matrix_est': s['matrix_est'],
                     'matrix_gt': s['matrix_gt'],
                     'D_injected': t['D_injected'][sid]})

    take('identity', lambda: None)

    def a_icp():
        import open3d as o3d
        o3d.utility.random.seed(0)
        from protocol import method_icp
        return method_icp(wb, t, sid)
    take('A_fpfh_icp', a_icp)
    MA = np.asarray(rows[-1]['matrix_est'], float)
    scA = float(np.cbrt(abs(np.linalg.det(MA[:3, :3]))))
    RA = MA[:3, :3] / scA

    cr, cm = wb.clouds[ref], wb.clouds[sid]
    sift = sift_correspondences(scene, ref, sid, cr, cm)
    cache['n_sift'] = sift['n_match'] if sift else 0
    if sift and len(sift['Pr']) >= 6:
        Pr, Pm = sift['Pr'], sift['Pm']
        uv_r, uv_m = sift['uv_r'], sift['uv_m']
        sky = sky_mask(scene, ref, sid, uv_r, uv_m)

        def b_rigid():
            fit = ransac_rigid(Pm, Pr)
            if fit is None:
                return {'error': 'no consensus'}
            set_absolute(wb, sid, fit['R'], fit['t'])
            return {'inliers': fit['inliers'], 'n': fit['n']}
        take('B_sift_rigid', b_rigid)

        def c_rigid_sky():
            fit = ransac_rigid(Pm[sky], Pr[sky]) if sky.sum() >= 4 else None
            if fit is None:
                return {'error': 'no consensus'}
            set_absolute(wb, sid, fit['R'], fit['t'])
            return {'inliers': fit['inliers'], 'n': fit['n'], 'n_sky_kept': int(sky.sum())}
        take('C_sift_rigid_sky', c_rigid_sky)

        def d_rot_t():
            tt, spread = trim_translation(Pr, Pm, RA)
            set_absolute(wb, sid, RA, tt)
            return {'spread': spread, 'n': int(len(Pr))}
        take('D_rotA_sift_t', d_rot_t)

        def e_rot_t_sky():
            tt, spread = trim_translation(Pr[sky], Pm[sky], RA)
            set_absolute(wb, sid, RA, tt)
            return {'spread': spread, 'n': int(sky.sum())}
        take('E_rotA_sift_t_sky', e_rot_t_sky)

        def f_corr():
            tt, _ = trim_translation(Pr[sky], Pm[sky], RA)
            R1, t1, deg = correction(Pr[sky], Pm[sky], RA)
            sp0 = float(np.median(np.linalg.norm(Pr[sky] - (Pm[sky] @ RA.T + tt), axis=1)))
            sp1 = float(np.median(np.linalg.norm(Pr[sky] - (Pm[sky] @ R1.T + t1), axis=1)))
            if sp1 <= sp0:
                set_absolute(wb, sid, R1, t1)
                return {'accepted': True, 'correction_deg': deg, 'spread': [sp0, sp1]}
            set_absolute(wb, sid, RA, tt)
            return {'accepted': False, 'correction_deg': deg, 'spread': [sp0, sp1]}
        take('F_rotA_sift_corr', f_corr)

        def g_reproj():
            tt, _ = trim_translation(Pr[sky], Pm[sky], RA)
            R1, t1, rms = reproj_refine(Pr[sky], Pm[sky], RA, tt, uv_r[sky], cr.K)
            set_absolute(wb, sid, R1, t1)
            return {'rms_px': rms, 'n': int(sky.sum())}
        take('G_rotA_sift_reproj', g_reproj)

        def h_rigid_reproj():
            fit = ransac_rigid(Pm[sky], Pr[sky]) if sky.sum() >= 4 else None
            if fit is None:
                return {'error': 'no consensus'}
            R1, t1, rms = reproj_refine(Pr[sky], Pm[sky], fit['R'], fit['t'],
                                        uv_r[sky], cr.K)
            set_absolute(wb, sid, R1, t1)
            return {'rms_px': rms, 'n': int(sky.sum()), 'ransac_inliers': fit['inliers']}
        take('H_sift_rigid_reproj', h_rigid_reproj)
    else:
        for m in ('B_sift_rigid', 'C_sift_rigid_sky', 'D_rotA_sift_t',
                  'E_rotA_sift_t_sky', 'F_rotA_sift_corr',
                  'G_rotA_sift_reproj', 'H_sift_rigid_reproj'):
            rows.append({'scene': scene, 'tag': tag, 'sid': sid, 'ref': ref,
                         'method': m, 'error': 'too_few_sift_matches', 'n_sift': 0})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    cases = [(scene, tag, sid, t)
             for scene, subs in TASK.items()
             for tag, t in subs.items()
             for sid in t['ids'] if sid != t['ref']]
    for i, (scene, tag, sid, t) in enumerate(cases, 1):
        r = run_case(scene, tag, sid)
        rows.extend(r)
        s = next(x for x in r if x['method'] == 'G_rotA_sift_reproj')
        a = next(x for x in r if x['method'] == 'A_fpfh_icp')
        print('[%2d/%d] %-14s %-3s %-12s  A %6.1f deg / %6.2f m   G %6.1f deg / %6.2f m'
              % (i, len(cases), scene, tag, sid, a.get('rot_err', np.nan),
                 a.get('lat_err', np.nan), s.get('rot_err', np.nan),
                 s.get('lat_err', np.nan)), flush=True)
    p = OUT / 'eval' / 'baselines.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=1))
    print('\nwrote %s  (%d rows, %.1f s)' % (p, len(rows), time.time() - t0))

    methods = sorted(set(x['method'] for x in rows))
    print('%-22s %3s %7s %6s %7s %6s %6s' %
          ('method', 'n', 'rot_med', '<2', 'lat_med', '<0.5', 'succ'))
    for m in methods:
        rr = [x for x in rows if x['method'] == m and 'rot_err' in x]
        rot = np.array([x['rot_err'] for x in rr])
        lat = np.array([x['lat_err'] for x in rr])
        print('%-22s %3d %7.2f %6d %7.2f %6d %6d' %
              (m, len(rr), np.median(rot), int((rot < 2).sum()),
               np.median(lat), int((lat < 0.5).sum()), int((rot < 2).sum() & 0) +
               int(((rot < 2) & (lat < 0.5)).sum())))


if __name__ == '__main__':
    main()
