"""Does minimising REPROJECTION error beat 3D-3D fitting on these correspondences?

Every solve so far fits in 3D: Kabsch, or the small-rotation linearisation, minimising distances
between matched 3D points. But the measurements that produced those points live in the image, and
their noise is roughly isotropic in pixels, not in metres. Minimising reprojection error therefore
weights the correspondences the way the sensor actually errs -- distant points contribute less
metres per pixel, near points more -- which is the correct treatment and a different estimator.

Tested on the same 25 targets as everything else, with the same sky-masked SIFT correspondences:
  init     baseline rotation + median translation from the matches
  refine   robust (soft-L1) least squares over 6 parameters: a rotation vector and a translation,
           minimising the reprojection error of the moving points in the REFERENCE image
Reported against the baseline and against the 3D-3D correction measured earlier.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import cv2
import numpy as np
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
BASE = json.loads((ROOT / 'out' / 'baseline_rescored.json').read_text())


def rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else 'sift'
    if kind == 'sift':
        det, norm = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02), cv2.NORM_L2
    elif kind == 'akaze':
        det, norm = cv2.AKAZE_create(threshold=0.0008), cv2.NORM_HAMMING
    else:
        det, norm = cv2.ORB_create(nfeatures=4000, fastThreshold=12), cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm)
    rows = []
    for x in BASE:
        if x['method'] != 'A_icp' or x['score_corrected']['rot_err_deg'] >= 20:
            continue
        scene, tag, sid = x['scene'], x['tag'], x['sid']
        wb = Workbench(ROOT / 'agent_data' / scene, stride=32)
        setup(wb, TASK[scene][tag], TASK[scene][tag]['ref'])
        ref_c, mov_c = wb.clouds[wb.ref], wb.clouds[sid]
        M = np.asarray(x['matrix_est'], float)
        R0 = M[:3, :3] / max(np.cbrt(abs(np.linalg.det(M[:3, :3]))), 1e-12)
        T = np.asarray(x['matrix_gt'], float)
        R_true, t_true = T[:3, :3], T[:3, 3]

        gi, gj = si.load_rgb(scene, wb.ref), si.load_rgb(scene, sid)
        if gi is None or gj is None:
            continue
        k1, d1 = det.detectAndCompute(gi, None)
        k2, d2 = det.detectAndCompute(gj, None)
        if d1 is None or d2 is None:
            continue
        m = bf.knnMatch(d1, d2, k=2)
        good = [a for a, b in m if a.distance < 0.8 * b.distance]
        m2 = bf.knnMatch(d2, d1, k=2)
        back = {a.queryIdx: a.trainIdx for a, b in m2 if a.distance < 0.8 * b.distance}
        good = [a for a in good if back.get(a.trainIdx) == a.queryIdx]
        if len(good) < 20:
            continue
        uv_r = np.array([k1[a.queryIdx].pt for a in good]) / si.SCALE
        uv_m = np.array([k2[a.trainIdx].pt for a in good]) / si.SCALE
        Pr, okr = si.backproject(uv_r, ref_c.depth, ref_c.K)
        Pm, okm = si.backproject(uv_m, mov_c.depth, mov_c.K)

        cr = cv2.cvtColor(cv2.imread(str(ROOT / 'agent_data' / scene / (wb.ref + '.png'))),
                          cv2.COLOR_BGR2RGB).astype(float)
        cm = cv2.cvtColor(cv2.imread(str(ROOT / 'agent_data' / scene / (sid + '.png'))),
                          cv2.COLOR_BGR2RGB).astype(float)
        ur = np.clip(np.round(uv_r[:, 0]).astype(int), 0, cr.shape[1] - 1)
        vr = np.clip(np.round(uv_r[:, 1]).astype(int), 0, cr.shape[0] - 1)
        um = np.clip(np.round(uv_m[:, 0]).astype(int), 0, cm.shape[1] - 1)
        vm = np.clip(np.round(uv_m[:, 1]).astype(int), 0, cm.shape[0] - 1)
        prc, pmc = cr[vr, ur], cm[vm, um]
        sky = (((prc[:, 2] > prc[:, 0] + 25) & (prc[:, 2] > 110) & (prc[:, 1] > prc[:, 0])) |
               ((pmc[:, 2] > pmc[:, 0] + 25) & (pmc[:, 2] > 110) & (pmc[:, 1] > pmc[:, 0])))
        keep = okr & okm & ~sky
        if keep.sum() < 20:
            continue
        Pm_k = Pm[keep]
        uv_r_k = uv_r[keep]
        t0 = np.median(Pr[keep] - Pm_k @ R0.T, axis=0)

        K = ref_c.K
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        use_scale = '--scale' in sys.argv

        def resid(p):
            R = rodrigues(p[:3]) @ R0
            sc = float(np.exp(p[6])) if use_scale else 1.0
            q = sc * (Pm_k @ R.T) + p[3:6]
            z = np.clip(q[:, 2], 1e-3, None)
            u = fx * q[:, 0] / z + cx
            v = fy * q[:, 1] / z + cy
            return np.concatenate([u - uv_r_k[:, 0], v - uv_r_k[:, 1]])

        x0 = np.concatenate([np.zeros(3), t0, [0.0]]) if use_scale else             np.concatenate([np.zeros(3), t0])
        sol = least_squares(resid, x0, loss='soft_l1', f_scale=8.0, max_nfev=200)
        R1 = rodrigues(sol.x[:3]) @ R0
        t1 = sol.x[3:6]
        s_est = float(np.exp(sol.x[6])) if use_scale else 1.0

        rows.append({'scene': scene, 'tag': tag, 'sid': sid, 'n': int(keep.sum()),
                     'rot_base': float(rotdeg(R0 @ R_true.T)),
                     'lat_base': float(np.linalg.norm((M[:3, 3] - t_true)[:2])),
                     'rot_rep': float(rotdeg(R1 @ R_true.T)),
                     'lat_rep': float(np.linalg.norm((t1 - t_true)[:2])),
                     'rms_px': float(np.sqrt(np.mean(resid(sol.x) ** 2))),
                     'scale': s_est})
        r = rows[-1]
        print('%-14s%-4s%-12s n=%3d rms %.1f px | rot %5.2f -> %5.2f | lat %5.2f -> %5.2f'
              % (scene, tag, sid, r['n'], r['rms_px'], r['rot_base'], r['rot_rep'],
                 r['lat_base'], r['lat_rep']), flush=True)

    tag_out = kind + ('_scale' if '--scale' in sys.argv else '')
    p = ROOT / 'out' / ('reproj_refine_%s.json' % tag_out)
    p.write_text(json.dumps(rows, indent=1))
    if rows:
        g = lambda k: np.array([r[k] for r in rows])
        print('\nn=%d' % len(rows))
        print('rotation median %.2f -> %.2f°   <2°: %d -> %d'
              % (np.median(g('rot_base')), np.median(g('rot_rep')),
                 int((g('rot_base') < 2).sum()), int((g('rot_rep') < 2).sum())))
        print('lateral  median %.2f -> %.2f m   <0.5 m: %d -> %d'
              % (np.median(g('lat_base')), np.median(g('lat_rep')),
                 int((g('lat_base') < 0.5).sum()), int((g('lat_rep') < 0.5).sum())))
        if use_scale:
            print('median estimated scale %.4f' % np.median(g('scale')))
        print('improved %d  worsened %d'
              % (int((g('lat_rep') < g('lat_base') - 0.02).sum()),
                 int((g('lat_rep') > g('lat_base') + 0.02).sum())))
    print('wrote', p)


if __name__ == '__main__':
    main()
