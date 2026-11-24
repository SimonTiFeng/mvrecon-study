"""Rotation from matched plane normals, aimed at the case dominant frames cannot fix.

Dominant-frame alignment fails on facade: the oracle over all 24 alignments is no better than the
coarse rotation, because the two views' dominant planes are not the same physical surfaces. But
the views may still share *some* planes. Plane normals from thousands of points are the most
accurate orientation primitive available, and two non-parallel planes determine rotation fully.

Method:
  1. segment up to 8 planes per cloud by iterative RANSAC
  2. candidate correspondences: plane pairs whose normals agree within 30 deg under the coarse
     rotation (sign-free, since a fitted normal has no inherent direction)
  3. RANSAC over pairs of correspondences: each pair of non-parallel normals fixes a rotation;
     score by how many other correspondences it explains within 10 deg
  4. report the best rotation, and the translation solved from SIFT correspondences with it

Selection is entirely ground-truth free. The diagnostic value is in the per-plane angular errors:
if no plane pair agrees under R_true, the information simply is not there and no matcher can help.
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

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval'))
sys.path.insert(0, str(ROOT / 'workbench'))
from protocol import setup, rotdeg  # noqa: E402
from wb import Workbench  # noqa: E402

_spec = importlib.util.spec_from_file_location('si', ROOT / 'eval' / 'sift_identity.py')
si = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(si)
_spec2 = importlib.util.spec_from_file_location('fpi', ROOT / 'eval' / 'frame_plus_identity.py')
fpi = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(fpi)

TASK = json.loads((ROOT / 'eval_data/task.json').read_text())
BASE = json.loads((ROOT / 'out' / 'baseline_rescored.json').read_text())
RNG = np.random.default_rng(0)


def seg_planes(pts, max_planes=8, dist=0.15):
    planes = []
    P = pts.copy()
    for _ in range(max_planes):
        if len(P) < 300:
            break
        try:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(P)
            model, inl = pc.segment_plane(distance_threshold=dist, ransac_n=3,
                                          num_iterations=800)
        except Exception:                               # noqa: BLE001
            break
        inl = np.asarray(inl)
        if len(inl) < 0.05 * len(P) or len(inl) < 400:
            break
        n = np.asarray(model[:3], float)
        n /= max(np.linalg.norm(n), 1e-12)
        planes.append({'n': n, 'n_inl': int(len(inl)), 'd': float(model[3])})
        mask = np.ones(len(P), bool)
        mask[inl] = False
        # also drop points just outside the threshold so the same plane is not refound
        resid = np.abs(P @ np.asarray(model[:3]) + model[3])
        P = P[mask & (resid > 1.5 * dist)]
    return planes


def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R


def normals_rotation(pairs, R_coarse):
    """pairs: list of (n_ref, n_mov). Sign-resolution by the coarse rotation, then Kabsch."""
    A, B = [], []
    for nr, nm in pairs:
        s = 1.0 if (R_coarse @ nm) @ nr >= 0 else -1.0
        A.append(s * nm)
        B.append(nr)
    return kabsch(np.array(A), np.array(B))


def main():
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    rows = []
    for x in BASE:
        if x['method'] != 'A_icp':
            continue
        if x['score_corrected']['rot_err_deg'] >= 20:
            continue
        scene, tag, sid = x['scene'], x['tag'], x['sid']
        wb = Workbench(ROOT / 'agent_data' / scene, stride=32)
        setup(wb, TASK[scene][tag], TASK[scene][tag]['ref'])
        ref_sid = wb.ref
        ref_c, mov_c = wb.clouds[ref_sid], wb.clouds[sid]
        M = np.asarray(x['matrix_est'], float)
        R_coarse = M[:3, :3] / max(np.cbrt(abs(np.linalg.det(M[:3, :3]))), 1e-12)
        T = np.asarray(x['matrix_gt'], float)
        R_true, t_true = T[:3, :3], T[:3, 3]

        pr_ref = seg_planes(ref_c.points)
        pr_mov = seg_planes(mov_c.points)
        if len(pr_ref) < 2 or len(pr_mov) < 1:
            print('%-14s%-4s%-12s too few planes' % (scene, tag, sid), flush=True)
            continue

        # how well do planes agree under the TRUE rotation? (diagnostic, not used for choice)
        best_true = 90.0
        for a in pr_ref:
            for b in pr_mov:
                ang = np.degrees(np.arccos(np.clip(abs((R_true @ b['n']) @ a['n']), 0, 1)))
                best_true = min(best_true, ang)
        n_agree_true = sum(
            1 for a in pr_ref for b in pr_mov
            if np.degrees(np.arccos(np.clip(abs((R_true @ b['n']) @ a['n']), 0, 1))) < 5.0)

        cand = []
        for a in pr_ref:
            for b in pr_mov:
                ang = np.degrees(np.arccos(np.clip(abs((R_coarse @ b['n']) @ a['n']), 0, 1)))
                if ang < 30.0:
                    cand.append((a['n'], b['n'], min(a['n_inl'], b['n_inl'])))
        if len(cand) < 2:
            print('%-14s%-4s%-12s no usable normal pairs (best vs true %.1f deg)'
                  % (scene, tag, sid, best_true), flush=True)
            continue

        def assign(R, tol=10.0):
            """One-to-one greedy assignment of reference planes to moving planes.

            Counting pairs independently lets one reference plane explain several near-parallel
            moving planes, which inflates support for wrong rotations. Each plane may be used once.
            """
            A = np.array([[np.degrees(np.arccos(np.clip(abs((R @ m['n']) @ r['n']), 0, 1)))
                           for m in pr_mov] for r in pr_ref])
            used_r, used_m, score, wsum = set(), set(), 0, 0.0
            while True:
                sub = A.copy()
                for i in used_r:
                    sub[i, :] = 1e9
                for j in used_m:
                    sub[:, j] = 1e9
                i, j = np.unravel_index(np.argmin(sub), sub.shape)
                if sub[i, j] > tol:
                    break
                used_r.add(int(i))
                used_m.add(int(j))
                score += 1
                wsum += min(pr_ref[i]['n_inl'], pr_mov[j]['n_inl'])
            return score, wsum

        best = None
        idxs = list(range(len(cand)))
        trials = min(600, 60 * len(idxs))
        for _ in range(trials):
            i, j = RNG.choice(idxs, 2, replace=False)
            if abs(cand[i][0] @ cand[i][1]) > 0.94:      # need a non-parallel pair
                continue
            R = normals_rotation([(cand[i][0], cand[i][1]), (cand[j][0], cand[j][1])], R_coarse)
            # a rotation more than 25 deg from the coarse estimate is a different hypothesis
            # altogether (typically a 180 deg flip about a matched normal, which plane normals
            # cannot distinguish); exclude it rather than let spurious support pick it
            if rotdeg(R @ R_coarse.T) > 25.0:
                continue
            sc, w = assign(R)
            key = (sc, w, -rotdeg(R @ R_coarse.T))
            if best is None or key > best[0]:
                best = (key, R)
        if best is None:
            continue
        R_pl = best[1]

        # translation from appearance correspondences, so only rotation is being judged
        pair = fpi.sift_pairs(scene, ref_sid, sid, ref_c, mov_c, sift, bf)
        if pair is None:
            continue
        Pr, Pm = pair
        ti = Pr - Pm @ R_pl.T
        t_pl = np.median(ti, axis=0)

        rows.append({'scene': scene, 'tag': tag, 'sid': sid,
                     'n_ref_planes': len(pr_ref), 'n_mov_planes': len(pr_mov),
                     'best_plane_agree_true': float(best_true), 'n_agree_true': int(n_agree_true),
                     'rot_coarse': float(rotdeg(R_coarse @ R_true.T)),
                     'rot_plane': float(rotdeg(R_pl @ R_true.T)),
                     'lat_coarse': float(np.linalg.norm((M[:3, 3] - t_true)[:2])),
                     'lat_plane': float(np.linalg.norm((t_pl - t_true)[:2])),
                     'inliers': int(best[0][0])})
        r = rows[-1]
        print('%-14s%-4s%-12s planes %d/%d agree@true %2d (best %.1f deg) | rot %5.2f -> %5.2f | lat %5.2f -> %5.2f | inl %d'
              % (scene, tag, sid, r['n_ref_planes'], r['n_mov_planes'], r['n_agree_true'],
                 r['best_plane_agree_true'], r['rot_coarse'], r['rot_plane'],
                 r['lat_coarse'], r['lat_plane'], r['inliers']), flush=True)

    p = ROOT / 'out' / 'plane_match_rotation.json'
    p.write_text(json.dumps(rows, indent=1))
    if rows:
        g = lambda k: np.array([r[k] for r in rows])
        print('\nn=%d' % len(rows))
        print('plane pairs agreeing under the TRUE rotation: median %d, never agreeing: %d/%d'
              % (np.median(g('n_agree_true')), int((g('n_agree_true') == 0).sum()), len(rows)))
        print('rotation median %5.2f -> %5.2f deg   improved %d  worsened %d'
              % (np.median(g('rot_coarse')), np.median(g('rot_plane')),
                 int((g('rot_plane') < g('rot_coarse')).sum()),
                 int((g('rot_plane') > g('rot_coarse') + 0.05).sum())))
        print('translation median %5.2f -> %5.2f m  (under 0.5 m %d -> %d)'
              % (np.median(g('lat_coarse')), np.median(g('lat_plane')),
                 int((g('lat_coarse') < 0.5).sum()), int((g('lat_plane') < 0.5).sum())))
    print('wrote', p)


if __name__ == '__main__':
    main()
