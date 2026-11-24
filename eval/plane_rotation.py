"""Precise rotation from dominant surface frames, since rotation is the whole error.

rotation_leak.py showed translation is exact once rotation is right, and that the residual is
almost entirely rotation error (corr +0.78). Feature-based coarse registration gives ~5 deg median
(9.2 deg worst). At 11-20 m levers that is 1-2 m of apparent translation.

Large planes are the most accurate orientation signal available: a normal fitted to thousands of
points on a wall is far better determined than a rotation implied by sparse feature matches.
Architectural and indoor scenes are dominated by a few near-orthogonal directions, so:

  1. cluster each cloud's normals to find its dominant axes (support-weighted, up to 3)
  2. orthogonalise them into a frame per cloud
  3. enumerate the 24 proper sign/permutation alignments between the two frames
  4. pick the one closest to the coarse feature rotation -- which resolves the ambiguity -- and
     take it as the rotation estimate

Reported per target: rotation error of the coarse estimate vs the frame estimate, and the
translation error each produces with the SAME oracle correspondence, so only rotation differs.
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np
import open3d as o3d

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval'))
sys.path.insert(0, str(ROOT / 'workbench'))
from protocol import setup, rotdeg  # noqa: E402
from wb import Workbench  # noqa: E402

TASK = json.loads((ROOT / 'eval_data/task.json').read_text())
BASE = json.loads((ROOT / 'out' / 'baseline_rescored.json').read_text())
RNG = np.random.default_rng(4)
TOL = np.deg2rad(12.0)
MAX_PTS = 4000


def normals_of(pts):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.6, max_nn=30))
    return np.asarray(pc.normals)


def dominant_axes(pts, n_max=3, tol=TOL):
    """Support-weighted search for the directions most normals align with.

    Normals must be estimated on the FULL cloud: the clouds are already strided, so subsampling
    before the normal fit leaves too few neighbours inside the search radius and the normals
    become noise. Subsample the normals afterwards, for the clustering only.
    """
    n = normals_of(pts)
    keep = np.isfinite(n).all(1)
    n = n[keep]
    if len(n) > MAX_PTS:
        n = n[RNG.choice(len(n), MAX_PTS, replace=False)]
    alive = np.ones(len(n), bool)
    axes = []
    for _ in range(n_max):
        idx = np.where(alive)[0]
        if len(idx) < 40:
            break
        A = n[idx]
        # candidate directions are the normals themselves; score = support within tol
        D = np.abs(A @ A.T)
        support = (D > np.cos(tol)).sum(1)
        best = int(np.argmax(support))
        if support[best] < max(40, 0.05 * len(idx)):
            break
        d = A[best] / max(np.linalg.norm(A[best]), 1e-12)
        keep = np.abs(A @ d) > np.cos(tol)
        if keep.sum() == 0:
            break
        # refine: mean direction of the cluster, sign-aligned
        C = A[keep] * np.sign(A[keep] @ d)[:, None]
        d = C.mean(0)
        d /= max(np.linalg.norm(d), 1e-12)
        axes.append(d)
        # remove this cluster from the pool, using the global mask
        glob = idx[keep]
        alive[glob] = False
        alive[idx[~keep]] = True
    while len(axes) < 3:
        if len(axes) == 2:
            axes.append(np.cross(axes[0], axes[1]))
        elif len(axes) == 1:
            tmp = np.array([1.0, 0, 0]) if abs(axes[0][0]) < 0.9 else np.array([0, 1.0, 0])
            axes.append(np.cross(axes[0], tmp))
        else:
            axes = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
            break
    # orthogonalise
    e1 = axes[0] / np.linalg.norm(axes[0])
    a2 = axes[1] - (axes[1] @ e1) * e1
    if np.linalg.norm(a2) < 1e-6:
        a2 = np.cross(e1, np.array([0, 1.0, 0]))
    e2 = a2 / np.linalg.norm(a2)
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=1)


def best_frame_rotation(F_ref, F_mov, R_coarse):
    """The proper sign/permutation alignment of the two frames closest to the coarse rotation."""
    best = (1e9, None)
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            P = np.zeros((3, 3))
            for j in range(3):
                P[perm[j], j] = signs[j]
            if np.linalg.det(P) < 0.5:
                continue
            R = F_ref @ P @ F_mov.T
            if np.linalg.det(R) < 0:
                continue
            ang = rotdeg(R @ R_coarse.T)
            if ang < best[0]:
                best = (ang, R)
    return best[1]


def main():
    rows = []
    for x in BASE:
        if x['method'] != 'A_icp':
            continue
        s = x['score_corrected']
        if s['rot_err_deg'] >= 20:
            continue
        scene, tag, sid = x['scene'], x['tag'], x['sid']
        wb = Workbench(ROOT / 'agent_data' / scene, stride=32)
        setup(wb, TASK[scene][tag], TASK[scene][tag]['ref'])
        ref_c, mov_c = wb.clouds[wb.ref], wb.clouds[sid]
        M = np.asarray(x['matrix_est'], float)
        R_coarse = M[:3, :3] / max(np.cbrt(abs(np.linalg.det(M[:3, :3]))), 1e-12)
        T = np.asarray(x['matrix_gt'], float)
        R_true, t_true = T[:3, :3], T[:3, 3]

        # own-frame points: as_o3d() returns transformed(), i.e. already at the injected D
        p_ref_all = ref_c.points
        p_mov_all = mov_c.points
        try:
            F_ref = dominant_axes(p_ref_all)
            F_mov = dominant_axes(p_mov_all)
            R_frame = best_frame_rotation(F_ref, F_mov, R_coarse)
        except Exception as exc:                       # noqa: BLE001
            print('%-14s%-4s%-12s frame failed: %s' % (scene, tag, sid, exc), flush=True)
            continue

        # one oracle correspondence, so the two rows differ only in rotation
        p_m = (p_ref_all - t_true) @ R_true
        z = p_m[:, 2]
        okz = z > 0.2
        fx, fy, cx, cy = mov_c.K[0, 0], mov_c.K[1, 1], mov_c.K[0, 2], mov_c.K[1, 2]
        H, W = mov_c.depth.shape
        u = np.full(len(p_ref_all), np.nan)
        v = np.full(len(p_ref_all), np.nan)
        u[okz] = fx * p_m[okz, 0] / z[okz] + cx
        v[okz] = fy * p_m[okz, 1] / z[okz] + cy
        frame = okz & np.isfinite(u) & np.isfinite(v) & (u >= 2) & (u < W - 2) & (v >= 2) & (v < H - 2)
        cand = np.where(frame)[0]
        if len(cand) == 0:
            continue
        iu = np.clip(np.round(u[cand]).astype(int), 1, W - 2)
        iv = np.clip(np.round(v[cand]).astype(int), 1, H - 2)
        zm = mov_c.depth[iv, iu].astype(float)
        good = np.isfinite(zm) & (zm > 0)
        vis = cand[good & (np.abs(zm - z[cand]) / np.maximum(z[cand], 1e-6) < 0.05)]
        if len(vis) == 0:
            continue
        i = int(vis[len(vis) // 2])
        pr, pm = p_ref_all[i], p_m[i]

        def lat(R):
            return float(np.linalg.norm(((pr - R @ pm) - t_true)[:2]))

        rows.append({'scene': scene, 'tag': tag, 'sid': sid,
                     'rot_coarse': float(rotdeg(R_coarse @ R_true.T)),
                     'rot_frame': float(rotdeg(R_frame @ R_true.T)),
                     'lat_coarse': lat(R_coarse), 'lat_frame': lat(R_frame)})
        r = rows[-1]
        print('%-14s%-4s%-12s rot %5.2f -> %5.2f deg | lat %6.2f -> %6.2f m'
              % (scene, tag, sid, r['rot_coarse'], r['rot_frame'],
                 r['lat_coarse'], r['lat_frame']), flush=True)

    p = ROOT / 'out' / 'plane_rotation.json'
    p.write_text(json.dumps(rows, indent=1))
    if rows:
        rc = np.array([r['rot_coarse'] for r in rows])
        rf = np.array([r['rot_frame'] for r in rows])
        lc = np.array([r['lat_coarse'] for r in rows])
        lf = np.array([r['lat_frame'] for r in rows])
        print('\nn=%d' % len(rows))
        print('rotation  median %5.2f -> %5.2f deg   improved %d/%d'
              % (np.median(rc), np.median(rf), int((rf < rc).sum()), len(rows)))
        print('translation median %5.2f -> %5.2f m  improved %d/%d'
              % (np.median(lc), np.median(lf), int((lf < lc).sum()), len(rows)))
        print('under 0.5 m: coarse %d/%d  frame %d/%d'
              % (int((lc < 0.5).sum()), len(rows), int((lf < 0.5).sum()), len(rows)))
    print('wrote', p)


if __name__ == '__main__':
    main()
