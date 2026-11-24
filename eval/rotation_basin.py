"""Is 1-degree rotation achievable at all, and how wide is the basin that reaches it?

rotation_leak.py established that translation is exact once rotation is right, and that the
residual is rotation error. So the only question left is how precisely rotation can be determined
from these two clouds. That is not a question about any particular front end -- it is a property
of the data, and it has two parts:

  floor  - run point-to-plane ICP STARTING AT THE TRUE ROTATION. Whatever it converges to is the
           precision the geometry supports. If it drifts to 5 deg, the constraint simply is not
           there and no method can do better.
  basin  - run ICP from rotations offset by delta about each axis. The largest delta that still
           converges to the floor is how close an initial guess must be. A narrow basin means a
           coarse guess (from a feature matcher, or from a model looking at pictures) cannot land
           in it and refinement must be incremental; a wide basin means a coarse guess suffices.

Translation is initialised from one oracle correspondence consistent with each starting rotation,
so only rotation is being varied. Reported per target and aggregated.
"""
from __future__ import annotations

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
OFFSETS = [-20, -10, -5, -2, 0, 2, 5, 10, 20]


def axis_rot(axis, deg):
    a = np.zeros(3)
    a[axis] = np.deg2rad(deg)
    return o3d.geometry.get_rotation_matrix_from_xyz(a)


def prep(pts, voxel=0.4):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc = pc.voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.8, max_nn=30))
    return pc


def run_icp(src, dst, R0, t0, thr=0.20, iters=60):
    T0 = np.eye(4)
    T0[:3, :3] = R0
    T0[:3, 3] = t0
    res = o3d.pipelines.registration.registration_icp(
        src, dst, thr, T0,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters))
    return res.transformation[:3, :3], res.transformation[:3, 3]


def main():
    rows = []
    for x in BASE:
        if x['method'] != 'A_icp':
            continue
        if x['score_corrected']['rot_err_deg'] >= 20:
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
        p_ref = ref_c.points
        p_mov_all = mov_c.points
        p_m = (p_ref - t_true) @ R_true
        # one oracle correspondence, and its moving-frame point
        z = p_m[:, 2]
        ok = z > 0.5
        fx, fy, cx, cy = mov_c.K[0, 0], mov_c.K[1, 1], mov_c.K[0, 2], mov_c.K[1, 2]
        H, W = mov_c.depth.shape
        u = np.full(len(p_ref), np.nan)
        v = np.full(len(p_ref), np.nan)
        u[ok] = fx * p_m[ok, 0] / z[ok] + cx
        v[ok] = fy * p_m[ok, 1] / z[ok] + cy
        fr = ok & np.isfinite(u) & np.isfinite(v) & (u >= 2) & (u < W - 2) & (v >= 2) & (v < H - 2)
        cand = np.where(fr)[0]
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
        p_ref_a, p_mov_a = p_ref[i], p_m[i]

        src = prep(p_mov_all)
        dst = prep(p_ref)

        def at(R0):
            t0 = p_ref_a - R0 @ p_mov_a
            R1, t1 = run_icp(src, dst, R0, t0)
            return rotdeg(R1 @ R_true.T), float(np.linalg.norm((t1 - t_true)[:2]))

        floor_rot, floor_lat = at(R_true)
        coarse_rot, coarse_lat = at(R_coarse)
        basin = {}
        for ax in range(3):
            curve = []
            for d in OFFSETS:
                r, _ = at(R_true @ axis_rot(ax, d))
                curve.append((d, r))
            basin['xyz'[ax]] = curve
        # half-width: largest |delta| whose result stays within max(2 deg, 2x floor)
        tol = max(2.0, 2.0 * floor_rot)
        widths = []
        for ax in 'xyz':
            ok_d = [abs(d) for d, r in basin[ax] if r <= tol]
            widths.append(max(ok_d) if ok_d else 0)
        rows.append({'scene': scene, 'tag': tag, 'sid': sid,
                     'floor_rot': floor_rot, 'floor_lat': floor_lat,
                     'coarse_in_rot': x['score_corrected']['rot_err_deg'],
                     'coarse_in_lat': x['score_corrected']['trans_err_m'],
                     'after_rot': coarse_rot, 'after_lat': coarse_lat,
                     'basin': basin, 'basin_width_deg': float(np.mean(widths))})
        r = rows[-1]
        print('%-14s%-4s%-12s floor %5.2f deg | from coarse %5.2f -> %5.2f deg, lat %5.2f -> %5.2f | basin %4.0f deg'
              % (scene, tag, sid, floor_rot, r['coarse_in_rot'], coarse_rot,
                 r['coarse_in_lat'], coarse_lat, r['basin_width_deg']), flush=True)

    p = ROOT / 'out' / 'rotation_basin.json'
    p.write_text(json.dumps(rows, indent=1))
    if rows:
        f = np.array([r['floor_rot'] for r in rows])
        c0 = np.array([r['coarse_in_rot'] for r in rows])
        c1 = np.array([r['after_rot'] for r in rows])
        l0 = np.array([r['coarse_in_lat'] for r in rows])
        l1 = np.array([r['after_lat'] for r in rows])
        bw = np.array([r['basin_width_deg'] for r in rows])
        print('\nn=%d' % len(rows))
        print('achievable rotation floor (ICP from truth): median %.2f deg  (max %.2f)'
              % (np.median(f), f.max()))
        print('from the coarse estimate: %.2f -> %.2f deg   improved %d/%d'
              % (np.median(c0), np.median(c1), int((c1 < c0).sum()), len(rows)))
        print('translation after that refinement: median %.2f -> %.2f m  (under 0.5 m %d/%d -> %d/%d)'
              % (np.median(l0), np.median(l1), int((l0 < 0.5).sum()), len(rows),
                 int((l1 < 0.5).sum()), len(rows)))
        print('basin half-width: median %.0f deg  (min %.0f, max %.0f)'
              % (np.median(bw), bw.min(), bw.max()))
    print('wrote', p)


if __name__ == '__main__':
    main()
