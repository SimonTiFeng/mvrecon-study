"""Does masking sky change the rotation floor and basin? (the VLM's observation, tested)

The agent looked at the playground clouds and reported that ~35-40% of each is sky-coloured and
that the monocular model puts sky at 10-30 m. Removing it dropped the apparent overlap and revealed
that the baseline's alignment was partly two fabricated far surfaces coinciding.

That raises a sharper question than "can the agent nudge rotation": were the measured rotation
floor (0.65 deg for playground) and the narrow +-2 deg basin themselves artefacts of those sky
points? If so, the agent's contribution is the mask, not the nudging, and it is a much better
result -- a principled preprocessing decision rather than a search.

For the four narrow-basin targets plus two controls, compare under each mask:
  floor       ICP started at the TRUE rotation  -> the precision the geometry supports
  from coarse ICP started at the baseline's rotation -> can local refinement reach the floor
  reached     whether from-coarse lands within 1 deg of the floor
"""
from __future__ import annotations

import cv2
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
RNG = np.random.default_rng(0)

TARGETS = [
    ('playground', 'v2', 'eth3d_0286'),
    ('playground', 'v3', 'eth3d_0286'),
    ('playground', 'v4', 'eth3d_0286'),
    ('facade', 'v3', 'eth3d_0209'),
    ('office', 'v3', 'eth3d_0149'),
    ('courtyard', 'v3', 'eth3d_0000'),
]
MASKS = [('none', None, False), ('sky', None, True),
         ('sky+12m', 12.0, True), ('sky+20m', 20.0, True)]


def colors_of(scene, sid, cloud, stride=32):
    img = cv2.cvtColor(cv2.imread(str(ROOT / 'agent_data' / scene / (sid + '.png'))),
                       cv2.COLOR_BGR2RGB)
    H, W = cloud.depth.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    z = cloud.depth[ys, xs]
    ok = np.isfinite(z) & (z > 0) & (z < 200)
    return img[ys[ok], xs[ok]].astype(float)


def keep_mask(scene, sid, cloud, max_dist, drop_sky):
    keep = np.ones(len(cloud.points), bool)
    if max_dist is not None:
        keep &= np.linalg.norm(cloud.points, axis=1) < max_dist
    if drop_sky:
        col = colors_of(scene, sid, cloud)
        if len(col) == len(keep):
            sky = (col[:, 2] > col[:, 0] + 25) & (col[:, 2] > 110) & (col[:, 1] > col[:, 0])
            keep &= ~sky
    return keep


def prep(P, voxel=0.5, max_pts=4000):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P)
    pc = pc.voxel_down_sample(voxel)
    if len(pc.points) > max_pts:
        pc = pc.select_by_index(RNG.choice(len(pc.points), max_pts, replace=False))
    return pc


def anchor_of(p_ref, R_true, t_true, K, depth):
    """One genuinely co-observed reference point, and its position in the moving frame."""
    p_m = (p_ref - t_true) @ R_true
    z = p_m[:, 2]
    ok = z > 0.5
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H, W = depth.shape
    u = np.full(len(p_ref), np.nan)
    v = np.full(len(p_ref), np.nan)
    u[ok] = fx * p_m[ok, 0] / z[ok] + cx
    v[ok] = fy * p_m[ok, 1] / z[ok] + cy
    fr = ok & np.isfinite(u) & np.isfinite(v) & (u >= 2) & (u < W - 2) & (v >= 2) & (v < H - 2)
    cand = np.where(fr)[0]
    if len(cand) == 0:
        return None, None
    i = int(cand[len(cand) // 2])
    return p_ref[i], p_m[i]


def main():
    rows = []
    for scene, tag, sid in TARGETS:
        x = next(r for r in BASE if r['method'] == 'A_icp' and r['scene'] == scene
                 and r['tag'] == tag and r['sid'] == sid)
        wb = Workbench(ROOT / 'agent_data' / scene, stride=32)
        setup(wb, TASK[scene][tag], TASK[scene][tag]['ref'])
        ref_c, mov_c = wb.clouds[wb.ref], wb.clouds[sid]
        M = np.asarray(x['matrix_est'], float)
        R_coarse = M[:3, :3] / max(np.cbrt(abs(np.linalg.det(M[:3, :3]))), 1e-12)
        T = np.asarray(x['matrix_gt'], float)
        R_true, t_true = T[:3, :3], T[:3, 3]
        pr_all = ref_c.points

        pa, pm = anchor_of(pr_all, R_true, t_true, mov_c.K, mov_c.depth)
        if pa is None:
            print('%-14s%-4s%-12s no anchor' % (scene, tag, sid), flush=True)
            continue
        row = {'scene': scene, 'tag': tag, 'sid': sid, 'masks': {}}
        for mname, md, ds in MASKS:
            kr = keep_mask(scene, ref_c.sid, ref_c, md, ds)
            km = keep_mask(scene, sid, mov_c, md, ds)
            if kr.sum() < 400 or km.sum() < 400:
                row['masks'][mname] = {'skip': 'too few points',
                                       'n': [int(kr.sum()), int(km.sum())]}
                continue
            src, dst = prep(mov_c.points[km]), prep(pr_all[kr])

            def at(R0, dst=dst, src=src):
                t0 = pa - R0 @ pm
                T0 = np.eye(4)
                T0[:3, :3] = R0
                T0[:3, 3] = t0
                r = o3d.pipelines.registration.registration_icp(
                    src, dst, 0.30, T0,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
                return rotdeg(r.transformation[:3, :3] @ R_true.T), float(r.fitness)

            fl, ff = at(R_true)
            co, cf = at(R_coarse)
            row['masks'][mname] = {'floor': fl, 'floor_fit': ff,
                                   'from_coarse': co, 'coarse_fit': cf,
                                   'reached': bool(co <= max(1.0, fl + 0.5)),
                                   'n': [int(kr.sum()), int(km.sum())]}
        rows.append(row)
        cells = '  '.join('%s: %.2f/%.2f%s' % (m, v['floor'], v['from_coarse'],
                                               '*' if v.get('reached') else '')
                          for m, v in row['masks'].items() if 'floor' in v)
        print('%-14s%-4s%-12s floor/from-coarse: %s' % (scene, tag, sid, cells), flush=True)

    p = ROOT / 'out' / 'masked_rotation.json'
    p.write_text(json.dumps(rows, indent=1))
    print('\n(* = local refinement reached the floor; entries are floor / from-coarse, degrees)')
    for mname, _, _ in MASKS:
        fl = [r['masks'][mname]['floor'] for r in rows if 'floor' in r['masks'].get(mname, {})]
        co = [r['masks'][mname]['from_coarse'] for r in rows if 'floor' in r['masks'].get(mname, {})]
        rc = sum(1 for r in rows if r['masks'].get(mname, {}).get('reached'))
        if fl:
            print('%-8s median floor %5.2f deg | median from-coarse %5.2f deg | reached %d/%d'
                  % (mname, np.median(fl), np.median(co), rc, len(fl)))
    print('wrote', p)


if __name__ == '__main__':
    main()
