"""Is the translation failure caused by scale coupling?

Both clouds live in the same metric frame, but each single-image depth cloud carries its
own scale bias (measured earlier at 4-7%). A rigid registration cannot absorb that, and
along the optical axis a scale error is nearly indistinguishable from a translation error.

This runs, from a SHARED FPFH/RANSAC initialisation:
  - rigid ICP        (TransformationEstimationPointToPoint, with_scaling=False)
  - similarity ICP   (same estimator, with_scaling=True)
and scores both against T_true.

For the similarity result we report the estimated scale and also the rotation/translation
error after projecting back to a rigid transform, so the comparison to T_true stays
well-defined.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import open3d as o3d

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'eval'))
sys.path.insert(0, str(ROOT / 'workbench'))
from protocol import setup, rotdeg  # noqa: E402

TASK = json.loads((ROOT / 'eval_data/task.json').read_text())
AG = ROOT / 'agent_data'


def features(pc, vox=0.8):
    d = pc.voxel_down_sample(vox)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    f = o3d.pipelines.registration.compute_fpfh_feature(
        d, o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=60))
    return d, f


def decompose(M):
    s = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
    R = M[:3, :3] / max(s, 1e-12)
    return s, R, M[:3, 3]


def run_target(scene, tag, sid):
    """Return dict with rigid and similarity scores for one target."""
    from wb import Workbench
    wb = Workbench(AG / scene, stride=32)
    setup(wb, TASK[scene][tag], TASK[scene][tag]['ref'])
    T = np.asarray(TASK[scene][tag]['T_true'][sid], float)
    M0 = wb.clouds[sid].matrix().copy()

    src = wb.clouds[sid].as_o3d()
    dst = wb.clouds[wb.ref].as_o3d()
    fs, Fs = features(src)
    ft, Ft = features(dst)
    coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        fs, ft, Fs, Ft, True, 1.0,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [],
        o3d.pipelines.registration.RANSACConvergenceCriteria(12000, 0.95))
    init = coarse.transformation

    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

    out = {'scene': scene, 'tag': tag, 'sid': sid,
           'initial_rot': rotdeg(M0[:3, :3]), 'ransac_fitness': float(coarse.fitness)}
    for name, scaling in (('rigid', False), ('similarity', True)):
        est = o3d.pipelines.registration.TransformationEstimationPointToPoint(scaling)
        fine = o3d.pipelines.registration.registration_icp(
            src, dst, 0.5, init, est,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=120))
        M = fine.transformation @ M0
        s, R, t = decompose(M)
        # similarity result also compared after normalising its scale away
        Rn, tn = R, t / max(s, 1e-12)
        out[name] = {
            'scale': s,
            'rot_err': rotdeg(R @ T[:3, :3].T),
            'trans_err': float(np.linalg.norm(t - T[:3, 3])),
            'rot_err_scalenorm': rotdeg(R @ T[:3, :3].T),
            'trans_err_scalenorm': float(np.linalg.norm(tn - T[:3, 3])),
            'overlap': wb.overlap_stats(sid)[0] if name == 'rigid' else None,
            'icp_fitness': float(fine.fitness),
            'icp_rmse': float(fine.inlier_rmse),
        }
    return out


if __name__ == '__main__':
    scenes = sys.argv[1:] or sorted(TASK)
    rows = []
    dst_path = ROOT / 'out' / 'scale_coupling.json'
    for scene in scenes:
        for tag in ('v2', 'v3', 'v4'):
            for sid in TASK[scene][tag]['ids'][1:]:
                r = run_target(scene, tag, sid)
                rows.append(r)
                print('%-14s%-4s%-12s init %5.1f | rigid %6.1f/%6.3f | sim s=%.3f %6.1f/%6.3f'
                      % (scene, tag, sid, r['initial_rot'],
                         r['rigid']['rot_err'], r['rigid']['trans_err'],
                         r['similarity']['scale'],
                         r['similarity']['rot_err'], r['similarity']['trans_err']),
                      flush=True)
    prev = json.loads(dst_path.read_text()) if dst_path.exists() else []
    keys = {(r['scene'], r['tag'], r['sid']) for r in rows}
    prev = [r for r in prev if (r['scene'], r['tag'], r['sid']) not in keys]
    dst_path.write_text(json.dumps(prev + rows, indent=1))
    print('\nwrote', dst_path, 'rows', len(prev) + len(rows))
