"""Evaluate cumulative transforms against the true camera-j -> camera-ref transform.

COLMAP stores world->camera extrinsics. build_gt converts these to T_true (camera j to
reference camera). The displayed cloud starts at M=D. Every agent/baseline operation is
left-composed, so its final cumulative matrix M_est should equal T_true.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, 'mvrecon/workbench')
from wb import Workbench


def rotdeg(R):
    c = (np.trace(R) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def required_transform(task, sid):
    """The correct FINAL cumulative transform is T_true, not the incremental A.

    Initial cumulative state M0 = D_injected. Applying incremental A=T_true@inv(D)
    gives M_final = A @ D = T_true. Therefore score M_final directly against T_true.
    """
    return np.asarray(task['T_true'][sid], float)


def setup(wb, task, ref):
    """Place each non-reference cloud at its deliberately injected start transform D."""
    wb.set_reference(ref)
    for sid, mat in task['D_injected'].items():
        if sid == ref:
            continue
        M = np.asarray(mat, float)
        Sc = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
        c = wb.clouds[sid]
        c.R = M[:3, :3] / Sc
        c.t = M[:3, 3].copy()
        c.s = Sc


def score(wb, task, sid):
    T = required_transform(task, sid)
    M = wb.clouds[sid].matrix()
    # M may contain a uniform scale. Strip it before computing the rotation error.
    scale = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
    R_est = M[:3, :3] / max(scale, 1e-12)
    re = rotdeg(R_est @ T[:3, :3].T)
    te = float(np.linalg.norm(M[:3, 3] - T[:3, 3]))
    scale_err = abs(scale - 1.0)
    return {'rot_err': re, 'trans_err': te, 'scale_err': scale_err,
            'overlap': wb.overlap_stats(sid)[0],
            'matrix_est': M.tolist(), 'matrix_gt': T.tolist()}


def evaluate(scene, tag, method, verbose=True):
    AG = pathlib.Path('mvrecon/agent_data')
    TASK = json.loads(pathlib.Path('mvrecon/eval_data/task.json').read_text())
    t = TASK[scene][tag]
    wb = Workbench(AG / scene, stride=32)
    setup(wb, t, t['ref'])
    out = []
    for sid in t['ids'][1:]:
        initial = wb.clouds[sid].matrix().tolist()
        r = method(wb, t, sid)
        s = score(wb, t, sid)
        s.update({'sid': sid, 'method_result': r, 'initial_matrix': initial,
                  'D_injected': t['D_injected'][sid], 'T_true': t['T_true'][sid],
                  'reference_sid': t['ref']})
        out.append(s)
        if verbose:
            print('  %-24s rot %6.1f  trans %6.3f  ov %.3f'
                  % (sid, s['rot_err'], s['trans_err'], s['overlap']))
    return out


def method_identity(wb, task, sid):
    return 'identity'


def method_icp(wb, task, sid):
    """Baseline A on the task: FPFH+RANSAC then point-to-plane ICP."""
    import open3d as o3d
    src = wb.clouds[sid].as_o3d()
    dst = wb.clouds[wb.ref].as_o3d()
    fs = src.voxel_down_sample(0.80); ft = dst.voxel_down_sample(0.80)
    fs.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    ft.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    Fs = o3d.pipelines.registration.compute_fpfh_feature(
        fs, o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=60))
    Ft = o3d.pipelines.registration.compute_fpfh_feature(
        ft, o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=60))
    coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        fs, ft, Fs, Ft, True, 1.0,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [],
        o3d.pipelines.registration.RANSACConvergenceCriteria(12000, 0.95))
    # point-to-plane needs normals on BOTH clouds used for ICP
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    fine = o3d.pipelines.registration.registration_icp(
        src, dst, 0.5, coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    wb._snap()
    nm = fine.transformation @ wb.clouds[sid].matrix()
    c = wb.clouds[sid]
    Sc = float(np.cbrt(abs(np.linalg.det(nm[:3, :3]))))
    c.R, c.t, c.s = nm[:3, :3] / Sc, nm[:3, 3], Sc
    return {'method': 'icp', 'coarse_transform': coarse.transformation.tolist(),
            'fine_transform': fine.transformation.tolist(),
            'ransac_fitness': float(coarse.fitness),
            'icp_fitness': float(fine.fitness), 'icp_rmse': float(fine.inlier_rmse)}


if __name__ == '__main__':
    scenes = sys.argv[1:] or ['electro', 'playground', 'office']
    for scene in scenes:
        print('\n=== %s ===' % scene)
        for tag in ('v2', 'v3', 'v4'):
            print(' [%s] identity:' % tag)
            evaluate(scene, tag, method_identity)
            print(' [%s] baseline A:' % tag)
            evaluate(scene, tag, method_icp)
