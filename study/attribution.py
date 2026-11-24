#!/usr/bin/env python
"""Attribute each agent's final transform to the tool sequence it used.

For every session: list the state-changing operations, then compare the final matrix
to each traditional method's matrix on the same case.  Exact/near matches identify
runs whose outcome is the tool's output; distinct outcomes with better error are the
candidates for a genuine decision contribution.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'workbench'))
sys.path.insert(0, str(ROOT / 'eval'))
from protocol import rotdeg  # noqa: E402

OUT = ROOT / 'out' / 'agent_study_20261123'
SB = pathlib.Path(os.environ.get('MVREBENCH_SANDBOX',
                                 pathlib.Path(os.environ.get('TEMP', '/tmp')) / 'reg_wb_study_20261123'))
TASK = json.loads((ROOT / 'eval_data' / 'task.json').read_text())


def score_M(M, scene, tag, sid):
    T = np.asarray(TASK[scene][tag]['T_true'][sid], float)
    M = np.asarray(M, float)
    sc = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
    R = M[:3, :3] / max(sc, 1e-12)
    rot = rotdeg(R @ T[:3, :3].T)
    d = M[:3, 3] - T[:3, 3]
    return rot, float(np.linalg.norm(d[:2]))


def mat_diff(A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    return {'rot_deg': rotdeg(A[:3, :3] @ B[:3, :3].T),
            'trans_m': float(np.linalg.norm(A[:3, 3] - B[:3, 3]))}


def main():
    base = json.loads((OUT / 'eval' / 'baselines.json').read_text())
    agent = json.loads((OUT / 'eval' / 'agent_results.json').read_text())
    b = {}
    for r in base:
        if 'rot_err' in r:
            b[('%s_%s_%s' % (r['scene'], r['tag'], r['sid']), r['method'])] = r
    rows = []
    for a in agent:
        case = a['case']
        trace = [json.loads(l) for l in (SB / 'sessions' / case / 'trace.jsonl')
                 .read_text().splitlines() if l.strip()]
        ops = []
        for t in trace:
            op = t['op']
            if op == 'solve':
                ops.append('solve:' + str((t.get('params') or {}).get('kind')))
            elif op in ('icp', 'fpfh', 'rotate', 'translate', 'mask'):
                ops.append(op)
        changing = [o for o in ops if o in ('icp', 'fpfh', 'rotate', 'translate')
                    or o.startswith('solve:')]
        first = None
        for t in trace:
            res = t.get('result') or {}
            if t['op'] in ('solve', 'icp', 'fpfh') and isinstance(res, dict) and res.get('matrix'):
                first = {'op': t['op'] + (':' + str((t.get('params') or {}).get('kind'))
                                          if t['op'] == 'solve' else ''),
                         'matrix': res['matrix']}
                break
        first_score = None
        if first:
            r0, l0 = score_M(first['matrix'], a['scene'], a['tag'], a['sid'])
            first_score = {'solver': first['op'], 'rot': r0, 'lat': l0}
        closest, best = None, None
        for m in ('B_sift_rigid', 'C_sift_rigid_sky', 'D_rotA_sift_t',
                  'E_rotA_sift_t_sky', 'F_rotA_sift_corr',
                  'G_rotA_sift_reproj', 'H_sift_rigid_reproj'):
            br = b.get((case, m))
            if not br:
                continue
            d = mat_diff(a['matrix_est'], br['matrix_est'])
            score = d['rot_deg'] + 10 * d['trans_m']
            if best is None or score < best[0]:
                best = (score, m, d)
        if best is not None:
            closest = {'method': best[1], **best[2], 'same': best[2]['rot_deg'] < 0.1
                       and best[2]['trans_m'] < 0.01}
        rows.append({'case': case, 'sequence': changing,
                     'n_state_changes': len(changing),
                     'n_icp': changing.count('icp'), 'n_fpfh': changing.count('fpfh'),
                     'n_solve': sum(1 for o in changing if o.startswith('solve:')),
                     'n_manual': changing.count('rotate') + changing.count('translate'),
                     'closest_baseline': closest,
                     'first_solver': first_score,
                     'multi_round_delta_rot': (a['rot_err'] - first_score['rot']
                                               if first_score else None),
                     'multi_round_delta_lat': (a['lat_err'] - first_score['lat']
                                               if first_score else None),
                     'rot_err': a['rot_err'], 'lat_err': a['lat_err'],
                     'success': a['success'], 'status': a['status']})
    (OUT / 'eval' / 'attribution.json').write_text(json.dumps(rows, indent=1))

    from collections import Counter
    groups = Counter()
    for r in rows:
        seq = r['sequence']
        if not seq:
            groups['no_state_change'] += 1
        elif seq[0].startswith('solve:rigid3d') and r['n_icp'] == 0 and r['n_fpfh'] == 0:
            groups['rigid3d_only'] += 1
        elif seq[0].startswith('solve:rigid3d') and r['n_icp'] > 0:
            groups['rigid3d_then_icp'] += 1
        elif seq[0] == 'fpfh':
            groups['fpfh_start'] += 1
        elif seq[0].startswith('solve:reproj'):
            groups['reproj_start'] += 1
        else:
            groups['other'] += 1
    summary = {'n': len(rows), 'sequence_groups': dict(groups),
               'n_closest_same': sum(1 for r in rows if r['closest_baseline'] and
                                     r['closest_baseline']['same']),
               'closest_counts': dict(Counter(r['closest_baseline']['method']
                                              for r in rows if r['closest_baseline'])),
               'n_abstain': sum(1 for r in rows if r['status'] == 'abstain')}
    mr = [r for r in rows if r['first_solver']]
    summary['multi_round'] = {
        'n_with_first_solver': len(mr),
        'rot_improved': sum(1 for r in mr if r['multi_round_delta_rot'] < -0.1),
        'rot_worsened': sum(1 for r in mr if r['multi_round_delta_rot'] > 0.1),
        'lat_improved': sum(1 for r in mr if r['multi_round_delta_lat'] < -0.02),
        'lat_worsened': sum(1 for r in mr if r['multi_round_delta_lat'] > 0.02),
        'rot_delta_median': float(np.median([r['multi_round_delta_rot'] for r in mr])),
        'lat_delta_median': float(np.median([r['multi_round_delta_lat'] for r in mr]))}
    (OUT / 'eval' / 'attribution_summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
