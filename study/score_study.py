#!/usr/bin/env python
"""Score agent sessions, verify their trajectories, and summarise against baselines.

Reads only: sandbox sessions (agent traces/state), eval_data/task.json (GT, eval side),
eval/baselines.json.  Writes eval results and a summary; never touches a session.
"""
from __future__ import annotations

import csv
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'study'))
sys.path.insert(0, str(ROOT / 'eval'))
sys.path.insert(0, str(ROOT / 'workbench'))
from baselines_full import TwoView  # noqa: E402
from protocol import rotdeg  # noqa: E402

SB = pathlib.Path(os.environ.get('MVREBENCH_SANDBOX',
                                 pathlib.Path(os.environ.get('TEMP', '/tmp')) / 'reg_wb_study_20261123'))
OUT = ROOT / 'out' / 'agent_study_20261123'
TASK = json.loads((ROOT / 'eval_data' / 'task.json').read_text())


def score_matrix(M, T):
    M, T = np.asarray(M, float), np.asarray(T, float)
    sc = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
    R = M[:3, :3] / max(sc, 1e-12)
    rot = rotdeg(R @ T[:3, :3].T)
    d = M[:3, 3] - T[:3, 3]
    return {'rot_err': rot, 'lat_err': float(np.linalg.norm(d[:2])),
            'trans_err': float(np.linalg.norm(d)), 'scale_err': abs(sc - 1.0)}


def replay(case, sess, trace):
    M0 = np.asarray(sess['initial_matrix'], float)
    M = M0.copy()
    stack = []
    issues = []
    applied = 0
    for i, rec in enumerate(trace):
        before = rec.get('matrix_before')
        after = rec.get('matrix_after')
        if before is not None:
            err = float(np.abs(np.asarray(before, float) - M).max())
            if err > 1e-8:
                issues.append({'step': rec['step'], 'kind': 'continuity',
                               'error': err})
        inc = None
        res = rec.get('result') or {}
        if isinstance(res, dict):
            inc = res.get('increment_matrix', res.get('increment'))
        if inc is not None:
            inc = np.asarray(inc, float)
            stack.append(M.copy())
            M = inc @ M
            applied += 1
            if after is not None:
                err = float(np.abs(np.asarray(after, float) - M).max())
                if err > 1e-8:
                    issues.append({'step': rec['step'], 'kind': 'increment', 'error': err})
        elif rec.get('op') == 'undo':
            M = np.asarray(after, float)
            if stack:
                stack.pop()
        else:
            if after is not None:
                err = float(np.abs(np.asarray(after, float) - M).max())
                if err > 1e-8:
                    issues.append({'step': rec['step'], 'kind': 'no_op', 'error': err})
    final = np.eye(4)
    p = sess['poses'][sess['target']]
    final[:3, :3] = np.asarray(p['R'], float) * float(p['s'])
    final[:3, 3] = np.asarray(p['t'], float)
    final_err = float(np.abs(final - M).max())
    return {'case': case, 'n_records': len(trace), 'n_increment_applied': applied,
            'n_undo': sum(1 for r in trace if r.get('op') == 'undo'),
            'max_issue_error': max([x['error'] for x in issues], default=0.0),
            'n_issues': len(issues), 'final_replay_error': final_err,
            'passed': final_err < 1e-8 and not issues, 'issues': issues[:8]}


def main():
    cases = json.loads((SB / 'cases.json').read_text())
    TASK = json.loads((ROOT / 'eval_data' / 'task.json').read_text())
    base = json.loads((OUT / 'eval' / 'baselines.json').read_text())
    rows, recon = [], []
    for c in cases:
        case, scene, tag, sid = c['case'], c['scene'], c['tag'], c['target']
        d = SB / 'sessions' / case
        sess = json.loads((d / 'session.json').read_text())
        trace = [json.loads(l) for l in (d / 'trace.jsonl').read_text().splitlines() if l.strip()]
        r = replay(case, sess, trace)
        recon.append(r)
        p = sess['poses'][sid]
        M = np.eye(4)
        M[:3, :3] = np.asarray(p['R'], float) * float(p['s'])
        M[:3, 3] = np.asarray(p['t'], float)
        T = np.asarray(TASK[scene][tag]['T_true'][sid], float)
        sc = score_matrix(M, T)
        wb = TwoView(scene, c['ref'], sid)
        wb.clouds[sid].R = np.asarray(p['R'], float)
        wb.clouds[sid].t = np.asarray(p['t'], float)
        wb.clouds[sid].s = float(p['s'])
        ov = wb.overlap_stats(sid)[0]
        ops = [t.get('op') for t in trace]
        used = sess['used']
        rows.append({'case': case, 'scene': scene, 'tag': tag, 'sid': sid,
                     'ref': c['ref'], 'status': sess['status'],
                     'reason': sess.get('reason', ''),
                     'rot_err': sc['rot_err'], 'lat_err': sc['lat_err'],
                     'trans_err': sc['trans_err'], 'scale_err': sc['scale_err'],
                     'overlap': ov,
                     'success': bool(sc['rot_err'] < 2 and sc['lat_err'] < 0.5),
                     'n_actions': len(trace),
                     'n_state_changes': sum(1 for t in trace if (t.get('result') or {}).get(
                         'increment_matrix') is not None or (t.get('result') or {}).get(
                         'increment') is not None),
                     'n_undo': sum(1 for x in ops if x == 'undo'),
                     'n_match': sum(1 for x in ops if x == 'match'),
                     'n_solve': sum(1 for x in ops if x == 'solve'),
                     'n_icp': sum(1 for x in ops if x == 'icp'),
                     'n_fpfh': sum(1 for x in ops if x == 'fpfh'),
                     'n_render': sum(1 for x in ops if x == 'render'),
                     'used': used, 'budget': sess['budget'],
                     'recon_passed': r['passed'], 'recon_error': r['final_replay_error'],
                     'matrix_est': M.tolist(), 'matrix_gt': T.tolist()})

    OUT.joinpath('eval').mkdir(parents=True, exist_ok=True)
    (OUT / 'eval' / 'agent_results.json').write_text(json.dumps(rows, indent=1))
    (OUT / 'eval' / 'reconstruction_check.json').write_text(json.dumps(recon, indent=1))

    # combined per-sample csv
    csvp = OUT / 'eval' / 'per_sample.csv'
    with open(csvp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'scene', 'tag', 'sid', 'rot_err', 'lat_err', 'trans_err',
                    'overlap', 'success', 'status'])
        for r in base:
            w.writerow([r['method'], r['scene'], r['tag'], r['sid'],
                        r.get('rot_err', ''), r.get('lat_err', ''),
                        r.get('trans_err', ''), r.get('overlap', ''),
                        r.get('success', ''), ''])
        for r in rows:
            w.writerow(['AGENT', r['scene'], r['tag'], r['sid'], r['rot_err'],
                        r['lat_err'], r['trans_err'], r['overlap'], r['success'],
                        r['status']])

    summary = summarise(base, rows, recon)
    (OUT / 'eval' / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def metrics(rs):
    rot = np.array([r['rot_err'] for r in rs], float)
    lat = np.array([r['lat_err'] for r in rs], float)
    tr = np.array([r['trans_err'] for r in rs], float)
    return {'n': len(rs),
            'rot_median': float(np.median(rot)),
            'rot_mean': float(np.mean(rot)),
            'rot_q25': float(np.percentile(rot, 25)),
            'rot_q75': float(np.percentile(rot, 75)),
            'rot_lt2': int((rot < 2).sum()),
            'rot_lt5': int((rot < 5).sum()),
            'rot_lt20': int((rot < 20).sum()),
            'lat_median': float(np.median(lat)),
            'lat_lt0_5': int((lat < 0.5).sum()),
            'trans_median': float(np.median(tr)),
            'success': int(((rot < 2) & (lat < 0.5)).sum()),
            'success_rate': float(((rot < 2) & (lat < 0.5)).mean())}


def summarise(base, agent, recon):
    out = {'n_instances': len(agent),
           'n_unique_view_pairs': len({(r['scene'], r['ref'], r['sid']) for r in agent}),
           'n_abstain': sum(1 for r in agent if r['status'] == 'abstain'),
           'n_done': sum(1 for r in agent if r['status'] == 'done'),
           'n_no_finish': sum(1 for r in agent if r['status'] == 'running'),
           'reconstruction_passed': sum(1 for r in recon if r['passed']),
           'reconstruction_failed': [r['case'] for r in recon if not r['passed']],
           'agent_budget_used_median': float(np.median([r['used']['total'] for r in agent])),
           'agent_actions_median': float(np.median([r['n_actions'] for r in agent])),
           'agent_undo_total': int(sum(r['n_undo'] for r in agent)),
           'agent_match_total': int(sum(r['n_match'] for r in agent)),
           'agent_solve_total': int(sum(r['n_solve'] for r in agent)),
           'agent_icp_total': int(sum(r['n_icp'] for r in agent)),
           'agent_fpfh_total': int(sum(r['n_fpfh'] for r in agent))}
    methods = sorted(set(r['method'] for r in base)) + ['AGENT']
    allrows = base + [dict(r, method='AGENT') for r in agent]
    out['overall'] = {m: metrics([r for r in allrows if r['method'] == m and 'rot_err' in r])
                      for m in methods}
    out['by_scene'] = {m: {s: metrics([r for r in allrows if r['method'] == m
                                       and r['scene'] == s and 'rot_err' in r])
                           for s in sorted({r['scene'] for r in allrows})}
                       for m in methods}
    out['by_tag'] = {m: {t: metrics([r for r in allrows if r['method'] == m
                                     and r['tag'] == t and 'rot_err' in r])
                         for t in ('v2', 'v3', 'v4')}
                     for m in methods}
    # reporting subset: coarse A rotation < 20 deg (GT-based selection used ONLY here)
    A = {r['case'] if 'case' in r else '%s_%s_%s' % (r['scene'], r['tag'], r['sid']): r
         for r in base if r['method'] == 'A_fpfh_icp'}
    subset = {k for k, r in A.items() if r.get('rot_err', 999) < 20}
    out['subset_coarseA_lt20'] = {
        'rule': 'A_fpfh_icp rotation error < 20 deg (GT-based; reporting only)',
        'n': len(subset),
        'methods': {m: metrics([r for r in allrows if r['method'] == m
                                and ('case' in r and r['case'] in subset or
                                     'case' not in r and '%s_%s_%s' % (
                                         r['scene'], r['tag'], r['sid']) in subset)
                                and 'rot_err' in r])
                    for m in methods}}
    # unique pairs: lowest tag instance per (scene, ref, sid)
    order = {'v2': 0, 'v3': 1, 'v4': 2}
    rep = {}
    for r in allrows:
        key = (r['scene'], r.get('ref'), r['sid'], r['method'])
        if 'rot_err' not in r:
            continue
        if key not in rep or order[r['tag']] < order[rep[key]['tag']]:
            rep[key] = r
    for r in agent:
        key = (r['scene'], r['ref'], r['sid'], 'AGENT')
        cur = rep.get(key)
        if cur is None or order[r['tag']] < order[cur['tag']]:
            rep[key] = dict(r, method='AGENT')
    out['unique_pairs'] = {m: metrics([r for r in rep.values() if r['method'] == m])
                           for m in methods}
    # paired comparisons on the 40 instances
    b = {('%s_%s_%s' % (r['scene'], r['tag'], r['sid']), r['method']): r
         for r in base}
    ag = {r['case']: r for r in agent}
    out['paired_vs_baselines'] = {}
    for m in methods:
        if m == 'AGENT':
            continue
        better = worse = same = 0
        dlat, drot = [], []
        for c, a in ag.items():
            br = b.get((c, m))
            if not br or 'rot_err' not in br or 'rot_err' not in a:
                continue
            dl = a['lat_err'] - br['lat_err']
            dr = a['rot_err'] - br['rot_err']
            dlat.append(dl); drot.append(dr)
            if dl < -0.02: better += 1
            elif dl > 0.02: worse += 1
            else: same += 1
        out['paired_vs_baselines'][m] = {
            'lat_better': better, 'lat_worse': worse, 'lat_same': same,
            'lat_delta_median': float(np.median(dlat)) if dlat else None,
            'rot_delta_median': float(np.median(drot)) if drot else None}
    # per-case table
    out['per_case'] = [{'case': r['case'], 'scene': r['scene'], 'tag': r['tag'],
                        'sid': r['sid'], 'status': r['status'],
                        'rot_err': r['rot_err'], 'lat_err': r['lat_err'],
                        'success': r['success'], 'n_actions': r['n_actions'],
                        'budget_used': r['used']['total'],
                        'reason': r['reason'][:160]}
                       for r in agent]
    return out


if __name__ == '__main__':
    main()
