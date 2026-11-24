"""Score a completed agent run against the task ground truth.

Reads the run's final workbench state (persisted by run_agent) is not available, so this
re-runs the policy and scores it in-line with the same protocol as baseline A.
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE.parent / 'workbench'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'workbench'))  # AgentTools import inside score_run
from run_agent import run_one              # noqa: E402
from protocol import required_transform, rotdeg   # noqa: E402


def score_run(scene, tag, idx=1):
    res = run_one(scene, tag, target_index=idx)
    T = np.asarray(res['T_true_eval_only'], float)
    M = np.asarray(res['final_matrix'], float)
    scale = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
    R_est = M[:3, :3] / max(scale, 1e-12)
    re = rotdeg(R_est @ T[:3, :3].T)
    te = float(np.linalg.norm(M[:3, 3] - T[:3, 3]))
    return {'scene': scene, 'tag': tag, 'sid': res['sid'],
            'rot_err': re, 'trans_err': te, 'scale_err': abs(scale-1.0),
            'overlap': res['hist'][-1].get('overlap', None) if res['hist'] else None,
            'steps': len(res['hist']), 'total_calls': res['budget']['used']['total'],
            'initial_matrix': res['initial_matrix'], 'final_matrix': res['final_matrix'],
            'T_true': res['T_true_eval_only'], 'D_injected': res['D_injected_eval_only'],
            'trace': res['trace'], 'hist': res['hist']}


if __name__ == '__main__':
    scenes = sys.argv[1:] or ['electro']
    rows = []
    for sc in scenes:
        for tag in ('v2', 'v3', 'v4'):
            r = score_run(sc, tag)
            rows.append(r)
            print('%-14s%-4s%-12s rot %6.1f  trans %6.3f  ov %.3f  calls %d'
                  % (sc, tag, r['sid'], r['rot_err'], r['trans_err'],
                     r['overlap'], r['total_calls']))
    p = ROOT / 'out' / 'agent_scores.json'
    prev = json.loads(p.read_text()) if p.exists() else []
    keys = {(r['scene'], r['tag'], r['sid']) for r in rows}
    prev = [r for r in prev if (r['scene'], r['tag'], r['sid']) not in keys]
    p.write_text(json.dumps(prev + rows, indent=1))
    a = np.array([[r['rot_err'], r['overlap']] for r in rows], float)
    print('\nclosed loop: n=%d  median rot %.1f deg  median overlap %.3f  frac<20deg %.0f%%'
          % (len(a), np.median(a[:, 0]), np.median(a[:, 1]), (a[:, 0] < 20).mean() * 100))
