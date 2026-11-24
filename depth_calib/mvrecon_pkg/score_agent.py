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
    TASK = json.loads((ROOT / 'eval_data/task.json').read_text())
    t = TASK[scene][tag]
    A = required_transform(t, res['sid'])
    # rebuild the final state cheaply by replaying the same policy on fresh tools
    from agent_tools import AgentTools
    tools = AgentTools(ROOT / f'agent_data/{scene}', stride=32)
    tools.wb.set_reference(t['ref'])
    for sid, M in t['D_injected'].items():
        if sid == t['ref']:
            continue
        M = np.asarray(M)
        c = tools.wb.clouds[sid]
        Sc = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
        c.R, c.t, c.s = M[:3, :3] / Sc, M[:3, 3].copy(), float(c.s * Sc)
    from run_agent import Policy
    pol = Policy(tools, res['sid'])
    pol.run()
    M = tools.wb.clouds[res['sid']].matrix()
    re = rotdeg(M[:3, :3] @ np.linalg.inv(A[:3, :3]))
    te = float(np.linalg.norm(M[:3, 3] - A[:3, 3]))
    ov = tools.wb.overlap_stats(res['sid'])[0]
    return {'scene': scene, 'tag': tag, 'sid': res['sid'], 'rot_err': re,
            'trans_err': te, 'overlap': ov, 'steps': len(res['hist']),
            'total_calls': res['budget']['used']['total']}


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
