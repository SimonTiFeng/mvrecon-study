"""Closed-loop agent driver.

The policy here is deliberately simple and scripted: it is a *reference* closed loop
that exercises observe -> hypothesise -> act -> check -> undo, and produces a trace that
the evaluator can score. A VLM policy can later replace `decide()` without touching the
harness.

It never reads eval_data. Every number it acts on comes from AgentTools.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, 'mvrecon/workbench')
from agent_tools import AgentTools

OUT = pathlib.Path(__file__).resolve().parents[1] / 'out'


class Policy:
    """Scripted but adaptive: each step's choice depends on the last feedback."""

    def __init__(self, tools, target):
        self.t = tools
        self.target = target
        self.hist = []

    def note(self, kind, payload):
        self.hist.append({'kind': kind, **payload})

    def decide(self):
        """Return the next action name. Rule-based on measured feedback."""
        st = self.t.stats(self.target)
        self.note('stats', st)
        if st['overlap'] > 0.55 and st['residual'] < 0.12:
            return 'stop'
        n = len(self.hist)
        if n <= 1:
            return 'coarse'
        if n <= 3:
            return 'solve'
        if n <= 6:
            return 'refine'
        if n <= 8:
            return 'check_undo'
        return 'stop'

    def run(self, max_steps=10):
        for _ in range(max_steps):
            if self.t.exhausted():
                break
            a = self.decide()
            if a == 'stop':
                break
            if a == 'coarse':
                cr = self.t.wb.clouds[self.t.wb.ref].transformed()
                ct = self.t.wb.clouds[self.target].transformed()
                d = np.median(cr, 0) - np.median(ct, 0)
                r = self.t.transform(self.target, t=d)
                self.note('coarse', {'t': d.tolist(), 'overlap': r.get('overlap')})
            elif a == 'solve':
                H, W = self.t.wb.clouds[self.target].depth.shape
                u0 = int(W * 0.15); u1 = int(W * 0.85)
                v0 = int(H * 0.15); v1 = int(H * 0.85)
                r = self.t.solve_correspondences(self.target, (u0, v0, u1, v1),
                                                 (u0, v0, u1, v1))
                self.note('solve', {k: v for k, v in r.items()})
            elif a == 'refine':
                r = self.t.refine_icp(self.target, radius=0.5)
                self.note('refine', {k: v for k, v in r.items()})
            elif a == 'check_undo':
                before = self.t.stats(self.target)
                # Preserve both the live state and the undo-stack top so a rejected
                # undo can be rolled forward exactly (undo() itself is destructive).
                live = {sid: (c.R.copy(), c.t.copy(), c.s)
                        for sid, c in self.t.wb.clouds.items()}
                stack_top = self.t.wb._history[-1] if self.t.wb._history else None
                self.t.undo()
                after = self.t.stats(self.target)
                keep = after['overlap'] >= before['overlap']
                if not keep:
                    if stack_top is not None:
                        self.t.wb._history.append(stack_top)
                    for sid, (R, t, s) in live.items():
                        self.t.wb.clouds[sid].R = R
                        self.t.wb.clouds[sid].t = t
                        self.t.wb.clouds[sid].s = s
                self.note('check_undo', {'before': before['overlap'],
                                         'after': after['overlap'], 'kept_undo': keep})
        return {'target': self.target, 'hist': self.hist,
                'budget': self.t.report(), 'trace': self.t.trace}


def run_one(scene, tag, target_index=1, stride=32):
    TASK = json.loads((pathlib.Path(__file__).resolve().parents[1]
                       / 'eval_data/task.json').read_text())
    t = TASK[scene][tag]
    tools = AgentTools(pathlib.Path(__file__).resolve().parents[1] / f'agent_data/{scene}',
                       stride=stride)
    # harness applies the injected displacement; the policy does not know it
    tools.wb.set_reference(t['ref'])
    for sid, M in t['D_injected'].items():
        if sid == t['ref']:
            continue
        M = np.asarray(M)
        c = tools.wb.clouds[sid]
        Sc = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
        c.R, c.t, c.s = M[:3, :3] / Sc, M[:3, 3].copy(), float(c.s * Sc)
    tgt = t['ids'][target_index]
    initial_matrix = tools.wb.clouds[tgt].matrix().tolist()
    pol = Policy(tools, tgt)
    res = pol.run()
    final_matrix = tools.wb.clouds[tgt].matrix().tolist()
    res.update({'scene': scene, 'tag': tag, 'sid': tgt,
                'initial_matrix': initial_matrix, 'final_matrix': final_matrix,
                'T_true_eval_only': t['T_true'][tgt],
                'D_injected_eval_only': t['D_injected'][tgt]})
    return res


if __name__ == '__main__':
    scenes = sys.argv[1:] or ['electro']
    out = []
    for scene in scenes:
        TASK = json.loads((pathlib.Path(__file__).resolve().parents[1] / 'eval_data/task.json').read_text())
        for tag in ('v2', 'v3', 'v4'):
            n_targets = len(TASK[scene][tag]['ids']) - 1
            for target_index in range(1, n_targets + 1):
                r = run_one(scene, tag, target_index=target_index)
                print('%s %s %s  steps=%d  calls=%d'
                      % (scene, tag, r['sid'], len(r['hist']), r['budget']['used']['total']))
                out.append(r)
    p = OUT / 'agent_runs_corrected.json'
    p.write_text(json.dumps(out, indent=1))
    print('wrote', p, 'rows', len(out))
