"""Re-score saved baseline and agent traces with the corrected cumulative-transform rule.

Scoring rule: M_est is the CUMULATIVE transform applied to the original cloud. Compare
it to T_true (camera_j -> reference). Never compare it to the incremental A=T_true@inv(D).
"""
from __future__ import annotations
import json
import pathlib
import numpy as np

ROOT=pathlib.Path('mvrecon')

def rotdeg(R):
    c=(np.trace(R)-1)/2
    return float(np.degrees(np.arccos(np.clip(c,-1,1))))

def score(M,T):
    M=np.asarray(M,float); T=np.asarray(T,float)
    scale=float(np.cbrt(abs(np.linalg.det(M[:3,:3]))))
    R=M[:3,:3]/max(scale,1e-12)
    return {'rot_err_deg':rotdeg(R@T[:3,:3].T),
            'trans_err_m':float(np.linalg.norm(M[:3,3]-T[:3,3])),
            'scale_err':abs(scale-1.0)}

# Baseline A + identity: these rows contain the actual final cumulative matrix.
base=json.loads((ROOT/'out/baseline_corrected.json').read_text())
for r in base:
    r['score_corrected']=score(r['matrix_est'],r['matrix_gt'])
(ROOT/'out/baseline_rescored.json').write_text(json.dumps(base,indent=1))

# Agent: run_agent now persists initial/final matrices, true T, injected D, full tool trace.
agent=json.loads((ROOT/'out/agent_runs_corrected.json').read_text())
for r in agent:
    r['score_corrected']=score(r['final_matrix'],r['T_true_eval_only'])
    ovs=[h['overlap'] for h in r.get('hist',[]) if h.get('overlap') is not None]
    r['score_corrected']['overlap_final']=float(ovs[-1]) if ovs else None
(ROOT/'out/agent_rescored.json').write_text(json.dumps(agent,indent=1))

# Consistency audit: current task matrix, identity's M=D, and saved row's matrix must align.
task=json.loads((ROOT/'eval_data/task.json').read_text())
errs=[]
for r in base:
    D=np.asarray(task[r['scene']][r['tag']]['D_injected'][r['sid']],float)
    T=np.asarray(task[r['scene']][r['tag']]['T_true'][r['sid']],float)
    if r['method']=='identity':
        em=float(np.max(np.abs(np.asarray(r['matrix_est'])-D)))
        if em>1e-8: errs.append((r['scene'],r['tag'],r['sid'],em))
    if np.max(np.abs(np.asarray(r['matrix_gt'])-T))>1e-8:
        errs.append((r['scene'],r['tag'],r['sid'],'T mismatch'))
print('baseline rows',len(base),'agent rows',len(agent),'task/result mismatches',len(errs))
if errs: print('first',errs[:5])
for method in ('identity','A_icp'):
    a=[r['score_corrected'] for r in base if r['method']==method]
    print(method,'n',len(a),'rot med',round(float(np.median([x['rot_err_deg'] for x in a])),2),
          'trans med',round(float(np.median([x['trans_err_m'] for x in a])),3),
          'rot<20',sum(x['rot_err_deg']<20 for x in a),'/',len(a))
if agent:
    print('agent corrected:')
    for r in agent:
        sc=r['score_corrected']
        print(r['scene'],r['tag'],r['sid'],
              'rot',round(sc['rot_err_deg'],1),'trans',round(sc['trans_err_m'],3),
              'trace ops',len(r['trace']),'hist',len(r['hist']))
