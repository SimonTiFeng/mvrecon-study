"""Build the actual registration TASK.

Each cloud is still produced by an independent per-image inference. But before the agent
sees it, the harness places it in a COMMON frame using the reference pose, then applies a
deliberate unknown displacement D_i. The agent must recover the relative pose; the eval
side knows exactly what was removed.

Why: two raw mono clouds already overlap 56-100% at identity (measured), so without this
the task has no signal. Applying the true pose makes the shared structure actually align;
the injected D makes the starting state genuinely wrong by a known amount.
"""
import json, pathlib, numpy as np

AG = pathlib.Path('mvrecon/agent_data')
EV = pathlib.Path('mvrecon/eval_data')
GT = json.loads((EV/'gt_transforms.json').read_text())

rng = np.random.default_rng(20261122)
task = {}
for scene in sorted(GT):
    ev = json.loads((EV/scene/'meta.json').read_text())
    pose = {x['sid']: x for x in ev['images']}
    for tag in ('v2','v3','v4'):
        if tag not in GT[scene]: continue
        d = GT[scene][tag]; ids = d['ids']; ref = d['ref']
        # displacement injected per view: a real rotation + translation, seeded per view
        D = {}
        for sid in ids:
            if sid == ref: 
                D[sid] = np.eye(4).tolist(); continue
            ang = rng.uniform(25, 70) * np.pi/180
            axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
            K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
            R = np.eye(3)+np.sin(ang)*K+(1-np.cos(ang))*(K@K)
            # translation scale tied to the scene's own true baseline
            tb = float(np.linalg.norm(np.asarray(d['T'][sid])[:3,3]))
            t = rng.normal(size=3); t /= np.linalg.norm(t)
            t = t * max(tb*1.2, 0.35)
            M = np.eye(4); M[:3,:3]=R; M[:3,3]=t
            D[sid] = M.tolist()
        task.setdefault(scene,{})[tag] = {
            'ref': ref, 'ids': ids,
            'T_true': d['T'],              # cam_j -> cam_ref (what a method should recover)
            'D_injected': D,               # applied on top, unknown to the agent
        }

(EV/'task.json').write_text(json.dumps(task, indent=1))
print(f"{'scene':<14}{'tag':<4}{'n':>3}{'inj_rot_deg':>12}{'inj_trans_m':>13}{'true_baseline':>14}")
for scene in sorted(task):
    for tag in ('v2','v3','v4'):
        t=task[scene][tag]; ids=t['ids']
        rots=[];trs=[]
        for sid in ids:
            M=np.asarray(t['D_injected'][sid])
            c=(np.trace(M[:3,:3])-1)/2
            rots.append(np.degrees(np.arccos(np.clip(c,-1,1))))
            trs.append(np.linalg.norm(M[:3,3]))
        print(f"{scene:<14}{tag:<4}{len(ids):>3}{np.mean(rots):>12.1f}{np.mean(trs):>13.2f}"
              f"{np.mean([np.linalg.norm(np.asarray(t['T_true'][s])[:3,3]) for s in ids]):>14.2f}")
