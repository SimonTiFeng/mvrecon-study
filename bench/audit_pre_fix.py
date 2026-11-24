"""Recompute initial-state rotation under old package and the correct COLMAP convention."""
import json,pathlib,numpy as np
ARCH=pathlib.Path('mvrecon/archive/pre_pose_fix')
old=json.loads((ARCH/'task.json').read_text())
oldrows=json.loads((ARCH/'baseline_table.json').read_text())

def rotdeg(R):
 c=(np.trace(R)-1)/2
 return float(np.degrees(np.arccos(np.clip(c,-1,1))))

print(f"{'scene':<14}{'tag':<4}{'sid':<13}{'old JSON':>10}{'M vs A_bad':>12}{'M vs T_good':>14}")
for scene in sorted(old):
    poses={x['sid']:x for x in json.loads((pathlib.Path('mvrecon/eval_data')/scene/'meta.json').read_text())['images']}
    for tag in ('v2','v3','v4'):
        t=old[scene].get(tag)
        if not t: continue
        ref=t['ref'];
        Rr,tr=np.asarray(poses[ref]['R']),np.asarray(poses[ref]['t'])
        for sid in t['ids'][1:]:
            D=np.asarray(t['D_injected'][sid]); Tb=np.asarray(t['T_true'][sid])
            A_bad=Tb@np.linalg.inv(D)
            bad=rotdeg(D[:3,:3]@A_bad[:3,:3].T)
            Rj,tj=np.asarray(poses[sid]['R']),np.asarray(poses[sid]['t'])
            Tg=np.eye(4); Tg[:3,:3]=Rr@Rj.T; Tg[:3,3]=tr-Rr@Rj.T@tj
            good=rotdeg(D[:3,:3]@Tg[:3,:3].T)
            saved=[x['rot_err'] for x in oldrows if x.get('scene')==scene and x.get('tag')==tag and x.get('method')=='identity']
            if saved:
                print(f"{scene:<14}{tag:<4}{sid:<13}{saved[0]:>10.1f}{bad:>12.1f}{good:>14.1f}")
