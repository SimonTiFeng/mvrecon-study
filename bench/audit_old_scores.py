"""Forensic reproduction of the user's office-v2 mismatch using the archived pre-fix task."""
import json, pathlib, numpy as np
ARCH=pathlib.Path('mvrecon/archive/pre_pose_fix')
old=json.loads((ARCH/'task.json').read_text())
scene='office'; tag='v2'; t=old[scene][tag]; sid=t['ids'][1]; ref=t['ref']

def rotdeg(R):
    c=(np.trace(R)-1)/2
    return float(np.degrees(np.arccos(np.clip(c,-1,1))))
D=np.asarray(t['D_injected'][sid]); Told=np.asarray(t['T_true'][sid])
# current/old table's identity score was cumulative D against A=T_old D^-1
Aold=Told@np.linalg.inv(D)
oldscore=rotdeg(D[:3,:3]@np.linalg.inv(Aold[:3,:3]))
# Correct T from COLMAP world->camera poses
meta=json.loads(pathlib.Path(f'mvrecon/eval_data/{scene}/meta.json').read_text())
poses={x['sid']:x for x in meta['images']}
ri=poses[ref]; rj=poses[sid]
Ri,ti=np.asarray(ri['R']),np.asarray(ri['t'])
Rj,tj=np.asarray(rj['R']),np.asarray(rj['t'])
T=np.eye(4); T[:3,:3]=Ri@Rj.T; T[:3,3]=ti-Ri@Rj.T@tj
correctscore=rotdeg(D[:3,:3]@T[:3,:3].T)
# What was in the old baseline table
bt=json.loads((ARCH/'baseline_table.json').read_text())
found=[x for x in bt if x.get('scene')==scene and x.get('tag')==tag and x.get('method')=='identity']
print('office v2 target',sid,'ref',ref)
print('old score formula (M vs A_old):',round(oldscore,2))
print('recomputed old T_bad direct:',round(rotdeg(D[:3,:3]@Told[:3,:3].T),2))
print('correct COLMAP T formula, D vs T:',round(correctscore,2))
print('published baseline_table rows:',[(x.get('rot_med'),x.get('rot_err'),x) for x in found])
print('D trans',np.round(D[:3,3],3),'|D|',round(np.linalg.norm(D[:3,3]),3))
print('T_bad trans',np.round(Told[:3,3],3),'|Tbad|',round(np.linalg.norm(Told[:3,3]),3))
print('T_good trans',np.round(T[:3,3],3),'|Tgood|',round(np.linalg.norm(T[:3,3]),3))
