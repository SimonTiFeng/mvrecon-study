import json,pathlib,numpy as np

def rotdeg(R):
 c=(np.trace(R)-1)/2
 return float(np.degrees(np.arccos(np.clip(c,-1,1))))
for file in ('mvrecon/archive/pre_pose_fix/task.json','mvrecon/eval_data/task.json'):
 t=json.loads(pathlib.Path(file).read_text())['office']['v2']; sid=t['ids'][1]
 D=np.asarray(t['D_injected'][sid]); T=np.asarray(t['T_true'][sid])
 A=T@np.linalg.inv(D)
 print(file,'sid',sid)
 print('  D vs T rot',rotdeg(D[:3,:3]@T[:3,:3].T))
 print('  old score M vs A rot',rotdeg(D[:3,:3]@A[:3,:3].T))
 print('  T trans',np.round(T[:3,3],3),'D trans',np.round(D[:3,3],3))
 # proper transform from metadata
 poses={x['sid']:x for x in json.loads(pathlib.Path('mvrecon/eval_data/office/meta.json').read_text())['images']}
 ref=t['ref']; Ri,ti=np.asarray(poses[ref]['R']),np.asarray(poses[ref]['t']); Rj,tj=np.asarray(poses[sid]['R']),np.asarray(poses[sid]['t'])
 Tg=np.eye(4); Tg[:3,:3]=Ri@Rj.T; Tg[:3,3]=ti-Ri@Rj.T@tj
 print('  proper T rot',rotdeg(D[:3,:3]@Tg[:3,:3].T),'Tgood trans',np.round(Tg[:3,3],3))
