import json,pathlib,numpy as np
T=json.load(open('mvrecon/eval_data/task.json'))
maxerr=0;n=0
for scene,st in T.items():
    poses={x['sid']:x for x in json.load(open(f'mvrecon/eval_data/{scene}/meta.json'))['images']}
    for tag,d in st.items():
        ref=d['ref']; Rr,tr=np.asarray(poses[ref]['R']),np.asarray(poses[ref]['t'])
        for sid in d['ids']:
            Rj,tj=np.asarray(poses[sid]['R']),np.asarray(poses[sid]['t'])
            E=np.eye(4); E[:3,:3]=Rr@Rj.T; E[:3,3]=tr-Rr@Rj.T@tj
            err=float(np.max(np.abs(E-np.asarray(d['T_true'][sid]))))
            maxerr=max(maxerr,err); n+=1
print('current task T_true vs independently recomputed COLMAP w2c formula:')
print('  transforms',n,'max abs error',maxerr)
print('  pass',maxerr<1e-8)
