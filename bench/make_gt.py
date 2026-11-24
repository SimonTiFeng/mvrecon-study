"""Build the evaluation-side ground truth: relative transforms between views.

For views i,j with world->cam (Ri,ti),(Rj,tj), COLMAP uses
    p_cam = R p_world + t.
The transform taking view j's camera frame INTO view i's camera frame is therefore:
    R_ij = Ri Rj^T,   t_ij = ti - Ri Rj^T tj.
This is what a registration method must recover.
"""
import json, pathlib, numpy as np

EV = pathlib.Path('mvrecon/eval_data')
subs = json.loads(pathlib.Path('mvrecon/bench/subsets.json').read_text())
out={}
for scene, s in subs.items():
    ev = json.loads((EV/scene/'meta.json').read_text())
    pose = {x['sid']: x for x in ev['images']}
    for tag in ('v2','v3','v4'):
        ids = s[tag]
        if not all(i in pose for i in ids): continue
        ref = ids[0]
        Rr = np.asarray(pose[ref]['R']); tr = np.asarray(pose[ref]['t'])
        T = {}
        for sid in ids:
            R = np.asarray(pose[sid]['R']); t = np.asarray(pose[sid]['t'])
            # COLMAP pose is world->camera. Invert view j to world, then map to ref.
            Rij = Rr @ R.T
            tij = tr - Rr @ R.T @ t
            M = np.eye(4); M[:3,:3]=Rij; M[:3,3]=tij
            T[sid] = M
            rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(Rij)-1)/2,-1,1))))
            trans = float(np.linalg.norm(tij))
        out.setdefault(scene,{})[tag]={'ref':ref,'ids':ids,'T':{k:v.tolist() for k,v in T.items()},
                                       'ref_rot_deg':rot_deg,'ref_trans_m':trans}
print(f"{'scene':<15}{'tag':<5}{'ref':<13}{'max_rot_deg':>12}{'max_trans_m':>12}")
for scene in sorted(out):
    for tag in ('v2','v3','v4'):
        if tag not in out[scene]: continue
        d=out[scene][tag]
        rots=[float(np.degrees(np.arccos(np.clip((np.trace(np.asarray(m)[:3,:3])-1)/2,-1,1)))) for m in d['T'].values()]
        trs=[float(np.linalg.norm(np.asarray(m)[:3,3])) for m in d['T'].values()]
        print(f"{scene:<15}{tag:<5}{d['ref']:<13}{max(rots):>12.1f}{max(trs):>12.2f}")
pathlib.Path('mvrecon/eval_data/gt_transforms.json').write_text(json.dumps(out,indent=1))
print('\nwrote mvrecon/eval_data/gt_transforms.json')
