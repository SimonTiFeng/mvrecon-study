"""Select subsets with MEANINGFUL baselines AND sufficient overlap.

Reject near-duplicate views (baseline < 0.10 m) like office 0130/0135 which are 2 cm apart.
"""
import json, pathlib, numpy as np, collections
from scipy.spatial import cKDTree

AG=pathlib.Path('mvrecon/agent_data'); EV=pathlib.Path('mvrecon/eval_data')

def cloud_world(scene, sid, meta, pose, stride=32):
    d=np.load(AG/scene/f'{sid}_depth.npy').astype(np.float64)
    K=np.asarray(meta['K']); H,W=d.shape
    ys,xs=np.mgrid[0:H:stride,0:W:stride]
    z=d[ys,xs]; ok=(z>0)&np.isfinite(z)&(z<200)
    ys,xs,z=ys[ok],xs[ok],z[ok]
    fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
    P=np.stack([(xs-cx)/fx*z,(ys-cy)/fy*z,z],1)
    R=np.asarray(pose['R']); t=np.asarray(pose['t'])
    # COLMAP extrinsics are world->camera: X_cam = R X_world + t.
    # Invert to put this independently inferred cloud into world coordinates.
    return (P-t)@R

out={}
for scene in sorted(p.name for p in AG.iterdir() if p.is_dir()):
    ag=json.loads((AG/scene/'meta.json').read_text())
    ev=json.loads((EV/scene/'meta.json').read_text())
    pose={x['sid']:x for x in ev['images']}
    meta={x['sid']:x for x in ag['images']}
    sids=[s for s in sorted(meta) if s in pose]
    if len(sids)<3: continue
    clouds={s: cloud_world(scene,s,meta[s],pose[s]) for s in sids}
    cpos={s: -np.asarray(pose[s]['R']).T@np.asarray(pose[s]['t']) for s in sids}
    ov={}
    for i in range(len(sids)):
        for j in range(i+1,len(sids)):
            a,b=sids[i],sids[j]
            base=float(np.linalg.norm(cpos[a]-cpos[b]))
            if len(clouds[a])<40 or len(clouds[b])<40: continue
            d,_=cKDTree(clouds[a]).query(clouds[b],k=1)
            ov[(a,b)]=(float((d<0.35).mean()), base)
    def score(a,b):
        o,base=ov.get((a,b), ov.get((b,a),(0,0)))
        if base<0.10: return -1.0         # near-duplicate views rejected
        # want overlap in a useful band: not zero, not 1.0
        return o if 0.05<o<0.85 else -1.0
    best=None
    for i in range(len(sids)):
        for j in range(i+1,len(sids)):
            a,b=sids[i],sids[j]
            sc=score(a,b)
            if sc>0 and (best is None or sc>best[0]): best=(sc,a,b)
    if not best:
        print(f'{scene:<15} no usable pair'); continue
    _,a,b=best
    base=np.linalg.norm(cpos[a]-cpos[b])
    # grow to 3 and 4 by adding the view with best score against the existing set
    def grow(cur):
        cand=[s for s in sids if s not in cur]
        if not cand: return None
        def sc(s):
            vals=[score(s,x) for x in cur if score(s,x)>0]
            return min(vals) if vals else -1
        b2=max(cand,key=sc)
        return b2 if sc(b2)>0 else None
    v2=[a,b]; v3=v2+[x] if (x:=grow(v2)) else v2
    v4=v3+[x] if (x:=grow(v3)) else v3
    out[scene]={'v2':v2,'v3':v3,'v4':v4,
                'v2_overlap':round(best[0],2),'v2_baseline_m':round(float(base),3)}
    print(f'{scene:<15} ov {best[0]:.2f} base {base:.2f}m  v2={v2} v3={v3} v4={v4}')

pathlib.Path('mvrecon/bench/subsets.json').write_text(json.dumps(out,indent=1))
print('wrote subsets.json')
