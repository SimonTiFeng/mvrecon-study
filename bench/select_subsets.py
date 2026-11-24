"""Select 2/3/4-view subsets by MEASURED overlap in the reference frame.

Requirement: nested subsets (2 -> +3 -> +4), covering varied overlap, not just adjacent
frames. Selection may use eval-side geometry; the agent only receives the chosen images.
"""
import json, pathlib, numpy as np
import collections
from scipy.spatial import cKDTree

AG = pathlib.Path('mvrecon/agent_data'); EV = pathlib.Path('mvrecon/eval_data')
rows = {json.loads(l)['sample_id']: json.loads(l)
        for l in open('manifests/formal_all.jsonl') if l.strip()}

def pts_world(scene, sid, meta, pose):
    d = np.load(AG/scene/f'{sid}_depth.npy').astype(np.float64)
    K = np.asarray(meta['K']); H,W = d.shape
    ys, xs = np.mgrid[0:H:32, 0:W:32]
    z = d[ys,xs]; ok=(z>0)&np.isfinite(z)&(z<200)
    ys,xs,z = ys[ok],xs[ok],z[ok]
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
    X=(xs-cx)/fx*z; Y=(ys-cy)/fy*z
    p = np.stack([X,Y,z],1)
    R=np.asarray(pose['R']); t=np.asarray(pose['t'])
    return p @ R.T + t

def overlap(A,B,thr=0.35):
    if len(A)<50 or len(B)<50: return 0.0
    d,_=cKDTree(A).query(B,k=1)
    return float((d<thr).mean())

info={}
for scene in sorted(p.name for p in AG.iterdir() if p.is_dir()):
    ag=json.loads((AG/scene/'meta.json').read_text())
    ev=json.loads((EV/scene/'meta.json').read_text())
    pose={x['sid']:x for x in ev['images']}
    clouds={}
    for im in ag['images']:
        if im['sid'] in pose:
            clouds[im['sid']]=pts_world(scene, im['sid'], im, pose[im['sid']])
    sids=sorted(clouds)
    ov={}
    for i in range(len(sids)):
        for j in range(i+1,len(sids)):
            ov[(sids[i],sids[j])]=overlap(clouds[sids[i]],clouds[sids[j]])
    info[scene]={'sids':sids,'overlap':{f'{a}|{b}':v for (a,b),v in ov.items()}}

# choose the 2-view seed with the HIGHEST overlap, then grow by max added overlap
out={}
for scene,d in info.items():
    sids=d['sids']; ov=d['overlap']
    if len(sids)<2: continue
    def get(a,b): return ov.get(f'{a}|{b}', ov.get(f'{b}|{a}',0.0))
    best=None
    for i in range(len(sids)):
        for j in range(i+1,len(sids)):
            v=get(sids[i],sids[j])
            if best is None or v>best[0]: best=(v,sids[i],sids[j])
    seed2=[best[1],best[2]]
    # 3rd: maximise min-overlap to the existing pair
    cand=[s for s in sids if s not in seed2]
    s3=max(cand, key=lambda s: min(get(s,seed2[0]), get(s,seed2[1])) if cand else 0) if cand else None
    seed3=seed2+([s3] if s3 else [])
    cand=[s for s in sids if s not in seed3]
    s4=max(cand, key=lambda s: min([get(s,x) for x in seed3])) if cand else None
    seed4=seed3+([s4] if s4 else [])
    out[scene]={'v2':seed2,'v3':seed3,'v4':seed4,
                'seed_overlap':round(best[0],3)}
    print(f'{scene:<15} seed ov {best[0]:.2f}  v2={seed2} v3+={s3} v4+={s4}')

pathlib.Path('mvrecon/bench/subsets.json').write_text(json.dumps(out, indent=1))
print('wrote mvrecon/bench/subsets.json')
