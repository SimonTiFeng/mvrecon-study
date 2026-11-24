"""Formal evaluation on the complete cache: Metric raw vs Mono oracles, by depth band."""
from __future__ import annotations
import json, pathlib, sys
import numpy as np
from scipy.optimize import minimize
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config
cfg = load_config(ROOT / "configs" / "pilot.yaml")
cache = pathlib.Path("D:/depth_calib_cache/formal")
rows = [json.loads(l) for l in (ROOT/"manifests/formal_all.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
BANDS = [(0,0.5),(0.5,1.5),(1.5,3),(3,5),(5,float("inf"))]
NAMES = ["<0.5","0.5-1.5","1.5-3","3-5",">=5"]
def ar(z,g): return float(np.mean(np.abs(z-g)/g))
def fit_scale(p,g):
    from scipy.optimize import minimize_scalar
    r=minimize_scalar(lambda a: ar(a*p,g) if a>0 else 1e9, bounds=(1e-4,1e4), method="bounded",
                      options={"xatol":1e-8})
    return float(r.x), float(r.fun)
def fit_affine(p,g):
    med=float(np.median(p))
    def loss(t):
        a=float(np.exp(t[0])); b=float(t[1]*med); z=a*p+b
        return 10.0 if np.any(z<=0) else ar(z,g)
    best=None
    for la in (0.0,np.log(3.0)):
        for bh in (-1.,0.,1.):
            r=minimize(loss,np.array([la,bh]),method="L-BFGS-B",
                       bounds=[(np.log(1e-4),np.log(1e4)),(-20,20)],options={"maxiter":150})
            if best is None or r.fun<best.fun: best=r
    return float(np.exp(best.x[0])),float(best.x[1]*med),float(best.fun)
import argparse
ap=argparse.ArgumentParser(); ap.add_argument("--start",type=int,default=0); ap.add_argument("--count",type=int,default=200)
ap.add_argument("--dataset",default=None)
ap.add_argument("--tag",default="part"); A=ap.parse_args()
out=[]
rows = [r for r in rows if A.dataset is None or r["dataset"]==A.dataset]
for row in rows[A.start:A.start+A.count]:
    sid=row["sample_id"]; fa=cache/f"{sid}__A.npz"; fb=cache/f"{sid}__B.npz"
    if not (fa.exists() and fb.exists()): continue
    try:
        gt=np.load(pathlib.Path(row["dataset_root"])/row["gt"]).astype(np.float64)
        mask=np.load(pathlib.Path(row["dataset_root"])/row["mask"]).astype(bool)
        dA=np.load(fa)["depth"].astype(np.float64); dB=np.load(fb)["depth"].astype(np.float64)
        mB=json.loads((cache/f"{sid}__B.json").read_text(encoding="utf-8"))
    except Exception: continue
    K=np.asarray(row["K"],float); Hp,Wp=mB["processed_shape"]
    dB*=((K[0,0]*Wp/row["width"])+(K[1,1]*Hp/row["height"]))/2/300.0
    m=mask&np.isfinite(gt)&(gt>0)&np.isfinite(dA)&(dA>0)&np.isfinite(dB)&(dB>0)
    if m.sum()<100: continue
    g=gt[m]; pA=dA[m]; pB=dB[m]
    s,so=fit_scale(pA,g); a,b,ao=fit_affine(pA,g)
    rec={"sample_id":sid,"dataset":row["dataset"],"scene_id":row["scene_id"],
         "n_pix":int(g.size),"median":float(np.median(g)),
         "B_raw":ar(pB,g),"A_scale_oracle":so,"A_affine_oracle":ao,
         "B_raw_band":{},"A_aff_band":{}}
    for nm,(lo,hi) in zip(NAMES,BANDS):
        sel=(g>=lo)&(g<hi) if hi!=float("inf") else (g>=lo)
        if sel.sum()>=100:
            rec["B_raw_band"][nm]=ar(pB[sel],g[sel])
            rec["A_aff_band"][nm]=ar((a*pA+b)[sel],g[sel])
    out.append(rec)
dst=ROOT/f"results/formal_eval_{A.tag}.json"
dst.write_text(json.dumps(out,indent=1),encoding="utf-8")
print("wrote",dst,"n",len(out))
import sys; sys.exit(0)
def med(v):
    v=[x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")
print(f"images={len(out)} pixels={sum(r['n_pix'] for r in out):,}")
for ds in sorted({r['dataset'] for r in out}):
    sub=[r for r in out if r['dataset']==ds]
    print(f"\n== {ds}: images={len(sub)} scenes={len({r['scene_id'] for r in sub})} pixels={sum(r['n_pix'] for r in sub):,}")
    print(f"   Metric raw          {med([r['B_raw'] for r in sub]):.4f}")
    print(f"   Mono scale oracle   {med([r['A_scale_oracle'] for r in sub]):.4f}")
    print(f"   Mono affine oracle  {med([r['A_affine_oracle'] for r in sub]):.4f}")
    print("   band        Metric    Mono-affine   n_images")
    for nm in NAMES:
        bb=[r['B_raw_band'].get(nm) for r in sub]; aa=[r['A_aff_band'].get(nm) for r in sub]
        n=sum(1 for x in bb if x is not None)
        print(f"   {nm:<10s} {med(bb):8.4f} {med(aa):12.4f} {n:8d}")
