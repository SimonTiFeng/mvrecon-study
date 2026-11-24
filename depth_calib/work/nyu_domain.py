"""NYUv2 as an indoor domain check: how biased is DA3-metric indoors?"""
import json, pathlib, numpy as np, cv2
C=pathlib.Path("cache/hpc/depths")
rows={json.loads(l)["sample_id"]:json.loads(l)
      for l in open("manifests/formal_all.jsonl") if l.strip()}
def ab(z,g): return float(np.mean(np.abs(z-g)/g))
def gtopt(p,g):
    if p.size>6000:
        i=np.linspace(0,p.size-1,6000).astype(np.int64); p,g=p[i],g[i]
    r=p/g; best=(1.0,np.inf)
    for s in np.arange(0.6,1.6,0.01):
        v=float(np.mean(np.abs(s*r-1.0)))
        if v<best[1]: best=(float(s),v)
    return best[0]
sc=[];raw=[]
n=0
for sid,r in rows.items():
    if r["dataset"]!="nyuv2": continue
    npz=C/f"{sid}__B.npz"
    if not npz.exists(): continue
    sd=json.loads((C/f"{sid}__B.json").read_text()); Hp,Wp=sd["processed_shape"]
    K=np.asarray(r["K"],float)
    d=np.load(npz)["depth"].astype(np.float64)
    root=pathlib.Path(r["dataset_root"])
    gp=np.load(root/r["gt"]).astype(np.float64); mk=np.load(root/r["mask"]).astype(bool)
    f=((K[0,0]*Wp/r["width"])+(K[1,1]*Hp/r["height"]))/2/300.0
    d2=cv2.resize(d,(r["width"],r["height"]))*f
    m=mk&np.isfinite(d2)&(d2>0)&np.isfinite(gp)&(gp>0)
    if m.sum()<500: continue
    p,g=d2[m],gp[m]
    sc.append(gtopt(p,g)); raw.append(ab(p,g)); n+=1
sc=np.array(sc); raw=np.array(raw)
print(f"=== NYUv2 indoor, n={n} ===")
print(f"  GT-optimal scale: median {np.median(sc):.4f}  sd {sc.std():.4f}  "
      f"range [{np.percentile(sc,5):.3f},{np.percentile(sc,95):.3f}]")
print(f"  raw AbsRel      : median {np.median(raw):.4f}")
print(f"  published DA3-metric NYUv2 = 0.070")
print()
print("Compare with the outdoor datasets:")
print("  ETH3D scene optima 0.930-1.184   |  KITTI drive optima 0.997-1.256")
print(f"  NYUv2 image optima median {np.median(sc):.4f}")
print()
print("NYUv2 has no SfM/LiDAR reference, so the estimator cannot be applied there.")
print("This is reported as the domain where the method is NOT available, and the")
print("scale bias distribution shows whether that limitation matters.")
