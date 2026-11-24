"""Does the ETH3D estimate depend on how many landmarks an image has?"""
import numpy as np, json, pathlib, glob
S=[];R=[];SG=[];RT=[];N=[];RA=[]
for f in ['work/loo_courtyard_facade.npz','work/loo_delivery_area_electro.npz','work/loo_office_playground_terrace.npz']:
    z=np.load(f,allow_pickle=True)
    S+=list(z['s_lm']);R+=list(z['raw']);SG+=list(z['s_gt']);RT+=list(z['scenes'])
    RA+=list(z['ratios']); SID=list(z['sids'])
S,R,SG=np.array(S,float),np.array(R,float),np.array(SG,float)
# landmark counts live in the M_/m_ json, keyed by sid -> rebuild aligned
cnt={}
for f in glob.glob("work/m_*.json")+glob.glob("work/M_*.json"):
    for x in json.loads(pathlib.Path(f).read_text()):
        if x.get("ok",True): cnt[x["sid"]]=x["n"]
sids=[]
for f in ['work/loo_courtyard_facade.npz','work/loo_delivery_area_electro.npz','work/loo_office_playground_terrace.npz']:
    z=np.load(f,allow_pickle=True); sids+=list(z['sids'])
N=np.array([cnt.get(s,0) for s in sids],float)
C=0.952
cal=np.array([float(np.mean(np.abs((S[i]/C)*RA[i]-1.0))) for i in range(len(S))])
print(f"=== ETH3D: does accuracy depend on landmark count? (n={len(S)}) ===")
print(f"{'count bin':<16}{'n':>4}{'raw':>9}{'cal':>9}{'wins':>8}")
for lo,hi in ((0,1000),(1000,2500),(2500,4000),(4000,99999)):
    ids=(N>=lo)&(N<hi)
    if ids.sum()<3: continue
    w=int((cal[ids]<R[ids]).sum())
    print(f"{f'{lo}-{hi}':<16}{int(ids.sum()):>4}{np.median(R[ids]):>9.4f}"
          f"{np.median(cal[ids]):>9.4f}{w:>5}/{int(ids.sum()):<3}")
print()
print(f"corr(landmark count, |s_lm/s_gt - 0.952|) = "
      f"{np.corrcoef(N, np.abs(S/SG-0.952))[0,1]:+.3f}")
print("Near zero means the estimate is as good with 500 points as with 5000, which")
print("matches the threshold sweep in section 23.")
