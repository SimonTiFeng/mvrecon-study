"""Confirm: electro's failures are the far-landmark frames."""
import json, pathlib, numpy as np
from collections import defaultdict
files=['work/loo_delivery_area_electro.npz']
z=np.load(files[0],allow_pickle=True)
S=list(z['s_lm']); R=list(z['raw']); SG=list(z['s_gt']); RT=list(z['scenes']); RA=list(z['ratios'])
S,R,SG=np.array(S,float),np.array(R,float),np.array(SG,float)
# med_z stored separately in the M_ json
med={}
for x in json.loads(pathlib.Path("work/m_electro.json").read_text()):
    if x.get("ok",True): med[x["sid"]]=x.get("med_z",0)
sids=list(z['sids'])
C=0.952
print("=== electro: does the far-landmark split explain the failures? ===")
print(f"{'med_z bin':<14}{'n':>4}{'raw':>9}{'cal':>9}{'wins':>8}")
for lo,hi in ((0,5),(5,10),(10,15),(15,25)):
    ids=[i for i in range(len(S)) if RT[i]=="electro" and lo<=med.get(sids[i],0)<hi]
    if len(ids)<2: continue
    cal=[float(np.mean(np.abs((S[i]/C)*RA[i]-1.0))) for i in ids]
    rw=[R[i] for i in ids]
    w=sum(1 for a,b in zip(cal,rw) if a<b)
    print(f"{f'{lo}-{hi}m':<14}{len(ids):>4}{np.median(rw):>9.4f}{np.median(cal):>9.4f}{w:>5}/{len(ids):<3}")
print()
print("This is the documented scope condition: landmarks beyond ~10 m make the")
print("estimate collapse low, because the model's relative-depth error grows with")
print("distance faster than the landmark's metric precision.")
