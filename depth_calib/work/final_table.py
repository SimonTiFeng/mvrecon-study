"""Produce the per-scene table the paper needs, with fixed-0.952 (label-free)."""
import numpy as np
from collections import defaultdict
files=['work/loo_courtyard_facade.npz','work/loo_delivery_area_electro.npz','work/loo_office_playground_terrace.npz']
S=[];R=[];SG=[];RT=[];RA=[]
for f in files:
    z=np.load(f,allow_pickle=True)
    S+=list(z['s_lm']);R+=list(z['raw']);SG+=list(z['s_gt']);RT+=list(z['scenes']);RA+=list(z['ratios'])
S,R,SG=np.array(S,float),np.array(R,float),np.array(SG,float)
sc=defaultdict(list)
for i,s in enumerate(RT): sc[s].append(i)
C=0.952
print("=== ETH3D, label-free (fixed reference constant 0.952) ===")
print(f"{'scene':<15}{'n':>4}{'raw':>9}{'cal':>9}{'wins':>8}{'delta':>9}")
raws=[];cals=[];wins=0
for s in sorted(sc):
    ids=sc[s]
    cal=[float(np.mean(np.abs((S[i]/C)*RA[i]-1.0))) for i in ids]
    rw=[R[i] for i in ids]
    w=sum(1 for a,b in zip(cal,rw) if a<b)
    print(f"{s:<15}{len(ids):>4}{np.median(rw):>9.4f}{np.median(cal):>9.4f}"
          f"{w:>5}/{len(ids):<3}{np.median(cal)-np.median(rw):>+9.4f}")
    raws+=rw; cals+=cal; wins+=w
print(f"\nTOTAL {len(S)} images: raw {np.median(raws):.4f} -> {np.median(cals):.4f}"
      f"   wins {wins}/{len(S)} = {wins/len(S)*100:.1f}%")
print(f"published DA3-metric ETH3D = 0.104  ->  {(1-np.median(cals)/0.104)*100:.0f}% lower")
