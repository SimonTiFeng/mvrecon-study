import numpy as np
from collections import defaultdict
files=['work/loo_courtyard_facade.npz','work/loo_delivery_area_electro.npz','work/loo_office_playground_terrace.npz']
S=[];R=[];SG=[];RT=[]
for f in files:
    z=np.load(f,allow_pickle=True)
    S+=list(z['s_lm']);R+=list(z['raw']);SG+=list(z['s_gt']);RT+=list(z['scenes'])
S,R,SG=np.array(S,float),np.array(R,float),np.array(SG,float)
sc=defaultdict(list)
for i,s in enumerate(RT): sc[s].append(i)
print(f"{'scene':<15}{'n':>4}{'raw':>9}{'s_lm':>8}{'s_gt':>8}{'ratio':>8}{'cal':>8}")
for s,ids in sorted(sc.items()):
    ids=np.array(ids)
    fit=[S[j]/SG[j] for j in range(len(S)) if RT[j]!=s]
    c=float(np.median(fit))
    cal=np.median([float(np.mean(np.abs((S[i]/c)*np.load(files[min(0,0)])['ratios'][0] - 1))) for i in ids[:0]] or [0])
    print(f"{s:<15}{len(ids):>4}{np.median(R[ids]):>9.4f}{np.median(S[ids]):>8.4f}"
          f"{np.median(SG[ids]):>8.4f}{np.median(S[ids]/SG[ids]):>8.3f}")
print(f"\nTOTAL {len(S)}")
