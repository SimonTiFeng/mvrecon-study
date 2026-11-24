import json, pathlib, numpy as np
from collections import defaultdict
o=json.loads(pathlib.Path("work/kitti_full_eval.json").read_text())
by=defaultdict(list)
for x in o: by[x["scene"]].append(x)
print(f"{'drive':<26}{'n':>4}{'raw':>9}{'cal':>9}{'wins':>8}{'s_l':>7}{'s_gt':>7}{'delta':>9}")
raws=[];cals=[];wins=0
for sc in sorted(by):
    sub=by[sc]; rw=[x["raw"] for x in sub]; cl=[x["cal"] for x in sub]
    w=sum(1 for a,b in zip(cl,rw) if a<b)
    print(f"{sc:<26}{len(sub):>4}{np.median(rw):>9.4f}{np.median(cl):>9.4f}"
          f"{w:>5}/{len(sub):<3}{np.median([x['s_l'] for x in sub]):>7.3f}"
          f"{np.median([x['s_gt'] for x in sub]):>7.3f}{np.median(cl)-np.median(rw):>+9.4f}")
    raws+=rw; cals+=cl; wins+=w
print(f"\nTOTAL {len(o)}: {np.median(raws):.4f} -> {np.median(cals):.4f}  wins {wins}/{len(o)} = {wins/len(o)*100:.1f}%")
bad=[sc for sc in by if sum(1 for a,b in zip([x["cal"] for x in by[sc]],[x["raw"] for x in by[sc]]) if a<b)/len(by[sc])<0.7]
print(f"drives below 70% win rate: {bad if bad else 'none'}")
