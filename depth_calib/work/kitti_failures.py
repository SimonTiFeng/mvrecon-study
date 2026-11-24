"""Which KITTI frames fail, and do they share a property?"""
import json, pathlib, numpy as np
o = json.loads(pathlib.Path("work/kitti_full_eval.json").read_text())
fails = [x for x in o if x["cal"] >= x["raw"]]
good  = [x for x in o if x["cal"] < x["raw"]]
print(f"total {len(o)}  good {len(good)}  fail {len(fails)}")
print()
def stat(x): return (x["s_l"], x["s_gt"], x["n_pts"], x["raw"])
print(f"{'group':<8}{'n':>4}{'s_l':>8}{'s_gt':>8}{'ratio':>8}{'n_pts':>9}{'raw':>9}")
for nm, g in (("good", good), ("fail", fails)):
    if not g: continue
    sl=np.array([x["s_l"] for x in g]); sg=np.array([x["s_gt"] for x in g])
    nn=np.array([x["n_pts"] for x in g]); rr=np.array([x["raw"] for x in g])
    print(f"{nm:<8}{len(g):>4}{np.median(sl):>8.4f}{np.median(sg):>8.4f}"
          f"{np.median(sl/sg):>8.4f}{np.median(nn):>9}{np.median(rr):>9.4f}")
print()
print("Failure breakdown:")
print(f"  ratio (s_l/s_gt) > 1.05 : {sum(1 for x in fails if x['s_l']/x['s_gt']>1.05)}")
print(f"  ratio < 0.95            : {sum(1 for x in fails if x['s_l']/x['s_gt']<0.95)}")
print(f"  n_pts < 5000            : {sum(1 for x in fails if x['n_pts']<5000)}")
print()
print("If failures cluster at extreme ratios, the estimator is wrong there.")
print("If they cluster at low n_pts, it is a data-coverage issue.")
print()
by_drive={}
for x in fails: by_drive.setdefault(x["scene"], []).append(x)
print("failures per drive:")
for d, v in sorted(by_drive.items()):
    print(f"  {d:<26} {len(v):>3}")
