"""How does baseline length affect the pose-derived scale's precision?"""
import numpy as np, itertools
# courtyard baselines measured earlier (metres)
pairs=[("0286","0287",0.788,0.754),("0286","0288",1.387,1.338),("0286","0289",1.982,1.908),
       ("0287","0288",0.601,0.585),("0287","0289",1.194,1.155),("0288","0289",0.596,0.571)]
print("=== per-pair scale, sorted by baseline length ===")
print(f"{'pair':<12}{'baseline m':>12}{'ratio':>8}")
for a,b,p,g in sorted(pairs,key=lambda t:t[2]):
    print(f"{a+'-'+b:<12}{p:>12.3f}{g/p:>8.4f}")
r=np.array([g/p for a,b,p,g in pairs])
bl=np.array([p for a,b,p,g in pairs])
print()
print(f"all pairs: median {np.median(r):.4f}  sd {r.std():.4f}")
short=[x for x,p in zip(r,bl) if p<0.8]
long=[x for x,p in zip(r,bl) if p>1.2]
print(f"short baselines (<0.8 m): median {np.median(short):.4f}  sd {np.std(short):.4f}  n={len(short)}")
print(f"long  baselines (>1.2 m): median {np.median(long):.4f}  sd {np.std(long):.4f}  n={len(long)}")
print()
print("Design guidance: a longer baseline between the views used for calibration gives")
print("a more precise scale, because a fixed relative pose error translates to a")
print("smaller fractional error in a longer distance. For a new capture, choose the two")
print("views furthest apart.")
