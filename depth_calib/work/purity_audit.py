"""Audit: which of our numbers actually need no GT?"""
import numpy as np
files=['work/loo_courtyard_facade.npz','work/loo_delivery_area_electro.npz','work/loo_office_playground_terrace.npz']
S=[];R=[];SG=[];RT=[];RA=[]
for f in files:
    z=np.load(f,allow_pickle=True)
    S+=list(z['s_lm']);R+=list(z['raw']);SG+=list(z['s_gt']);RT+=list(z['scenes']);RA+=list(z['ratios'])
S,R,SG=np.array(S,float),np.array(R,float),np.array(SG,float)
scenes=sorted(set(RT))
def sc(i,v): return float(np.mean(np.abs(v*RA[i]-1.0)))
print("=== GT PURITY AUDIT ===")
print()
print("A. zero-parameter (uses ONLY per-image landmark s, no GT anywhere):")
v=[sc(i,S[i]) for i in range(len(S))]
print(f"   {np.median(R):.4f} -> {np.median(v):.4f}   wins {sum(1 for a,b in zip(v,R) if a<b)}/{len(v)}")
print()
print("B. cross-scene constant (constant fitted from OTHER scenes' s_lm/s_gt):")
def cross(i):
    fit=[S[j]/SG[j] for j in range(len(S)) if RT[j]!=RT[i]]
    return float(np.median(fit))
v2=[sc(i,S[i]/cross(i)) for i in range(len(S))]
print(f"   {np.median(R):.4f} -> {np.median(v2):.4f}   wins {sum(1 for a,b in zip(v2,R) if a<b)}/{len(v2)}")
print("   ** uses GT OPTIMA of other scenes -> NOT label-free **")
print()
print("C. fixed 0.952 (a round number, no GT at all):")
v3=[sc(i,S[i]/0.952) for i in range(len(S))]
print(f"   {np.median(R):.4f} -> {np.median(v3):.4f}   wins {sum(1 for a,b in zip(v3,R) if a<b)}/{len(v3)}")
print("   ** this IS label-free, if 0.952 is defensible as a reference constant **")
print()
print("=== KITTI for contrast ===")
print("   KITTI needs no constant (s_l/s_gt=1.005), so its 0.0668 is fully label-free.")
