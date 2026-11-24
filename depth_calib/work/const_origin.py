"""Is the 0.952 constant scene-dependent (model property) or fixed (SfM property)?"""
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
ratios=np.array([np.median(S[sc[s]]/SG[sc[s]]) for s in sorted(sc)])
raws=np.array([np.median(R[sc[s]]) for s in sorted(sc)])
sgt=np.array([np.median(SG[sc[s]]) for s in sorted(sc)])
names=sorted(sc)
print("=== Test A: is the ratio independent of how biased the model is? ===")
print("If the constant is a fixed SfM property, it should be uncorrelated with s_gt.")
print(f"  corr(ratio, s_gt) = {np.corrcoef(ratios, sgt)[0,1]:+.3f}")
print(f"  corr(ratio, raw AbsRel) = {np.corrcoef(ratios, raws)[0,1]:+.3f}")
print()
print("=== Test B: spread of the ratio ===")
print(f"  ratios: {[round(float(r),3) for r in ratios]}")
print(f"  median {np.median(ratios):.4f}  sd {ratios.std():.4f}  "
      f"range [{ratios.min():.4f}, {ratios.max():.4f}]")
print(f"  coefficient of variation: {ratios.std()/ratios.mean()*100:.1f}%")
print()
print("=== Test C: how much better is s_lm than a fixed 0.95 constant? ===")
for use_med in (True, False):
    c = float(np.median(ratios)) if use_med else 0.95
    print(f"  constant = {c:.4f}" + ("  (median of measured ratios)" if use_med else "  (round number)"))
print()
print("Reading: a CV of a few percent means the constant is nearly fixed but not exactly;")
print("the honest description is 'a near-constant reference offset with scene-level drift'.")
