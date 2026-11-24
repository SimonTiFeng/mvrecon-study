"""How much headroom is left between the achieved number and the per-image optimum?"""
import numpy as np
files=['work/loo_courtyard_facade.npz','work/loo_delivery_area_electro.npz','work/loo_office_playground_terrace.npz']
S=[];R=[];SG=[];RT=[];RA=[]
for f in files:
    z=np.load(f,allow_pickle=True)
    S+=list(z['s_lm']);R+=list(z['raw']);SG+=list(z['s_gt']);RT+=list(z['scenes'])
    RA+=list(z['ratios'])
S,R,SG=np.array(S,float),np.array(R,float),np.array(SG,float)
scenes=sorted(set(RT))
print("=== achieved vs per-image optimum (using each image's OWN s_gt) ===")
ach=[];opt=[];raw=[]
for i in range(len(S)):
    rt=RA[i]
    ach.append(float(np.mean(np.abs(S[i]*rt-1.0))))
    opt.append(float(np.mean(np.abs(SG[i]*rt-1.0))))
    raw.append(R[i])
ach=np.array(ach);opt=np.array(opt);raw=np.array(raw)
print(f"  raw            {np.median(raw):.4f}")
print(f"  + landmark s   {np.median(ach):.4f}")
print(f"  per-image oracle {np.median(opt):.4f}")
print(f"  gap achieved->oracle: {np.median(ach)-np.median(opt):+.4f}")
print()
print("=== How far are the achieved scales from each image's own optimum? ===")
err=np.array([abs(S[i]-SG[i])/SG[i] for i in range(len(S))])
print(f"  |s_lm - s_gt| / s_gt: median {np.median(err)*100:.2f}%  p90 {np.percentile(err,90)*100:.2f}%")
print(f"  images within 5%  : {int((err<0.05).sum())}/{len(err)}")
print(f"  images within 10% : {int((err<0.10).sum())}/{len(err)}")
print()
print("Reading: if the gap to oracle is small, the estimator is near-optimal and the")
print("remaining error is the model's own shape error, which no scale can fix.")
