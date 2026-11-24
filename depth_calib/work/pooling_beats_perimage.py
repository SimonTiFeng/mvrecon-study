"""Why does pooling beat per-image landmark scale? Quantify the noise/bias tradeoff."""
import numpy as np
files=['work/loo_courtyard_facade.npz','work/loo_delivery_area_electro.npz','work/loo_office_playground_terrace.npz']
S=[];R=[];SG=[];RT=[];RA=[]
for f in files:
    z=np.load(f,allow_pickle=True)
    S+=list(z['s_lm']);R+=list(z['raw']);SG+=list(z['s_gt']);RT+=list(z['scenes']);RA+=list(z['ratios'])
S,R,SG=np.array(S,float),np.array(R,float),np.array(SG,float)
scenes=sorted(set(RT))
idx={s:[i for i in range(len(S)) if RT[i]==s] for s in scenes}

def score(scale_fn):
    v=[]
    for i in range(len(S)):
        s=scale_fn(i); v.append(float(np.mean(np.abs(s*RA[i]-1.0))))
    return np.array(v)

print("=== Three pooling levels ===")
# 1. per-image own landmark scale
v_pi = score(lambda i: S[i])
# 2. same-scene median (leave-one-out within scene: excludes the image itself)
def scene_loo(i):
    o=[j for j in idx[RT[i]] if j!=i]
    return float(np.median([S[j] for j in o])) if o else S[i]
v_sl = score(scene_loo)
# 3. cross-scene constant (the headline)
def cross(i):
    fit=[S[j]/SG[j] for j in range(len(S)) if RT[j]!=RT[i]]
    return float(np.median(fit))
v_cs = score(lambda i: S[i]/cross(i))
print(f"  per-image landmark  {np.median(v_pi):.4f}   sd(log s) {np.log(S).std():.3f}")
print(f"  scene LOO median    {np.median(v_sl):.4f}")
print(f"  cross-scene const   {np.median(v_cs):.4f}")
print()
print("=== The noise/bias decomposition ===")
# how noisy is per-image s vs the scene mean?
noise=[];bias=[]
for s in scenes:
    a=np.array([S[i] for i in idx[s]])
    noise.append(float(np.log(a).std()))
    bias.append(float(np.log(np.median(a)) - np.log(np.median(SG[idx[s]]))))
print(f"  per-image noise (sd of log s within scene): median {np.median(noise):.3f}")
print(f"  scene bias (log s_lm - log s_gt)          : median {np.median(bias):.4f}")
print(f"  ratio noise/bias: {np.median(noise)/abs(np.median(bias)+1e-9):.1f}x")
print()
print("If noise >> bias, averaging within a scene removes noise while keeping the bias")
print("that the cross-scene constant is designed to correct. That is why pooling wins.")
