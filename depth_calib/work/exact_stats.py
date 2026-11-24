"""Exact paired bootstrap for ETH3D using the persisted per-image ratio vectors."""
import numpy as np, pathlib
rng = np.random.default_rng(0)
files = ["work/loo_courtyard_facade.npz",
         "work/loo_delivery_area_electro.npz",
         "work/loo_office_playground_terrace.npz"]
S, R, SG, RA, RT = [], [], [], [], []
for f in files:
    z = np.load(f, allow_pickle=True)
    S += list(z["s_lm"]); R += list(z["raw"]); SG += list(z["s_gt"])
    RA += list(z["ratios"]); RT += list(z["scenes"])
S, R, SG = np.array(S), np.array(R), np.array(SG)
print(f"loaded {len(S)} images from {len(files)} batches")
# per-scene leave-one-out constant, in the same grouping used for the headline
scenes = sorted(set(RT))
deltas = []
raws = []
for i in range(len(S)):
    hold = RT[i]
    fit = [S[j]/SG[j] for j in range(len(S)) if RT[j] != hold]
    c = float(np.median(fit)) if fit else 1.0
    rt = RA[i]
    cal = float(np.mean(np.abs((S[i]/c)*rt - 1.0)))
    deltas.append(cal - R[i])
    raws.append(R[i])
deltas = np.array(deltas); raws = np.array(raws)
cal_all = raws + deltas
print(f"raw {np.median(raws):.4f} -> calibrated {np.median(cal_all):.4f}")
print(f"wins {int((deltas<0).sum())}/{len(deltas)} = {(deltas<0).mean()*100:.1f}%")
idx = rng.integers(0, deltas.size, (8000, deltas.size))
b = np.median(deltas[idx], axis=1)
m, lo, hi = np.median(deltas), np.percentile(b, 2.5), np.percentile(b, 97.5)
print(f"median paired delta {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
print(f"CI excludes 0: {'YES' if hi < 0 else 'NO'}")
