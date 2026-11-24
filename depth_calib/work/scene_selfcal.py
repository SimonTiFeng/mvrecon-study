"""Is the ETH3D 0.95 constant a per-scene SfM dof? If so it can be removed without GT."""
import json, pathlib, glob, numpy as np
print("=== The SfM scale argument ===")
print()
print("Each ETH3D scene is one SfM reconstruction, which is metric up to ONE scale.")
print("If the model were unbiased, the landmark estimator would return s_lm = k_scene")
print("(a constant per scene, the SfM scale error) and nothing else.")
print()
print("So the TESTABLE prediction is: within one scene, does s_lm have the SAME")
print("distribution as it would if the model were unbiased? That is, is the")
print("per-scene spread of s_lm much SMALLER than the spread predicted by model bias?")
print()
E = {}
for f in glob.glob("work/m_*.json") + glob.glob("work/M_*.json"):
    nm = pathlib.Path(f).stem.split("_", 1)[1]
    E[nm] = [x for x in json.loads(pathlib.Path(f).read_text()) if x.get("ok", True)]
print(f"{'scene':<15}{'n':>4}{'s_lm median':>13}{'within-scene sd':>18}{'median log_sd':>15}")
tot=[]
for nm, d in sorted(E.items()):
    if len(d) < 5: continue
    s = np.array([x["s"] for x in d])
    lsd = np.array([x["log_sd"] for x in d])
    tot.append((nm, len(d), float(np.median(s)), float(s.std()), float(np.median(lsd))))
    print(f"{nm:<15}{len(d):>4}{np.median(s):>13.4f}{s.std():>18.4f}{np.median(lsd):>15.4f}")
print()
print("If within-scene sd were near the per-image landmark noise (~0.02-0.10),")
print("then per-scene s_lm IS one number and the estimator is measuring that scene's")
print("SfM scale, not the model's per-image bias. That would mean the model's bias is")
print("already constant within a scene, which section 14's data supports.")
