"""Decompose within-scene variance of s_lm into landmark noise vs real per-image variation."""
import json, pathlib, glob, numpy as np
E = {}
for f in glob.glob("work/m_*.json") + glob.glob("work/M_*.json"):
    nm = pathlib.Path(f).stem.split("_", 1)[1]
    E[nm] = [x for x in json.loads(pathlib.Path(f).read_text()) if x.get("ok", True)]

print("=== Is the within-scene spread explained by landmark noise alone? ===")
print("If s_lm were one constant per scene, sd(s_lm) should be ~ median(log_sd).")
print(f"\n{'scene':<15}{'n':>4}{'sd(s_lm)':>10}{'noise':>8}{'excess':>9}{'verdict':>12}")
for nm, d in sorted(E.items()):
    if len(d) < 8: continue
    s = np.array([x["s"] for x in d])
    lsd = np.array([x["log_sd"] for x in d])
    sd_s, noise = float(np.log(s).std()), float(np.median(lsd))
    excess = sd_s - noise
    verdict = "constant" if excess <= 0.3 * max(noise, 1e-6) else "varies"
    print(f"{nm:<15}{len(d):>4}{sd_s:>10.4f}{noise:>8.4f}{excess:>+9.4f}{verdict:>12}")
print()
print("Interpretation: 'constant' would mean the estimator is reading one SfM scale per")
print("scene and the model's bias is uniform there. 'varies' means genuine per-image")
print("model variation beyond what landmark precision explains.")
