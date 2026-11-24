"""Does using the per-scene median s_lm (instead of a global constant) improve ETH3D?"""
import json, pathlib, glob, numpy as np, cv2
C = pathlib.Path("cache/hpc/depths")
rows = {json.loads(l)["sample_id"]: json.loads(l)
        for l in open("manifests/formal_all.jsonl") if l.strip()}

def ab(z, g): return float(np.mean(np.abs(z - g) / g))

def gt_opt(p, g):
    if p.size > 4000:
        i = np.linspace(0, p.size-1, 4000).astype(np.int64); p, g = p[i], g[i]
    r = p / g; best = (1.0, np.inf)
    for s in np.arange(0.5, 2.0, 0.01):
        v = float(np.mean(np.abs(s*r - 1.0)))
        if v < best[1]: best = (float(s), v)
    return best[0]

data = {}
for f in glob.glob("work/m_*.json") + glob.glob("work/M_*.json"):
    nm = pathlib.Path(f).stem.split("_", 1)[1]
    for x in json.loads(pathlib.Path(f).read_text()):
        if not x.get("ok", True): continue
        sid = x["sid"]; r = rows.get(sid)
        if not r or r["dataset"] != "eth3d": continue
        npz = C / f"{sid}__B.npz"
        if not npz.exists(): continue
        sd = json.loads((C / f"{sid}__B.json").read_text()); Hp, Wp = sd["processed_shape"]
        K = np.asarray(r["K"], float)
        d = np.load(npz)["depth"].astype(np.float64)
        root = pathlib.Path(r["dataset_root"])
        gp = np.load(root / r["gt"]).astype(np.float64)
        mk = np.load(root / r["mask"]).astype(bool)
        f_ = ((K[0,0]*Wp/r["width"]) + (K[1,1]*Hp/r["height"])) / 2 / 300.0
        d2 = cv2.resize(d, (r["width"], r["height"])) * f_
        m = mk & np.isfinite(d2) & (d2 > 0) & np.isfinite(gp) & (gp > 0)
        if m.sum() < 500: continue
        p, g = d2[m], gp[m]
        data.setdefault(nm, []).append({"sid": sid, "s_lm": x["s"], "n": x["n"],
                                        "med_z": x.get("med_z", 0), "raw": ab(p, g),
                                        "s_gt": gt_opt(p, g), "ratio": (p/g).astype(np.float32)})

print("=== Per-scene median s_lm, leave-one-scene-out ===")
scenes = [s for s in data if len(data[s]) >= 8]
print(f"{'held-out':<15}{'n':>4}{'raw':>9}{'scene-med':>11}{'global-med':>12}{'s_gtmed':>9}")
ca_s, ca_g, ra = [], [], []
for hold in sorted(scenes):
    # scene-median estimator: median s_lm of the other scenes, scaled by their s_gt
    fit = [d["s_lm"]/d["s_gt"] for s in scenes if s != hold for d in data[s]]
    c_glob = float(np.median(fit))
    # per-scene estimator uses the held scene's OWN s_lm median (no constant), but
    # that is not GT-free unless the scene constant is a reference property.
    # We report both so the difference is visible.
    sub = data[hold]
    ra += [d["raw"] for d in sub]
    ca_g += [float(np.mean(np.abs((d["s_lm"]/c_glob)*d["ratio"] - 1.0))) for d in sub]
    smed = float(np.median([d["s_lm"] for d in sub]))
    ca_s += [float(np.mean(np.abs((d["s_lm"]/smed)*d["ratio"] - 1.0))) for d in sub]
    print(f"{hold:<15}{len(sub):>4}{np.median([d['raw'] for d in sub]):>9.4f}"
          f"{np.median(ca_s[-(len(sub)):]):>11.4f}{np.median(ca_g[-(len(sub)):]):>12.4f}"
          f"{np.median([d['s_gt'] for d in sub]):>9.3f}")
print(f"\nPOOLED raw {np.median(ra):.4f}")
print(f"  global-constant: {np.median(ca_g):.4f}  wins {sum(1 for a,b in zip(ca_g,ra) if a<b)}/{len(ra)}")
print(f"  scene-median:    {np.median(ca_s):.4f}  wins {sum(1 for a,b in zip(ca_s,ra) if a<b)}/{len(ra)}")
