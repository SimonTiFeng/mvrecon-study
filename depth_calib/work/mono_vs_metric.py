"""Cross-model: can landmark scaling bring a scale-free mono model to metric quality?"""
import json, pathlib, glob, numpy as np, cv2

C = pathlib.Path("cache/hpc/depths")
rows = {json.loads(l)["sample_id"]: json.loads(l)
        for l in open("manifests/formal_all.jsonl") if l.strip()}

def ab(z, g): return float(np.mean(np.abs(z - g) / g))

def scores(prefix, lm_map):
    out = []
    for sid, r in rows.items():
        if r["dataset"] != "eth3d": continue
        npz = C / f"{sid}__{prefix}.npz"
        if not npz.exists(): continue
        sd = json.loads((C / f"{sid}__{prefix}.json").read_text())
        Hp, Wp = sd["processed_shape"]
        K = np.asarray(r["K"], float)
        d = np.load(npz)["depth"].astype(np.float64)
        root = pathlib.Path(r["dataset_root"])
        gp = np.load(root / r["gt"]).astype(np.float64)
        mk = np.load(root / r["mask"]).astype(bool)
        f = ((K[0,0]*Wp/r["width"]) + (K[1,1]*Hp/r["height"])) / 2 / 300.0
        d2 = cv2.resize(d, (r["width"], r["height"])) * f
        m = mk & np.isfinite(d2) & (d2 > 0) & np.isfinite(gp) & (gp > 0)
        if m.sum() < 500: continue
        p, g = d2[m], gp[m]
        if p.size > 20000:
            idx = np.linspace(0, p.size-1, 20000).astype(np.int64)
            p, g = p[idx], g[idx]
        grid = np.arange(0.4, 3.0, 0.02)
        best = min(((s, ab(s*p, g)) for s in grid), key=lambda t: t[1])
        lm = lm_map.get(sid)
        out.append({"sid": sid, "raw": ab(p, g), "s_opt": best[0], "opt": best[1],
                    "s_lm": lm, "cal": ab(lm*p, g) if lm else None})
    return out

lm = {}
for f in glob.glob("work/m_*.json") + glob.glob("work/M_*.json"):
    for x in json.loads(pathlib.Path(f).read_text()):
        if x.get("ok", True): lm[x["sid"]] = x["s"]

print(f"{'branch':<10}{'n':>4}{'raw':>10}{'oracle':>10}{'+landmark':>11}{'wins':>8}")
for br, name in (("A","DA3-Mono (scale-free)"), ("B","DA3-Metric")):
    r = scores(br, lm)
    if not r: continue
    cal = [x["cal"] for x in r if x["cal"] is not None]
    raw = [x["raw"] for x in r if x["cal"] is not None]
    wins = sum(1 for a,b in zip(cal,raw) if a < b)
    print(f"{br:<10}{len(r):>4}{np.median(raw):>10.4f}{np.median([x['opt'] for x in r]):>10.4f}"
          f"{np.median(cal):>11.4f}{wins:>5}/{len(cal)}")
print()
print("A is scale-free: its raw AbsRel is meaningless until scaled. The oracle column")
print("is its ceiling given the best single scale; the landmark column is what the")
print("same GT-free estimator achieves on it.")
