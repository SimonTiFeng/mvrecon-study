"""Leave-one-scene-out evaluation of the landmark scale estimator on ETH3D.

Fits the SfM-vs-GT constant on the scenes left in, evaluates the held-out scene.
Writes incrementally so a killed process still leaves a partial result.
"""
from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCENES = ("courtyard", "facade", "delivery_area", "electro", "office", "playground", "terrace")


def ab(z, g):
    return float(np.mean(np.abs(z - g) / g))


def load(scene):
    cache = ROOT / "cache/hpc/depths"
    rows = {json.loads(l)["sample_id"]: json.loads(l)
            for l in (ROOT / "manifests/formal_all.jsonl").read_text().splitlines() if l.strip()}
    p = ROOT / f"work/L_{scene}.json"
    if not p.exists():
        return []
    out = []
    for x in json.loads(p.read_text()):
        if not x.get("ok", True):
            continue
        r = rows.get(x["sid"])
        if not r or r["dataset"] != "eth3d":
            continue
        npz = cache / f"{x['sid']}__B.npz"
        if not npz.exists():
            continue
        root = pathlib.Path(r["dataset_root"])
        gp = np.load(root / r["gt"]).astype(np.float64)
        mk = np.load(root / r["mask"]).astype(bool)
        d = np.load(npz)["depth"].astype(np.float64)
        sd = json.loads((cache / f"{x['sid']}__B.json").read_text())
        Hp, Wp = sd["processed_shape"]
        K = np.asarray(r["K"], float)
        f = ((K[0, 0] * Wp / r["width"]) + (K[1, 1] * Hp / r["height"])) / 2 / 300.0
        d2 = cv2.resize(d, (r["width"], r["height"])) * f
        m = mk & np.isfinite(d2) & (d2 > 0) & np.isfinite(gp) & (gp > 0)
        out.append({"sid": x["sid"], "scene": scene, "s_lm": x["s"],
                    "raw": ab(d2[m], gp[m]), "p": d2[m], "g": gp[m]})
    return out


def gt_opt(p, g):
    """Best single scale. Sub-samples and loops the grid so memory stays flat."""
    if p.size > 8000:
        idx = np.linspace(0, p.size - 1, 8000).astype(np.int64)
        p, g = p[idx], g[idx]
    r = p / g
    best_s, best_v = 1.0, np.inf
    for s in np.arange(0.60, 1.60, 0.005):
        v = float(np.mean(np.abs(s * r - 1.0)))
        if v < best_v:
            best_s, best_v = float(s), v
    return best_s


def main():
    out_path = ROOT / "work/loo_result.txt"
    data = []
    import os
    sc_set = os.environ.get("SC", "").split(",") or SCENES
    for sc in (sc_set if sc_set != [""] else SCENES):
        data += load(sc)
        print(f"loaded {sc}: total {len(data)}", flush=True)

    lines = [f"images={len(data)} scenes={len({d['scene'] for d in data})}"]
    lines.append(f"{'held-out':<15}{'const':>8}{'n':>4}{'raw':>9}{'cal':>9}{'wins':>7}")
    ca, ra = [], []
    for hold in SCENES:
        sub = [d for d in data if d["scene"] == hold]
        if not sub:
            continue
        fit = [d["s_lm"] / gt_opt(d["p"], d["g"]) for d in data if d["scene"] != hold]
        c = float(np.median(fit))
        rw = [d["raw"] for d in sub]
        cal = [ab((d["s_lm"] / c) * d["p"], d["g"]) for d in sub]
        lines.append(f"{hold:<15}{c:>8.4f}{len(sub):>4}{np.median(rw):>9.4f}"
                     f"{np.median(cal):>9.4f}{sum(1 for a, b in zip(cal, rw) if a < b):>4}/{len(sub):<3}")
        ca += cal
        ra += rw
    ca, ra = np.array(ca), np.array(ra)
    lines.append("")
    lines.append(f"POOLED raw {np.median(ra):.4f} -> {np.median(ca):.4f} "
                 f"wins {int((ca < ra).sum())}/{len(ra)}")
    lines.append("")
    lines.append("zero-parameter version (raw s_lm, no constant):")
    zz = [ab(d["s_lm"] * d["p"], d["g"]) for d in data]
    rr = [d["raw"] for d in data]
    lines.append(f"  {np.median(rr):.4f} -> {np.median(zz):.4f} "
                 f"wins {sum(1 for a, b in zip(zz, rr) if a < b)}/{len(zz)}")
    txt = "\n".join(lines)
    out_path.write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
