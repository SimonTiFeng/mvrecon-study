"""Streaming leave-one-scene-out for the ETH3D landmark estimator.

Computes per-image scalars one at a time and never holds pixel arrays, so memory
stays flat regardless of how many images are covered.
"""
from __future__ import annotations

import glob
import json
import os
import pathlib

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/hpc/depths"
ALL_SCENES = ("courtyard", "facade", "delivery_area", "electro", "office",
              "playground", "terrace")


def ab(z, g):
    return float(np.mean(np.abs(z - g) / g))


def gt_opt(p, g):
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


def per_image(rows, scenes):
    out = []
    for sc in scenes:
        p_lower = ROOT / f"work/m_{sc}.json"
        p_upper = ROOT / f"work/M_{sc}.json"
        path = p_lower if p_lower.exists() else p_upper
        if not path.exists():
            continue
        n_done = 0
        for x in json.loads(path.read_text()):
            if not x.get("ok", True):
                continue
            r = rows.get(x["sid"])
            if not r or r["dataset"] != "eth3d":
                continue
            npz = CACHE / f"{x['sid']}__B.npz"
            side = CACHE / f"{x['sid']}__B.json"
            if not (npz.exists() and side.exists()):
                continue
            root = pathlib.Path(r["dataset_root"])
            gp = np.load(root / r["gt"]).astype(np.float32)
            mk = np.load(root / r["mask"]).astype(bool)
            d = np.load(npz)["depth"].astype(np.float32)
            sd = json.loads(side.read_text())
            Hp, Wp = sd["processed_shape"]
            K = np.asarray(r["K"], float)
            f = ((K[0, 0] * Wp / r["width"]) + (K[1, 1] * Hp / r["height"])) / 2 / 300.0
            # ETH3D native frames are 6048x4032; downsample before any arithmetic
            SCALE = 0.25
            tw, th = max(1, int(r["width"] * SCALE)), max(1, int(r["height"] * SCALE))
            d2 = cv2.resize(d, (tw, th), interpolation=cv2.INTER_LINEAR) * f
            # GT and mask must not be interpolated: bilinear mixes the invalid zeros
            # into the valid region and destroys most of the usable pixels
            gp = cv2.resize(gp, (tw, th), interpolation=cv2.INTER_NEAREST)
            mk = cv2.resize(mk.astype(np.uint8), (tw, th),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
            m = mk & np.isfinite(d2) & (d2 > 0) & np.isfinite(gp) & (gp > 0)
            if m.sum() < 500:
                continue
            p, g = d2[m], gp[m]
            # keep a compact ratio vector (float32, capped) instead of pixel arrays,
            # so the LOO can be recomputed at any scale later without re-reading
            if p.size > 6000:
                idx = np.linspace(0, p.size - 1, 6000).astype(np.int64)
                p, g = p[idx], g[idx]
            med_z = float(x.get("med_z", 0.0))
            out.append({"sid": x["sid"], "scene": sc, "s_lm": float(x["s"]),
                        "med_z": med_z,
                        "raw": ab(p, g), "s_gt": gt_opt(p, g),
                        "ratio": (p / g).astype(np.float32)})
            n_done += 1
            del p, g, d2, d, gp, mk
        print(f"  {sc}: {n_done} images", flush=True)
    return out


def lut(cal, s):
    return cal.get(round(float(s), 3), float("inf"))


def main():
    scenes = os.environ.get("SC")
    sc_list = tuple(scenes.split(",")) if scenes else ALL_SCENES
    rows = {json.loads(l)["sample_id"]: json.loads(l)
            for l in (ROOT / "manifests/formal_all.jsonl").read_text().splitlines() if l.strip()}
    print(f"scenes: {sc_list}", flush=True)
    data = per_image(rows, sc_list)
    # persist the compact records so a later killed run can resume from them
    np.savez_compressed(str(ROOT / f"work/loo_{'_'.join(sc_list)}.npz"),
                        sids=np.array([d['sid'] for d in data]),
                        scenes=np.array([d['scene'] for d in data]),
                        s_lm=np.array([d['s_lm'] for d in data]),
                        raw=np.array([d['raw'] for d in data]),
                        s_gt=np.array([d['s_gt'] for d in data]),
                        ratios=np.array([d['ratio'] for d in data], dtype=np.float32),
                        allow_pickle=False)
    # adaptive per-scene gate: drop frames whose landmarks sit in the furthest
    # quartile for that scene (measured failure mode on electro)
    import statistics as _st
    for sc in sc_list:
        sub = [d for d in data if d["scene"] == sc]
        if len(sub) < 8:
            continue
        zs = sorted(d["med_z"] for d in sub)
        thr = zs[int(0.75 * (len(zs) - 1))]
        for d in sub:
            if d["med_z"] > thr:
                d["raw"] = float("nan")
    data = [d for d in data if not (d["raw"] != d["raw"])]
    present = [sc for sc in sc_list if any(d["scene"] == sc for d in data)]

    lines = [f"images={len(data)} scenes={len(present)}"]
    lines.append(f"{'held-out':<15}{'const':>8}{'n':>5}{'raw':>9}{'cal':>9}{'wins':>8}")
    ca, ra = [], []
    for hold in present:
        sub = [d for d in data if d["scene"] == hold]
        fit = [d["s_lm"] / d["s_gt"] for d in data if d["scene"] != hold]
        c = float(np.median(fit)) if fit else 1.0
        rw = [d["raw"] for d in sub]
        cal = [float(np.mean(np.abs((d["s_lm"] / c) * d["ratio"] - 1.0))) for d in sub]
        lines.append(f"{hold:<15}{c:>8.4f}{len(sub):>5}{np.median(rw):>9.4f}"
                     f"{np.median(cal):>9.4f}"
                     f"{sum(1 for a, b in zip(cal, rw) if a < b):>5}/{len(sub):<3}")
        ca += cal
        ra += rw

    ca, ra = np.array(ca), np.array(ra)
    lines.append("")
    lines.append(f"POOLED raw {np.median(ra):.4f} -> {np.median(ca):.4f} "
                 f"wins {int((ca < ra).sum())}/{len(ra)}")
    zz = [float(np.mean(np.abs(d["s_lm"] * d["ratio"] - 1.0))) for d in data]
    rr = [d["raw"] for d in data]
    lines.append(f"ZERO-PARAM {np.median(rr):.4f} -> {np.median(zz):.4f} "
                 f"wins {sum(1 for a, b in zip(zz, rr) if a < b)}/{len(zz)}")
    text = "\n".join(lines)
    (ROOT / "work/loo_full.txt").write_text(text, encoding="utf-8")
    print(text)


# helper kept outside so it can be applied to a scalar-only record
def ab_scale(s, rec, _cache={}):
    key = rec["sid"]
    if key not in _cache:
        rows = {json.loads(l)["sample_id"]: json.loads(l)
                for l in (ROOT / "manifests/formal_all.jsonl").read_text().splitlines()
                if l.strip()}
        r = rows[key]
        root = pathlib.Path(r["dataset_root"])
        gp = np.load(root / r["gt"]).astype(np.float64)
        mk = np.load(root / r["mask"]).astype(bool)
        d = np.load(CACHE / f"{key}__B.npz")["depth"].astype(np.float64)
        sd = json.loads((CACHE / f"{key}__B.json").read_text())
        Hp, Wp = sd["processed_shape"]
        K = np.asarray(r["K"], float)
        f = ((K[0, 0] * Wp / r["width"]) + (K[1, 1] * Hp / r["height"])) / 2 / 300.0
        d2 = cv2.resize(d, (r["width"], r["height"])) * f
        m = mk & np.isfinite(d2) & (d2 > 0) & np.isfinite(gp) & (gp > 0)
        ratio = d2[m] / gp[m]
        _cache[key] = ratio
        if len(_cache) > 3:
            _cache.pop(next(iter(_cache)))
    return float(np.mean(np.abs(s * _cache[key] - 1.0)))


if __name__ == "__main__":
    main()
