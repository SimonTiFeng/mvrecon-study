"""Pooled global scale from LLM-selected anchors.

Each image yields few (1-3) noisy anchors, so a per-image scale is unidentifiable.
The report's oracle shows one GLOBAL scale recovers half the gap (0.1040 -> 0.0519).
This tests whether the LLM anchors, POOLED ACROSS IMAGES, estimate that scale.

Estimator never touches GT: s_i = L_i / ||q1*d1 - q2*d2|| with L_i the anchor's
claimed length and d the metric backend's own prediction at the two endpoints.

Evaluation is leave-one-image-out: image k is scored with a scale estimated only
from the other images' anchors.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.geometry.camera import Camera, sample_depth_bilinear  # noqa: E402
from src.geometry.depth_affine import fit_depth_affine  # noqa: E402
from src.geometry.metric_scale import fit_metric_scale  # noqa: E402
from src.geometry.types import AnchorGeom  # noqa: E402


def absrel(z, g):
    return float(np.mean(np.abs(z - g) / g))


def load_pilot():
    rows = [json.loads(l) for l in
            (ROOT / "manifests/pilot.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    return {r["sample_id"]: r for r in rows}


def anchors_for(sid):
    p = ROOT / "results/pilot_f1_001/samples" / sid / "mcp_out/anchors" / f"{sid}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("status") != "ok" or not d.get("anchors"):
        return None
    return d["anchors"]


def build(sid, row, cfg, metric):
    """Return (per-anchor records, (gt, metric_depth_on_mask), camera)."""
    dpath = ROOT / "cache/hpc/depths" / f"{sid}__B.npz"
    if not dpath.exists():
        return None
    d = np.load(dpath)["depth"].astype(np.float64)
    side = json.loads((ROOT / "cache/hpc/depths" / f"{sid}__B.json")
                      .read_text(encoding="utf-8"))
    gt = np.load(pathlib.Path(row["dataset_root"]) / row["gt"]).astype(np.float64)
    mask = np.load(pathlib.Path(row["dataset_root"]) / row["mask"]).astype(bool)
    if d.shape != gt.shape:
        return None
    # DA3 metric output needs the frozen focal/300 conversion to be in metres
    Hp, Wp = side["processed_shape"]
    K0 = np.asarray(row["K"], float)
    d *= ((K0[0, 0] * Wp / row["width"]) + (K0[1, 1] * Hp / row["height"])) / 2.0 / 300.0
    K = np.asarray(row["K"], float)
    cam = Camera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], d.shape[1], d.shape[0],
                 model=row.get("camera_model", "PINHOLE"), params=row.get("camera_params"))
    m = mask & np.isfinite(d) & np.isfinite(gt) & (d > 0) & (gt > 0)
    if m.sum() < 100:
        return None
    recs, geoms = [], []
    for a in anchors_for(sid) or []:
        p1 = np.asarray(a["p1_xy"], float)
        p2 = np.asarray(a["p2_xy"], float)
        dd, ok = sample_depth_bilinear(d, np.stack([p1, p2]),
                                       eps=cfg["quality"]["depth_valid_eps"])
        if not ok.all():
            continue
        q1, q2 = cam.q(p1)[0], cam.q(p2)[0]
        l_i = float(np.linalg.norm(q1 * float(dd[0]) - q2 * float(dd[1])))
        if not np.isfinite(l_i) or l_i <= 1e-9:
            continue
        L = float(a["length_m"])
        s_i = L / l_i
        iv = a.get("length_interval_m")
        width = float(iv[1] - iv[0]) if iv and iv[1] > iv[0] else None
        recs.append({"gid": a.get("knowledge_group_id") or a["anchor_id"],
                     "s_i": s_i, "width": width})
        geoms.append(AnchorGeom(a["anchor_id"], a.get("object_group_id", ""),
                                a.get("knowledge_group_id", ""), p1, p2, q1, q2, L,
                                d1=float(dd[0]), d2=float(dd[1])))
    return {"sid": sid, "anchors": recs, "geoms": geoms,
            "gt": gt[m], "pred": d[m]}


def pooled_scale(per_image, keep_sids, weighted=False):
    """Robust pooled scale in log space, one vote per (image, knowledge group)."""
    vals, wts = [], []
    for sid in keep_sids:
        g = {}
        for r in per_image[sid]["anchors"]:
            g.setdefault(r["gid"], []).append(r)
        for gid, rs in g.items():
            logs = [np.log(max(r["s_i"], 1e-9)) for r in rs]
            vals.append(float(np.median(logs)))
            if weighted:
                ws = [1.0 / (r["width"] ** 2) if r["width"] else 1.0 for r in rs]
                wts.append(float(np.mean(ws)))
            else:
                wts.append(1.0)
    if not vals:
        return None
    v = np.array(vals)
    w = np.array(wts)
    order = np.argsort(v)
    v, w = v[order], w[order]
    c = np.cumsum(w)
    return float(np.exp(v[np.searchsorted(c, 0.5 * c[-1])]))


def oracle_global(per_image, sids):
    logs = []
    for sid in sids:
        p, g = per_image[sid]["pred"], per_image[sid]["gt"]
        best = None
        for a in np.exp(np.linspace(np.log(0.3), np.log(3.0), 400)):
            v = absrel(a * p, g)
            if best is None or v < best[1]:
                best = (a, v)
        logs.append(np.log(best[0]))
    return float(np.exp(np.median(logs)))


def main():
    cfg = load_config(ROOT / "configs" / "pilot.yaml")
    rows = load_pilot()
    per_image = {}
    for sid, row in rows.items():
        r = build(sid, row, cfg, None)
        if r is not None and r["anchors"]:
            per_image[sid] = r
    sids = sorted(per_image)
    n_anc = sum(len(per_image[s]["anchors"]) for s in sids)
    print(f"images with usable anchors: {len(sids)}   anchors: {n_anc}")
    if not sids:
        return

    out = {"n_images": len(sids), "n_anchors": n_anc, "per_image": {}}
    pooled_all = pooled_scale(per_image, sids)
    pooled_w = pooled_scale(per_image, sids, weighted=True)
    s_or = oracle_global(per_image, sids)
    print(f"pooled scale (in-sample)      = {pooled_all:.4f}")
    print(f"pooled scale (interval-wtd)   = {pooled_w:.4f}")
    print(f"oracle global scale (GT-fit)  = {s_or:.4f}")

    res = {k: [] for k in ("raw", "per_image", "pooled_is", "pooled_loo",
                           "pooled_w_loo", "oracle_global", "oracle_affine")}
    n_applied = 0
    for sid in sids:
        d = per_image[sid]
        p, g = d["pred"], d["gt"]
        res["raw"].append(absrel(p, g))
        sol = fit_metric_scale(d["geoms"], cfg)
        s_pi = sol.get("s_consistent") if sol["status"] == "ok" else 1.0
        if sol["status"] == "ok":
            n_applied += 1
        res["per_image"].append(absrel(max(s_pi, 1e-8) * p, g))
        res["pooled_is"].append(absrel(max(pooled_all, 1e-8) * p, g))
        others = [s for s in sids if s != sid]
        pl = pooled_scale(per_image, others)
        plw = pooled_scale(per_image, others, weighted=True)
        res["pooled_loo"].append(absrel(max(pl, 1e-8) * p, g) if pl else np.nan)
        res["pooled_w_loo"].append(absrel(max(plw, 1e-8) * p, g) if plw else np.nan)
        res["oracle_global"].append(absrel(s_or * p, g))
        # per-image affine oracle (ceiling)
        best = None
        med = float(np.median(p))
        for a in np.exp(np.linspace(np.log(0.2), np.log(5.0), 60)):
            for b in np.linspace(-0.6 * med, 0.6 * med, 25):
                z = a * p + b
                if np.any(z <= 0):
                    continue
                v = absrel(z, g)
                if best is None or v < best:
                    best = v
        res["oracle_affine"].append(best if best is not None else np.nan)
        out["per_image"][sid] = {
            "n_anchors": len(d["anchors"]),
            "s_per_image": s_pi, "status": sol["status"],
            "absrel_raw": res["raw"][-1], "absrel_per_image": res["per_image"][-1],
            "absrel_pooled_loo": res["pooled_loo"][-1],
        }

    print(f"\nper-image corrections applied: {n_applied}/{len(sids)}")
    print(f"\n{'system':28s} {'median AbsRel':>14s}")
    for k in ("raw", "per_image", "pooled_is", "pooled_loo", "pooled_w_loo",
              "oracle_global", "oracle_affine"):
        v = np.array(res[k], float)
        v = v[np.isfinite(v)]
        print(f"  {k:26s} {np.median(v):>10.4f}   (n={len(v)})")

    raw = np.array(res["raw"])
    for k in ("per_image", "pooled_loo", "pooled_w_loo"):
        d_ = np.array(res[k]) - raw
        d_ = d_[np.isfinite(d_)]
        wins = int((d_ < 0).sum())
        rng = np.random.default_rng(0)
        idx = rng.integers(0, d_.size, size=(4000, d_.size))
        boot = np.median(d_[idx], axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  paired Δ {k:16s} {np.median(d_):+.4f}  [{lo:+.4f},{hi:+.4f}]  "
              f"wins {wins}/{d_.size}")
    out["summary"] = {k: float(np.median(np.array(res[k], float))) for k in res}
    out["summary"]["pooled_scale_in_sample"] = pooled_all
    out["summary"]["pooled_scale_interval_weighted"] = pooled_w
    out["summary"]["oracle_global_scale"] = s_or
    (ROOT / "results/exploratory/pooled_global_scale.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print("\nwrote results/exploratory/pooled_global_scale.json")


if __name__ == "__main__":
    main()
