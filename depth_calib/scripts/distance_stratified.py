"""Distance-stratified diagnostic per Depth_Distance_Diagnostic_Plan.md (2026-11-17).

Deliverables
  1. Overall comparison: Metric raw / Mono+anchors affine / Mono GT-scale oracle / Mono GT-affine oracle
  2. Pixel-level AbsRel by GT depth band (fixed boundaries, per-image mean then cross-image mean)
  3. Image-level AbsRel grouped by each image's GT median depth
  4. Anchor endpoint GT depth range vs whole-image depth range

Rules honoured:
  - One calibration parameter set per image; never re-fitted per band.
  - Empty bands reported as N/A, sparse bands descriptive only.
  - NYUv2 and ETH3D reported separately; image counts and pixel counts given.
  - All pixels used (subsampling limits come from config, currently 0 = all).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config                       # noqa: E402
from src.geometry.camera import Camera, sample_depth_bilinear   # noqa: E402
from src.geometry.depth_affine import fit_depth_affine    # noqa: E402
from src.geometry.metric_scale import fit_metric_scale    # noqa: E402
from src.geometry.types import AnchorGeom                 # noqa: E402

BANDS = [(0.0, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 5.0), (5.0, float("inf"))]
BAND_NAMES = ["<0.5", "0.5-1.5", "1.5-3", "3-5", ">=5"]
IMG_GROUPS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, float("inf"))]
IMG_GROUP_NAMES = ["median<1", "1-2", "2-3", ">=3"]


def absrel(z, g):
    return float(np.mean(np.abs(z - g) / g)) if z.size else float("nan")


def _band(fn, lo, hi):
    if lo == 0.0:
        return fn <= hi if np.isinf(hi) else fn < hi
    return (fn >= lo) & (fn < hi) if not np.isinf(hi) else fn >= lo


def oracle_scale(p, g):
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda a: absrel(a * p, g) if a > 0 else 1e9,
                        bounds=(1e-4, 1e4), method="bounded", options={"xatol": 1e-9})
    return float(r.x), absrel(r.x * p, g)


def oracle_affine(p, g):
    med = float(np.median(p))

    def loss(t):
        a = float(np.exp(t[0])); b = float(t[1] * med); z = a * p + b
        return 10.0 if np.any(z <= 0) else absrel(z, g)

    best = None
    for la in (np.log(0.1), np.log(0.5), 0.0, np.log(2.0), np.log(5.0)):
        for bh in (-2., -1., 0., 1., 2.):
            r = minimize(loss, np.array([la, bh]), method="L-BFGS-B",
                         bounds=[(np.log(1e-4), np.log(1e4)), (-30, 30)],
                         options={"maxiter": 400})
            if best is None or r.fun < best.fun:
                best = r
    return float(np.exp(best.x[0])), float(best.x[1] * med), float(best.fun)


def geoms_for(norm, depth, cam, cfg):
    out = []
    for a in norm["anchors"]:
        p1 = np.asarray(a["p1_xy"], float); p2 = np.asarray(a["p2_xy"], float)
        dd, ok = sample_depth_bilinear(depth, np.stack([p1, p2]),
                                       eps=cfg["quality"]["depth_valid_eps"])
        g = AnchorGeom(a["anchor_id"], a["object_group_id"], a["knowledge_group_id"],
                       p1, p2, cam.q(p1)[0], cam.q(p2)[0], float(a["length_m"]))
        if ok.all():
            g.d1, g.d2 = float(dd[0]), float(dd[1])
        out.append(g)
    return out


def collect(cfg, cache):
    """One record per image, with per-band and per-image AbsRel for each system."""
    runs = [("pilot_f1_001", "manifests/pilot.jsonl")]
    rows_by_sid = {}
    for _, mf in runs:
        for l in (ROOT / mf).read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                rows_by_sid.setdefault(r["sample_id"], r)
    recs = []
    for sid, row in rows_by_sid.items():
        nf = ROOT / "results" / "pilot_f1_001" / "samples" / sid / "mcp_out" / "anchors" / f"{sid}.json"
        fa, fb = cache / f"{sid}__A.npz", cache / f"{sid}__B.npz"
        if not (nf.exists() and fa.exists() and fb.exists()):
            continue
        norm = json.loads(nf.read_text(encoding="utf-8"))
        if norm.get("status") != "ok" or not norm.get("anchors"):
            continue
        gt = np.load(pathlib.Path(row["dataset_root"]) / row["gt"]).astype(np.float64)
        mask = np.load(pathlib.Path(row["dataset_root"]) / row["mask"]).astype(bool)
        K = np.asarray(row["K"], float)
        cam = Camera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], row["width"], row["height"],
                     model=row.get("camera_model", "PINHOLE"),
                     params=row.get("camera_params"))
        dA = np.load(fa)["depth"].astype(np.float64)
        dB = np.load(fb)["depth"].astype(np.float64)
        mB = json.loads((cache / f"{sid}__B.json").read_text(encoding="utf-8"))
        Hp, Wp = mB["processed_shape"]
        dB *= ((K[0, 0] * Wp / row["width"]) + (K[1, 1] * Hp / row["height"])) / 2 / 300.0
        m = mask & np.isfinite(gt) & (gt > 0) & np.isfinite(dA) & (dA > 0) \
            & np.isfinite(dB) & (dB > 0)
        if m.sum() < 100:
            continue
        g, pA, pB = gt[m], dA[m], dB[m]

        # one calibration per image, applied to every band
        solA = fit_depth_affine(geoms_for(norm, dA, cam, cfg), cfg)
        solB = fit_metric_scale(geoms_for(norm, dB, cam, cfg), cfg)
        a_ok = solA["status"] == "ok"
        pred_A = (solA["a"] * pA + solA["b"]) if a_ok else None
        pred_B = (solB.get("s_consistent", solB["s"]) * pB) if solB["status"] == "ok" else pB
        a_o, b_o, _ = oracle_affine(pA, g)
        s_o, _ = oracle_scale(pA, g)
        pred_Ao = a_o * pA + b_o
        pred_As = s_o * pA

        def per_band(pred):
            out = {}
            for nm, (lo, hi) in zip(BAND_NAMES, BANDS):
                sel = _band(g, lo, hi)
                out[nm] = (absrel(pred[sel], g[sel]) if pred is not None and sel.sum() >= 100
                           else None, int(sel.sum()))
            return out

        # anchor endpoint depth span vs image span
        ep = []
        for a in norm["anchors"]:
            for pv in ("p1_xy", "p2_xy"):
                v, okk = sample_depth_bilinear(gt, np.array([np.asarray(a[pv], float)]), mask)
                if okk[0]:
                    ep.append(float(v[0]))
        ep_span = (float(np.log(max(ep) / max(min(ep), 1e-9))) if len(ep) >= 2 else None)
        img_span = float(np.log(np.percentile(g, 95) / max(np.percentile(g, 5), 1e-9)))

        recs.append({
            "sample_id": sid, "dataset": row["dataset"], "scene_id": row["scene_id"],
            "n_pix": int(g.size), "median_depth": float(np.median(g)),
            "A_gated_ok": a_ok, "B_status": solB["status"], "n_anchors": len(norm["anchors"]),
            "absrel_ALL": {
                "B_raw": absrel(pB, g),
                "B_ours": absrel(pred_B, g),
                "A_ours": absrel(pred_A, g) if pred_A is not None else None,
                "A_scale_oracle": absrel(pred_As, g),
                "A_affine_oracle": absrel(pred_Ao, g),
            },
            "bands": {
                "B_raw": per_band(pB),
                "B_ours": per_band(pred_B),
                "A_ours": per_band(pred_A),
                "A_scale_oracle": per_band(pred_As),
                "A_affine_oracle": per_band(pred_Ao),
            },
            "anchor_ep_depth_min": min(ep) if ep else None,
            "anchor_ep_depth_max": max(ep) if ep else None,
            "anchor_depth_logspan": ep_span, "image_depth_logspan": img_span,
            "anchor_span_frac": (ep_span / img_span) if ep_span and img_span > 0 else None,
        })
    return recs


def report(recs):
    systems = ["B_raw", "B_ours", "A_ours", "A_scale_oracle", "A_affine_oracle"]
    print("\n## 1. Overall (median across images; A_ours over images where its solver accepts)")
    for ds in sorted({r["dataset"] for r in recs}):
        sub = [r for r in recs if r["dataset"] == ds]
        ok = [r for r in sub if r["A_gated_ok"]]
        print(f"\n{ds}: images={len(sub)}, A-solver-accepted={len(ok)}, "
              f"pixels={sum(r['n_pix'] for r in sub):,}")
        for s in systems:
            v_all = [r["absrel_ALL"][s] for r in sub if r["absrel_ALL"][s] is not None]
            v_ok = [r["absrel_ALL"][s] for r in ok if r["absrel_ALL"][s] is not None]
            sa = f"{np.median(v_all):.4f}" if v_all else "N/A"
            so = f"{np.median(v_ok):.4f}" if v_ok else "N/A"
            print(f"  {s:18s} all={sa}  (A-accepted subset={so})")

    print("\n## 2. Pixel-level AbsRel by GT depth band")
    print("   (per-image mean within band, then median across images; n = images with >=100 px)")
    for ds in sorted({r["dataset"] for r in recs}):
        sub = [r for r in recs if r["dataset"] == ds]
        ok = [r for r in sub if r["A_gated_ok"]]
        print(f"\n{ds} (all {len(sub)} images)")
        print("  band        " + "".join(f"{s:>18s}" for s in systems))
        for nm in BAND_NAMES:
            row = f"  {nm:<10s}"
            for s in systems:
                v = [r["bands"][s][nm][0] for r in sub
                     if r["bands"][s][nm][0] is not None]
                row += f"{(f'{np.median(v):.4f}' if v else 'N/A'):>18s}"
            print(row)
        print(f"  -- A-accepted images only ({len(ok)}) --")
        for nm in BAND_NAMES:
            row = f"  {nm:<10s}"
            for s in systems:
                v = [r["bands"][s][nm][0] for r in ok
                     if r["bands"][s][nm][0] is not None]
                row += f"{(f'{np.median(v):.4f}' if v else 'N/A'):>18s}"
            print(row)

    print("\n## 3. Image-level: AbsRel grouped by that image's GT median depth")
    for ds in sorted({r["dataset"] for r in recs}):
        sub = [r for r in recs if r["dataset"] == ds]
        print(f"\n{ds}")
        print("  group       n   " + "".join(f"{s:>18s}" for s in systems))
        for nm, (lo, hi) in zip(IMG_GROUP_NAMES, IMG_GROUPS):
            grp = [r for r in sub if _band(np.array([r["median_depth"]]), lo, hi)[0]]
            if not grp:
                print(f"  {nm:<10s} 0   " + "".join(f"{'N/A':>18s}" for _ in systems))
                continue
            row = f"  {nm:<10s} {len(grp):<3d} "
            for s in systems:
                v = [r["absrel_ALL"][s] for r in grp if r["absrel_ALL"][s] is not None]
                row += f"{(f'{np.median(v):.4f}' if v else 'N/A'):>18s}"
            print(row)

    print("\n## 4. Anchor depth coverage vs image depth range")
    for ds in sorted({r["dataset"] for r in recs}):
        sub = [r["sample_id"] for r in recs if r["dataset"] == ds]
        f = [r["anchor_span_frac"] for r in recs
             if r["dataset"] == ds and r["anchor_span_frac"] is not None]
        if f:
            print(f"  {ds}: anchor log-span / image log-span  median {np.median(f):.2f} "
                  f"(p10-p90 {np.percentile(f,10):.2f}-{np.percentile(f,90):.2f}); "
                  f"<30%: {sum(1 for x in f if x < 0.3)}/{len(f)}")


if __name__ == "__main__":
    cfg = load_config(ROOT / "configs" / "pilot.yaml")
    cache = ROOT / "cache" / "hpc" / "depths"
    recs = collect(cfg, cache)
    (ROOT / "results" / "exploratory" / "distance_stratified.json").write_text(
        json.dumps(recs, indent=1), encoding="utf-8")
    report(recs)
