"""Head-to-head: DA3 Mono + our anchors vs DA3 Metric (raw and with our anchors).

All metrics on all valid pixels of each image, no subsampling. Same images, same anchors,
same solver family. A's raw output is in arbitrary units, so it is never scored as a metric
prediction; where A's solver declines, the pre-declared fallback is B's metric output.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config                        # noqa: E402
from src.geometry.camera import Camera, sample_depth_bilinear   # noqa: E402
from src.geometry.depth_affine import fit_depth_affine    # noqa: E402
from src.geometry.metric_scale import fit_metric_scale    # noqa: E402
from src.geometry.types import AnchorGeom                 # noqa: E402


def absrel(z, g):
    return float(np.mean(np.abs(z - g) / g)) if z.size else float("nan")


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


def analyse(manifest: str, run_dir: str, label: str) -> list[dict]:
    cfg = load_config(ROOT / "configs" / "pilot.yaml")
    rows = {json.loads(l)["sample_id"]: json.loads(l) for l in
            (ROOT / manifest).read_text(encoding="utf-8").splitlines() if l.strip()}
    cache = ROOT / "cache" / "hpc" / "depths"
    recs = []
    for sid, row in rows.items():
        nf = ROOT / "results" / run_dir / "samples" / sid / "mcp_out" / "anchors" / f"{sid}.json"
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

        m = mask & np.isfinite(gt) & (gt > 0) & np.isfinite(dA) & (dA > 0) & np.isfinite(dB) & (dB > 0)
        if m.sum() < 100:
            continue
        g, pA, pB = gt[m], dA[m], dB[m]

        solA = fit_depth_affine(geoms_for(norm, dA, cam, cfg), cfg)
        solB = fit_metric_scale(geoms_for(norm, dB, cam, cfg), cfg)
        a_ok = solA["status"] == "ok"
        b_ok = solB["status"] == "ok"

        ar_Braw = absrel(pB, g)
        ar_Bours = absrel(np.maximum(solB["s_consistent"] * pB, 1e-9), g) if (
            b_ok and solB["status"] == "ok") else ar_Braw
        ar_Aours = absrel(np.maximum(solA["a"] * pA + solA["b"], 1e-9), g) if a_ok else None
        _, _, ar_Aoracle = oracle_affine(pA, g)

        recs.append({
            "sample_id": sid, "dataset": row["dataset"],
            "B_raw": ar_Braw, "B_ours": ar_Bours, "B_status": solB["status"],
            "A_ours": ar_Aours, "A_gated_ok": a_ok, "A_status": solA["status"],
            "A_oracle": ar_Aoracle,
            # pre-declared complete systems (handoff 6.6 / 9.1)
            "A_ours_B_fallback": ar_Aours if a_ok else ar_Braw,
            "A_oracle_B_fallback": ar_Aoracle if a_ok else ar_Braw,
        })
    pathlib.Path(ROOT / "results" / "exploratory" / f"metric_vs_mono_{label}.json").write_text(
        json.dumps(recs, indent=1), encoding="utf-8")
    return recs


def report(recs: list[dict], label: str) -> None:
    def med(k, sub=None):
        v = [r[k] for r in (sub if sub is not None else recs)
             if r[k] is not None and np.isfinite(r[k])]
        return float(np.median(v)) if v else float("nan")

    ok = [r for r in recs if r["A_gated_ok"]]
    print(f"\n=== {label}: submitted={len(recs)}, A-gated-ok={len(ok)}")
    print("  ALL images (A falls back to B where its solver declines):")
    print(f"    B raw                {med('B_raw'):.4f}")
    print(f"    B + our anchors      {med('B_ours'):.4f}")
    print(f"    A + our affine, B fallback   {med('A_ours_B_fallback'):.4f}")
    print(f"    A oracle affine, B fallback  {med('A_oracle_B_fallback'):.4f}")
    if ok:
        print(f"  ON THE {len(ok)} IMAGES WHERE A'S SOLVER ACCEPTS (same images, no fallback):")
        for k, nm in (("B_raw", "B raw"), ("B_ours", "B + our anchors"),
                      ("A_ours", "A + our affine"), ("A_oracle", "A oracle affine")):
            print(f"    {nm:18s} {med(k, ok):.4f}")
        wins = sum(1 for r in ok if r["A_ours"] < r["B_raw"])
        print(f"    A+ours beats B raw on {wins}/{len(ok)} of these images")
        wins2 = sum(1 for r in ok if r["A_ours"] < r["B_ours"])
        print(f"    A+ours beats B+ours on {wins2}/{len(ok)}")


if __name__ == "__main__":
    report(analyse("manifests/pilot.jsonl", "pilot_f1_001", "pilot"), "pilot")
    report(analyse("manifests/explor_20.jsonl", "explor_points_001", "explor_points"),
           "explor points")
    report(analyse("manifests/explor_20.jsonl", "explor_direct_001", "explor_direct"),
           "explor direct")
