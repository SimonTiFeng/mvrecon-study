"""Affine oracle vs our anchor-driven affine, stratified by identifiability.

Variants per image (all metrics on ALL valid pixels, no subsampling):
  A_oracle          per-image GT-fitted (a,b)              -- ceiling of the family
  A_ours_gated      our solver with identifiability gating (fallback -> no transform)
  A_ours_ungated    always fit something from whatever anchors exist
                    (1 anchor: b=0, a solved; >=2: robust affine, rank ignored)
  A_ours_ok_only    our solver's output restricted to images it declares usable
Reports per-image a/b drift versus the oracle so the error direction is visible.
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
from src.geometry.camera import Camera, sample_depth_bilinear  # noqa: E402
from src.geometry.depth_affine import fit_depth_affine, apply_depth_affine  # noqa: E402
from src.geometry.types import AnchorGeom                # noqa: E402


def absrel(z, g):
    return float(np.mean(np.abs(z - g) / g)) if z.size else float("nan")


def oracle_affine(p, g):
    med = float(np.median(p))

    def loss(t):
        a = float(np.exp(t[0])); b = float(t[1] * med)
        z = a * p + b
        if np.any(z <= 0):
            return 10.0
        return absrel(z, g)

    best = None
    for la in (np.log(0.2), 0.0, np.log(5.0)):
        for bh in (-1.0, 0.0, 1.0):
            r = minimize(loss, np.array([la, bh]), method="L-BFGS-B",
                         bounds=[(np.log(1e-4), np.log(1e4)), (-20, 20)],
                         options={"maxiter": 400})
            if best is None or r.fun < best.fun:
                best = r
    return float(np.exp(best.x[0])), float(best.x[1] * med), float(best.fun)


def ungated_affine(anchors, cfg):
    """Fit a,b from whatever anchors exist; ignore rank/degeneracy gating."""
    rows = []
    for a in anchors:
        if a.d1 is None or a.d2 is None:
            continue
        U = np.asarray(a.q1) * a.d1 - np.asarray(a.q2) * a.d2
        V = np.asarray(a.q1) - np.asarray(a.q2)
        rows.append((U, V, a.length_m))
    if not rows:
        return None
    if len(rows) == 1:                     # one equation: fix b=0, solve a
        U, V, L = rows[0]
        base = float(np.linalg.norm(U))
        if base <= 1e-12:
            return None
        return L / base, 0.0
    U = np.array([r[0] for r in rows]); V = np.array([r[1] for r in rows])
    L = np.array([r[2] for r in rows])
    med = float(np.median(np.linalg.norm(U, axis=1)))

    def loss(t):
        a = float(np.exp(t[0])); b = float(t[1] * med)
        Lh = np.linalg.norm(a * U + b * V, axis=1)
        r = np.log(np.maximum(Lh, 1e-12)) - np.log(L)
        return float(np.mean(np.minimum(r ** 2, 0.15 ** 2)))

    best = None
    for la in (np.log(0.2), 0.0, np.log(5.0)):
        for bh in (-1.0, 0.0, 1.0):
            r = minimize(loss, np.array([la, bh]), method="L-BFGS-B",
                         bounds=[(np.log(1e-6), np.log(1e6)), (-20, 20)],
                         options={"maxiter": 300})
            if best is None or r.fun < best.fun:
                best = r
    return float(np.exp(best.x[0])), float(best.x[1] * med)


def run(manifest: str, run_dir: str, label: str):
    cfg = load_config(ROOT / "configs" / "pilot.yaml")
    rows = [json.loads(l) for l in
            pathlib.Path(ROOT / manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    cache = ROOT / "cache" / "hpc" / "depths"
    out = []
    for row in rows:
        sid = row["sample_id"]
        nf = ROOT / "results" / run_dir / "samples" / sid / "mcp_out" / "anchors" / f"{sid}.json"
        fa = cache / f"{sid}__A.npz"
        if not (nf.exists() and fa.exists()):
            continue
        norm = json.loads(nf.read_text(encoding="utf-8"))
        if norm.get("status") != "ok" or not norm.get("anchors"):
            continue
        gt = np.load(pathlib.Path(row["dataset_root"]) / row["gt"]).astype(np.float64)
        mask = np.load(pathlib.Path(row["dataset_root"]) / row["mask"]).astype(bool)
        dA = np.load(fa)["depth"].astype(np.float64)
        K = np.asarray(row["K"], float)
        cam = Camera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], row["width"], row["height"],
                     model=row.get("camera_model", "PINHOLE"),
                     params=row.get("camera_params"))
        m = mask & np.isfinite(dA) & np.isfinite(gt) & (dA > 0) & (gt > 0)
        p, g = dA[m], gt[m]
        if p.size < 100:
            continue
        geoms = []
        for a in norm["anchors"]:
            p1 = np.asarray(a["p1_xy"], float); p2 = np.asarray(a["p2_xy"], float)
            dd, ok = sample_depth_bilinear(dA, np.stack([p1, p2]),
                                          eps=cfg["quality"]["depth_valid_eps"])
            gg = AnchorGeom(a["anchor_id"], a["object_group_id"], a["knowledge_group_id"],
                            p1, p2, cam.q(p1)[0], cam.q(p2)[0], float(a["length_m"]))
            if ok.all():
                gg.d1, gg.d2 = float(dd[0]), float(dd[1])
            geoms.append(gg)

        a_o, b_o, ar_o = oracle_affine(p, g)
        sol = fit_depth_affine(geoms, cfg)
        gated_ok = sol["status"] == "ok"
        ar_gated = absrel(np.maximum(sol["a"] * p + sol["b"], 1e-8), g) if gated_ok else ar_raw_of(dA, gt, mask)
        ung = ungated_affine(geoms, cfg)
        ar_ung = absrel(np.maximum(ung[0] * p + ung[1], 1e-8), g) if ung else float("nan")
        out.append({
            "sample_id": sid, "dataset": row["dataset"], "n_anchors": len(norm["anchors"]),
            "solver_status": sol["status"],
            "ar_raw": absrel(p, g),
            "oracle_a": a_o, "oracle_b": b_o, "oracle_absrel": ar_o,
            "ours_a": sol["a"] if gated_ok else None,
            "ours_b": sol["b"] if gated_ok else None,
            "ours_gated_absrel": ar_gated,
            "ungated_a": ung[0] if ung else None,
            "ungated_b": ung[1] if ung else None,
            "ungated_absrel": ar_ung,
            "log_a_ratio_ungated": (float(np.log(ung[0] / a_o)) if ung and a_o > 0 else None),
        })
    path = ROOT / "results" / "exploratory" / f"affine_oracle_vs_ours_{label}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")

    def med(k):
        v = [r[k] for r in out if r[k] is not None and np.isfinite(r[k])]
        return float(np.median(v)) if v else float("nan")

    ok = [r for r in out if r["solver_status"] == "ok"]
    print(f"\n=== {label}: submitted={len(out)} gated-ok={len(ok)}")
    print(f"  oracle affine      median {med('oracle_absrel'):.4f}")
    print(f"  ours gated (ok)    median {med('ours_gated_absrel'):.4f}" if ok else "  ours gated: none")
    print(f"  ours ungated       median {med('ungated_absrel'):.4f}")
    print(f"  raw A (no affine)  median {med('ar_raw'):.4f}")
    if ok:
        print("  per-image (oracle | gated | ungated | a_ours/a_oracle):")
        for r in ok:
            la = r["log_a_ratio_ungated"]
            print(f"    {r['sample_id']:14s} {r['oracle_absrel']:.3f} | "
                  f"{r['ours_gated_absrel']:.3f} | {r['ungated_absrel']:.3f} | "
                  f"{np.exp(la):.2f}x" if la is not None else "")
    return out


def ar_raw_of(d, gt, mask):
    m = mask & np.isfinite(d) & np.isfinite(gt) & (d > 0) & (gt > 0)
    return absrel(d[m], gt[m])


if __name__ == "__main__":
    run("manifests/pilot.jsonl", "pilot_f1_001", "pilot_f1")
    run("manifests/explor_20.jsonl", "explor_points_001", "explor_points")
    run("manifests/explor_20.jsonl", "explor_direct_001", "explor_direct")
