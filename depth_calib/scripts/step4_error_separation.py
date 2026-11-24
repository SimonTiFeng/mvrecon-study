"""Step 4: separate size-knowledge, endpoint-localisation and depth-read errors.

Variants per committed anchor set (evaluator-only, GT never fed to agents):
  actual          : agent endpoints + agent claimed lengths
  gt_length       : agent endpoints + GT 3-D length between the same endpoints
                    (NOT a pure "world-knowledge oracle": it also rewrites the semantic
                     correspondence between a stated size and the chosen endpoints)
  oracle_affine / oracle_scale : GT-fitted global map (upper bound of the family)
Endpoint diagnostics:
  perturb2px      : relative change of the model 3-D endpoint distance under +/-2 px
  gt_patch_range  : relative GT-depth range in a 3x3 patch at each endpoint (contour check)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config                      # noqa: E402
from src.geometry.camera import Camera, sample_depth_bilinear  # noqa: E402
from src.geometry.depth_affine import fit_depth_affine, apply_depth_affine  # noqa: E402
from src.geometry.metric_scale import fit_metric_scale, apply_metric_scale  # noqa: E402
from src.geometry.quality import evaluate_endpoints     # noqa: E402
from src.geometry.types import AnchorGeom               # noqa: E402
from src.evaluation.metrics import evaluate_depth       # noqa: E402


def cam_for(row):
    K = np.asarray(row["K"], float)
    return Camera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], row["width"], row["height"],
                  model=row.get("camera_model", "PINHOLE"),
                  params=row.get("camera_params"))


def gt_length(cam, gt, mask, p1, p2):
    d, ok = sample_depth_bilinear(gt, np.stack([p1, p2]), mask)
    if not ok.all():
        return None
    q1, q2 = cam.q(p1)[0], cam.q(p2)[0]
    return float(np.linalg.norm(d[0] * q1 - d[1] * q2))


def make_geom(cam, a, depth, L, cfg, model_depth):
    p1 = np.asarray(a["p1_xy"], float); p2 = np.asarray(a["p2_xy"], float)
    dd, ok = sample_depth_bilinear(depth, np.stack([p1, p2]),
                                   eps=cfg["quality"]["depth_valid_eps"])
    g = AnchorGeom(anchor_id=a["anchor_id"], object_group_id=a["object_group_id"],
                   knowledge_group_id=a["knowledge_group_id"], p1=p1, p2=p2,
                   q1=cam.q(p1)[0], q2=cam.q(p2)[0], length_m=float(L))
    if ok.all():
        g.d1, g.d2 = float(dd[0]), float(dd[1])
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "manifests" / "pilot.jsonl"))
    ap.add_argument("--run", default=str(ROOT / "results" / "pilot_f1_001"))
    ap.add_argument("--cache", default=str(ROOT / "cache" / "hpc" / "depths"))
    ap.add_argument("--out", default=str(ROOT / "results" / "exploratory" / "step4_error_separation.json"))
    args = ap.parse_args()
    cfg = load_config(ROOT / "configs" / "pilot.yaml")
    rows = [json.loads(l) for l in pathlib.Path(args.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    cache = pathlib.Path(args.cache)
    per_image = []
    for row in rows:
        sid = row["sample_id"]
        nf = pathlib.Path(args.run) / "samples" / sid / "mcp_out" / "anchors" / f"{sid}.json"
        fa = cache / f"{sid}__A.npz"
        if not nf.exists() or not fa.exists():
            continue
        norm = json.loads(nf.read_text(encoding="utf-8"))
        if norm.get("status") != "ok" or not norm.get("anchors"):
            continue
        gt = np.load(pathlib.Path(row["dataset_root"]) / row["gt"]).astype(np.float64)
        mask = np.load(pathlib.Path(row["dataset_root"]) / row["mask"]).astype(bool)
        dA = np.load(fa)["depth"].astype(np.float64)
        cam = cam_for(row)

        anchors_actual, anchors_gtL = [], []
        ep_stats = []
        for a in norm["anchors"]:
            lg = gt_length(cam, gt, mask, np.asarray(a["p1_xy"], float),
                           np.asarray(a["p2_xy"], float))
            anchors_actual.append(make_geom(cam, a, dA, a["length_m"], cfg
                                            if False else cfg, True))
            if lg is not None:
                anchors_gtL.append(make_geom(cam, a, dA, lg, cfg, True))
            p1 = np.asarray(a["p1_xy"], float); p2 = np.asarray(a["p2_xy"], float)
            # endpoint perturbation on the MODEL depth
            dm, okm = sample_depth_bilinear(dA, np.stack([p1, p2]))
            rel = None
            if okm.all():
                q1, q2 = cam.q(p1)[0], cam.q(p2)[0]
                L0 = np.linalg.norm(dm[0] * q1 - dm[1] * q2)
                changes = []
                for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
                    pp = [p1.copy(), p2.copy()]
                    pp[0] += np.array([dx, dy], float)
                    dd, okp = sample_depth_bilinear(dA, np.stack(pp))
                    if okp.all():
                        Lp = np.linalg.norm(dd[0] * cam.q(pp[0])[0] - dd[1] * cam.q(pp[1])[0])
                        changes.append(abs(Lp - L0) / max(L0, 1e-12))
                rel = float(max(changes)) if changes else None
            # GT contour check: 3x3 depth range at each endpoint
            patch = []
            for p in (p1, p2):
                vals = []
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        v, ok = sample_depth_bilinear(gt, np.array([[p[0] + dx, p[1] + dy]]), mask)
                        if ok[0]:
                            vals.append(float(v[0]))
                patch.append((max(vals) - min(vals)) / max(min(vals), 1e-9) if vals else None)
            ep_stats.append({"anchor_id": a["anchor_id"],
                             "perturb2px_rel": rel,
                             "gt_patch_range": patch,
                             "on_contour": bool((patch[0] or 0) > 0.05 or (patch[1] or 0) > 0.05)})

        # A: actual vs gt-length anchors
        sol_act = fit_depth_affine(anchors_actual, cfg)
        sol_gtL = fit_depth_affine(anchors_gtL, cfg)
        pred_act = apply_depth_affine(dA, sol_act["a"], sol_act["b"])
        pred_gtL = apply_depth_affine(dA, sol_gtL["a"], sol_gtL["b"])
        m_act = evaluate_depth(pred_act, gt, mask)["abs_rel"]
        m_gtL = evaluate_depth(pred_gtL, gt, mask)["abs_rel"]
        per_image.append({
            "sample_id": sid, "dataset": row["dataset"],
            "n_anchors": len(norm["anchors"]),
            "A_actual_absrel": m_act,
            "A_gtlen_absrel": m_gtL,
            "A_actual_status": sol_act["status"], "A_gtlen_status": sol_gtL["status"],
            "endpoint_stats": ep_stats,
        })
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(per_image, indent=1, default=float), encoding="utf-8")

    def med(key):
        v = [r[key] for r in per_image if np.isfinite(r[key])]
        return float(np.median(v)) if v else float("nan")
    def mean(key):
        v = [r[key] for r in per_image if np.isfinite(r[key])]
        return float(np.mean(v)) if v else float("nan")

    perturb = [s["perturb2px_rel"] for r in per_image for s in r["endpoint_stats"]
               if s["perturb2px_rel"] is not None]
    contour = sum(1 for r in per_image for s in r["endpoint_stats"] if s["on_contour"])
    n_ep = sum(len(r["endpoint_stats"]) for r in per_image)
    print(json.dumps({
        "n_images": len(per_image),
        "A_actual_absrel_median": med("A_actual_absrel"),
        "A_gtlen_absrel_median": med("A_gtlen_absrel"),
        "A_actual_absrel_mean": mean("A_actual_absrel"),
        "A_gtlen_absrel_mean": mean("A_gtlen_absrel"),
        "endpoints": n_ep,
        "perturb2px_rel_median": float(np.median(perturb)) if perturb else None,
        "perturb2px_rel_p90": float(np.percentile(perturb, 90)) if perturb else None,
        "endpoints_on_contour": contour,
    }, indent=1))


if __name__ == "__main__":
    main()
