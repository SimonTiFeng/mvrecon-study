"""Attribute anchor scale errors: world-knowledge vs endpoint/backend geometry.

For each anchor with valid GT depth at both endpoints:
  s_model     = L_anchor / (3-D length from model depth)
  s_knowledge = L_anchor / (3-D length from GT depth)
  endpoint/backend bias = s_model / s_knowledge
Evaluator-only; never fed back to the measurement agent.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config                    # noqa: E402
from src.geometry.camera import Camera, sample_depth_bilinear  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--depth-cache", required=True)
    ap.add_argument("--backend", default="B")
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rows = [json.loads(l) for l in pathlib.Path(args.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    recs = []
    for row in rows:
        sid = row["sample_id"]
        nf = pathlib.Path(args.run) / "samples" / sid / "mcp_out" / "anchors" / f"{sid}.json"
        dz = pathlib.Path(args.depth_cache) / f"{sid}__{args.backend}.npz"
        if not nf.exists() or not dz.exists():
            continue
        norm = json.loads(nf.read_text(encoding="utf-8"))
        if norm.get("status") != "ok":
            continue
        d = np.load(dz)["depth"].astype(np.float64)
        mp = pathlib.Path(args.depth_cache) / f"{sid}__{args.backend}.json"
        meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
        factor = 1.0
        if cfg["backends"][args.backend].get("metric"):
            K = np.asarray(row["K"], float)
            Hp, Wp = meta.get("processed_shape", (0, 0))
            fx = K[0, 0] * Wp / row["width"]; fy = K[1, 1] * Hp / row["height"]
            factor = ((fx + fy) / 2) / 300.0
        d = d * factor
        gt = np.load(pathlib.Path(row["dataset_root"]) / row["gt"]).astype(np.float64)
        msk = np.load(pathlib.Path(row["dataset_root"]) / row["mask"]).astype(bool)
        K = np.asarray(row["K"], float)
        cam = Camera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], row["width"], row["height"],
                     model=row.get("camera_model", "PINHOLE"),
                     params=row.get("camera_params"))
        for a in norm.get("anchors", []):
            p1 = np.asarray(a["p1_xy"]); p2 = np.asarray(a["p2_xy"])
            dm, okm = sample_depth_bilinear(d, np.stack([p1, p2]))
            dg, okg = sample_depth_bilinear(gt, np.stack([p1, p2]), msk)
            if not (okm.all() and okg.all()):
                continue
            q1, q2 = cam.q(p1)[0], cam.q(p2)[0]
            lm = float(np.linalg.norm(dm[0] * q1 - dm[1] * q2))
            lg = float(np.linalg.norm(dg[0] * q1 - dg[1] * q2))
            if lm <= 0 or lg <= 0:
                continue
            recs.append({"sample_id": sid, "scene_id": row["scene_id"],
                         "dataset": row["dataset"], "anchor_id": a["anchor_id"],
                         "prior_kind": a["prior_kind"],
                         "length_m": a["length_m"], "l_model": lm, "l_gt": lg,
                         "s_model": a["length_m"] / lm,
                         "s_knowledge": a["length_m"] / lg,
                         # explicit direction: ratio >1 means the model's 3-D length is SHORTER than GT
                         "gt_over_model_length": lg / lm,
                         "model_over_gt_length": lm / lg})
    out = pathlib.Path(args.out or (pathlib.Path(args.run) / "diagnostics_anchors.csv"))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()) if recs else ["sample_id"])
        w.writeheader(); w.writerows(recs)
    if recs:
        lk = np.log([r["s_knowledge"] for r in recs])
        le = np.log([r["model_over_gt_length"] for r in recs])
        print(f"anchors={len(recs)} median s_knowledge={np.exp(np.median(lk)):.3f} "
              f"(p10,p90)=({np.exp(np.percentile(lk,10)):.3f},{np.exp(np.percentile(lk,90)):.3f}) "
              f"median model/GT length={np.exp(np.median(le)):.3f} "
              f"(<1 means model depth is short between endpoints)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
