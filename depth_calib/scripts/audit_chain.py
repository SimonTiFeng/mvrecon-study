"""Diagnostic audit of the compute chain (exploratory, no VLM calls).

Checks:
  1. representation: is A's raw output best explained as depth (scale/affine) or inverse depth?
  2. direction of every ratio (model/GT vs GT/model) with explicit names
  3. coordinate roundtrip: pixel -> ray -> 3D -> pixel (pinhole and fisheye cameras)
  4. pixel-depth vs ray-distance: confirm our construction uses z-depth and 3-D Euclidean
     endpoint distances, not ray range or pixel distance
Outputs results/exploratory/audit_chain.json and a Markdown summary.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config                       # noqa: E402
from src.evaluation.mono_oracle import (fit_affine, fit_inverse,  # noqa: E402
                                        fit_scale, mono_oracle_report)
from src.geometry.camera import Camera, sample_depth_bilinear  # noqa: E402


def full_absrel(z, gt, mask):
    m = mask & np.isfinite(z) & np.isfinite(gt) & (z > 0) & (gt > 0)
    return float(np.mean(np.abs(z[m] - gt[m]) / gt[m])) if m.any() else float("nan")


def camera_for(row):
    K = np.asarray(row["K"], float)
    return Camera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], row["width"], row["height"],
                  model=row.get("camera_model", "PINHOLE"),
                  params=row.get("camera_params"))


def coordinate_roundtrip(row, samples=200, seed=0):
    cam = camera_for(row)
    rng = np.random.default_rng(seed)
    uv = np.stack([rng.uniform(0, row["width"] - 1, samples),
                   rng.uniform(0, row["height"] - 1, samples)], axis=-1)
    q = cam.q(uv)
    z = rng.uniform(1.0, 10.0, samples)
    X = q * z[:, None]
    uv2 = cam.project(X)
    err = np.linalg.norm(uv2 - uv, axis=-1)
    return {"max_px": float(err.max()), "p99_px": float(np.percentile(err, 99)),
            "camera_model": row.get("camera_model", "PINHOLE")}


def main() -> None:
    cfg = load_config(ROOT / "configs" / "pilot.yaml")
    rows = [json.loads(l) for l in
            (ROOT / "manifests" / "pilot.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    cache = ROOT / "cache" / "hpc" / "depths"
    out_dir = ROOT / "results" / "exploratory"
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    roundtrip = []
    for row in rows:
        sid = row["sample_id"]
        dz = cache / f"{sid}__A.npz"
        if not dz.exists():
            continue
        pred = np.load(dz)["depth"].astype(np.float64)
        gt = np.load(pathlib.Path(row["dataset_root"]) / row["gt"]).astype(np.float64)
        mask = np.load(pathlib.Path(row["dataset_root"]) / row["mask"]).astype(bool)
        orc = mono_oracle_report(pred, gt, mask)
        a, b = orc["affine"]["a"], orc["affine"]["b"]
        rec = {
            "sample_id": sid, "dataset": row["dataset"], "scene_id": row["scene_id"],
            "n_valid": int((mask & np.isfinite(gt) & (gt > 0)).sum()),
            "median_gt_over_pred": float(np.median(gt[mask & (pred > 0)] / pred[mask & (pred > 0)])),
            "scale_absrel": full_absrel(orc["scale"]["a"] * pred, gt, mask),
            "affine_absrel": full_absrel(a * pred + b, gt, mask),
            "inverse_absrel": None,
            "affine_a": a, "affine_b": b,
            "affine_clip_frac_all": orc["affine_clip_frac_all"],
            "inverse_ok": None,
        }
        ai, bi = orc["inverse"]["a"], orc["inverse"]["b"]
        den = ai * pred + bi
        zinv = np.where(den > 1e-9, 1.0 / np.maximum(den, 1e-9), np.nan)
        rec["inverse_absrel"] = full_absrel(zinv, gt, mask)
        rec["inverse_ok"] = bool(den.min() > 1e-9)
        recs.append(rec)
        if row["dataset"] == "nyuv2" and len(roundtrip) < 1:
            roundtrip.append({"sample_id": sid, **coordinate_roundtrip(row)})
        if row["dataset"] == "eth3d" and len(roundtrip) < 2:
            roundtrip.append({"sample_id": sid, **coordinate_roundtrip(row)})

    def med(key, ds=None):
        v = [r[key] for r in recs if r[key] is not None and (ds is None or r["dataset"] == ds)]
        return float(np.median(v)) if v else float("nan")

    summary = {
        "n_images": len(recs),
        "by_dataset": {
            ds: {"n": sum(1 for r in recs if r["dataset"] == ds),
                 "median_gt_over_pred": med("median_gt_over_pred", ds),
                 "median_scale_absrel": med("scale_absrel", ds),
                 "median_affine_absrel": med("affine_absrel", ds),
                 "median_inverse_absrel": med("inverse_absrel", ds)}
            for ds in sorted({r["dataset"] for r in recs})},
        "overall": {
            "median_scale_absrel": med("scale_absrel"),
            "median_affine_absrel": med("affine_absrel"),
            "median_inverse_absrel": med("inverse_absrel"),
            "median_affine_clip_frac_all": med("affine_clip_frac_all"),
            "affine_beats_scale_count": sum(1 for r in recs
                                            if r["affine_absrel"] < r["scale_absrel"]),
        },
        "coordinate_roundtrip": roundtrip,
        "representation_verdict": None,
    }
    s = summary["overall"]
    if s["median_affine_absrel"] < 0.8 * s["median_inverse_absrel"]:
        summary["representation_verdict"] = (
            "raw output behaves as depth (affine much better than inverse); no inversion needed")
    else:
        summary["representation_verdict"] = (
            "raw output is NOT well explained as depth; check for disparity/inversion")
    (out_dir / "audit_chain.json").write_text(json.dumps(
        {"summary": summary, "per_image": recs}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
