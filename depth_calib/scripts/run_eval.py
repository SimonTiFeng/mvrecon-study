"""Evaluate one Codex run against GT (evaluator-side, GT never leaves this side).

Rows per sample x system variant; per-scene aggregation; paired bootstrap on
per-image deltas (handoff 10). Oracle variants are analysis-only.
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

from src.config import load_config                      # noqa: E402
from src.evaluation.metrics import evaluate_depth, paired_bootstrap  # noqa: E402
from src.evaluation.oracle import (optimal_depth_affine,
                                   optimal_global_scale_absrel,
                                   optimal_global_scale_rmse)  # noqa: E402
from src.geometry.camera import Camera                  # noqa: E402
from src.geometry.pipeline import calibrate              # noqa: E402


def _load_npz(p: pathlib.Path):
    with np.load(p) as z:
        return z["depth"], z["conf"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--run", required=True, help="results/<run_id> dir")
    ap.add_argument("--depth-cache", default=str(ROOT / "cache" / "depths"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_dir = pathlib.Path(args.run)
    out = pathlib.Path(args.out or (run_dir / "eval"))
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in pathlib.Path(args.manifest).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    cache = pathlib.Path(args.depth_cache)

    per_image = []
    for row in rows:
        sid = row["sample_id"]
        sdir = run_dir / "samples" / sid
        norm_path = sdir / "mcp_out" / "anchors" / f"{sid}.json"
        run_status = {}
        if (sdir / "runner_status.json").exists():
            run_status = json.loads((sdir / "runner_status.json").read_text(encoding="utf-8"))
        gt = np.load(pathlib.Path(row["dataset_root"]) / row["gt"]).astype(np.float64)
        mask = np.load(pathlib.Path(row["dataset_root"]) / row["mask"]).astype(bool)
        K = np.asarray(row["K"], dtype=float)
        if row.get("camera_model"):
            cam = Camera(K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                         row["width"], row["height"],
                         model=row["camera_model"], params=row["camera_params"])
        else:
            cam = Camera(K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                         row["width"], row["height"])

        if not norm_path.exists():
            per_image.append({"sample_id": sid, "scene_id": row["scene_id"],
                              "dataset": row["dataset"], "system": "submission_missing",
                              "status": run_status.get("runner_status", "unknown"),
                              "abs_rel": "", "rmse_m": "", "delta1": "",
                              "coverage": 0, "fallback": "", "notes": "no committed submission"})
            continue
        norm = json.loads(norm_path.read_text(encoding="utf-8"))

        for letter, spec in cfg["backends"].items():
            if letter not in cfg["active_backends"]:
                continue
            dz = cache / f"{sid}__{letter}.npz"
            if not dz.exists():
                continue
            raw_depth, _ = _load_npz(dz)
            meta_p = cache / f"{sid}__{letter}.json"
            meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
            cal = calibrate(norm=norm, depth=raw_depth, cam=cam, cfg=cfg,
                            backend_spec=spec,
                            processed_shape=tuple(meta.get("processed_shape", (0, 0))))

            variants = {}
            if spec.get("metric"):
                factor = cal.get("backend_metric_conversion_factor", 1.0)
                native = np.asarray(raw_depth, dtype=np.float64) * factor
                variants[f"{letter}_metric_raw"] = native
                variants[f"{letter}_unconverted"] = raw_depth   # diagnostic only
                variants[f"{letter}_calib_plain"] = cal.get("pred_depth_plain")
                variants[f"{letter}_calib_consistent"] = cal.get("pred_depth_consistent")
                variants[f"{letter}_calib_conservative"] = cal.get("pred_depth_conservative")
                s_ls = optimal_global_scale_rmse(native, gt, mask)
                variants[f"{letter}_oracle_scale_ls"] = native * s_ls
                s_ar = optimal_global_scale_absrel(native, gt, mask)
                variants[f"{letter}_oracle_scale_absrel"] = native * s_ar
            else:
                variants[f"{letter}_raw_unscaled"] = raw_depth
                variants[f"{letter}_calib_affine"] = cal.get("pred_depth")

            for name, pred in variants.items():
                if pred is None:
                    continue
                if name.endswith(("raw_unscaled", "unconverted")):
                    rec = {"sample_id": sid, "scene_id": row["scene_id"],
                           "dataset": row["dataset"], "system": name,
                           "status": run_status.get("runner_status", ""),
                           "abs_rel": "", "rmse_m": "", "delta1": "",
                           "coverage": "", "fallback": "",
                           "notes": "not a metric prediction; excluded from main table"}
                    per_image.append(rec)
                    continue
                met = evaluate_depth(pred, gt, mask,
                                     cfg["evaluation"]["delta_thresholds"])
                per_image.append({
                    "sample_id": sid, "scene_id": row["scene_id"],
                    "dataset": row["dataset"], "system": name,
                    "status": run_status.get("runner_status", ""),
                    "abs_rel": met["abs_rel"], "rmse_m": met["rmse_m"],
                    "delta1": met["delta1"],
                    "coverage": cal.get("coverage", ""),
                    "fallback": int(bool(cal.get("fallback", False))),
                    "solver_status": (cal.get("solver") or {}).get("status", ""),
                    "n_anchors": len(norm.get("anchors", [])),
                    "notes": ""})

        # A with B fallback (complete system; handoff 6.6 / 9.1)
        a_pred = None
        b_pred = None
        if (cache / f"{sid}__A.npz").exists() and (cache / f"{sid}__B.npz").exists():
            dA, _ = _load_npz(cache / f"{sid}__A.npz")
            dB, _ = _load_npz(cache / f"{sid}__B.npz")
            metaB = json.loads((cache / f"{sid}__B.json").read_text(encoding="utf-8"))
            calA = calibrate(norm=norm, depth=dA, cam=cam, cfg=cfg,
                             backend_spec=cfg["backends"]["A"])
            calB = calibrate(norm=norm, depth=dB, cam=cam, cfg=cfg,
                             backend_spec=cfg["backends"]["B"],
                             processed_shape=tuple(metaB["processed_shape"]))
            a_pred = calA.get("pred_depth")
            b_pred = calB.get("pred_depth")
            if a_pred is not None and b_pred is not None:
                use_b = bool(calA.get("fallback")) or norm.get("status") != "ok"
                pred = b_pred if use_b else a_pred
                met = evaluate_depth(pred, gt, mask,
                                     cfg["evaluation"]["delta_thresholds"])
                per_image.append({
                    "sample_id": sid, "scene_id": row["scene_id"],
                    "dataset": row["dataset"], "system": "A_calib_B_fallback",
                    "status": run_status.get("runner_status", ""),
                    "abs_rel": met["abs_rel"], "rmse_m": met["rmse_m"],
                    "delta1": met["delta1"], "coverage": 1.0,
                    "fallback": int(use_b), "solver_status": "mixed",
                    "n_anchors": len(norm.get("anchors", [])),
                    "notes": "B fallback when A affine fails"})

    # ---- write tables
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in per_image for k in r}))
        w.writeheader(); w.writerows(per_image)

    summary = {}
    systems = sorted({r["system"] for r in per_image})
    for s in systems:
        vals = [r["abs_rel"] for r in per_image
                if r["system"] == s and isinstance(r["abs_rel"], float)
                and np.isfinite(r["abs_rel"])]
        cob = paired_bootstrap(np.array(vals), cfg["evaluation"]["bootstrap_samples"],
                               cfg["evaluation"]["bootstrap_seed"])
        per_ds = {}
        for ds in sorted({r["dataset"] for r in per_image if r["system"] == s}):
            vv = [r["abs_rel"] for r in per_image
                  if r["system"] == s and r["dataset"] == ds
                  and isinstance(r["abs_rel"], float) and np.isfinite(r["abs_rel"])]
            if vv:
                per_ds[ds] = {"abs_rel_mean": float(np.mean(vv)), "n": len(vv)}
        summary[s] = {"abs_rel": cob, "by_dataset": per_ds}

    # paired deltas vs B_raw
    if any(r["system"] == "B_metric_raw" for r in per_image):
        base = {r["sample_id"]: r["abs_rel"] for r in per_image
                if r["system"] == "B_metric_raw" and isinstance(r["abs_rel"], float)}
        deltas = {}
        for s in systems:
            if s == "B_metric_raw":
                continue
            cur = {r["sample_id"]: r["abs_rel"] for r in per_image
                   if r["system"] == s and isinstance(r["abs_rel"], float)}
            common = sorted(set(base) & set(cur))
            if common:
                d = np.array([base[k] - cur[k] for k in common])
                deltas[s] = {"delta_abs_rel": paired_bootstrap(
                    d, cfg["evaluation"]["bootstrap_samples"],
                    cfg["evaluation"]["bootstrap_seed"]),
                    "n_pairs": len(common),
                    "improved": int((d > 0).sum()), "worse": int((d < 0).sum())}
        summary["_paired_vs_B_metric_raw"] = deltas

    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1)[:4000])


if __name__ == "__main__":
    main()
