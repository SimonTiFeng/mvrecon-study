"""Mono oracle table + monotone check on the worst images (exploratory, no VLM)."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.mono_oracle import mono_oracle_report  # noqa: E402
from src.evaluation.oracle import (optimal_global_scale_absrel,
                                   optimal_global_scale_rmse)  # noqa: E402
from src.evaluation.metrics import evaluate_depth  # noqa: E402


def full_absrel(z, gt, mask):
    m = mask & np.isfinite(z) & np.isfinite(gt) & (z > 0) & (gt > 0)
    return float(np.mean(np.abs(z[m] - gt[m]) / gt[m])) if m.any() else float("nan")


def main() -> None:
    rows = [json.loads(l) for l in
            (ROOT / "manifests" / "pilot.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    cache = ROOT / "cache" / "hpc" / "depths"
    per = []
    for row in rows:
        sid = row["sample_id"]
        fa, fb = cache / f"{sid}__A.npz", cache / f"{sid}__B.npz"
        if not (fa.exists() and fb.exists()):
            continue
        gt = np.load(pathlib.Path(row["dataset_root"]) / row["gt"]).astype(np.float64)
        mask = np.load(pathlib.Path(row["dataset_root"]) / row["mask"]).astype(bool)
        dA = np.load(fa)["depth"].astype(np.float64)
        dB = np.load(fb)["depth"].astype(np.float64)
        metaB = json.loads((cache / f"{sid}__B.json").read_text(encoding="utf-8"))
        K = np.asarray(row["K"], float)
        Hp, Wp = metaB["processed_shape"]
        fB = ((K[0, 0] * Wp / row["width"]) + (K[1, 1] * Hp / row["height"])) / 2 / 300.0
        Bn = dB * fB
        orc = mono_oracle_report(dA, gt, mask, run_monotone=False)
        rec = {
            "sample_id": sid, "dataset": row["dataset"], "scene_id": row["scene_id"],
            "A_scale_absrel": full_absrel(orc["scale"]["a"] * dA, gt, mask),
            "A_affine_absrel": full_absrel(orc["affine"]["a"] * dA + orc["affine"]["b"],
                                           gt, mask),
            "B_raw_absrel": full_absrel(Bn, gt, mask),
            "B_scale_oracle_absrel": full_absrel(
                Bn * optimal_global_scale_absrel(Bn, gt, mask), gt, mask),
            "affine_a": orc["affine"]["a"], "affine_b": orc["affine"]["b"],
        }
        per.append(rec)

    def med(key, ds=None):
        v = [r[key] for r in per if ds is None or r["dataset"] == ds]
        return float(np.median(v)) if v else float("nan")

    def pct(key, q):
        return float(np.percentile([r[key] for r in per], q))

    summary = {
        "n": len(per),
        "medians": {
            "A_scale_oracle": med("A_scale_absrel"),
            "A_affine_oracle": med("A_affine_absrel"),
            "B_raw": med("B_raw_absrel"),
            "B_scale_oracle": med("B_scale_oracle_absrel"),
        },
        "A_affine_oracle_p10_p90": [pct("A_affine_absrel", 10), pct("A_affine_absrel", 90)],
        "count_A_affine_worse_than_B_raw": sum(
            1 for r in per if r["A_affine_absrel"] > r["B_raw_absrel"]),
        "by_dataset": {ds: {"n": sum(1 for r in per if r["dataset"] == ds),
                            "A_affine_oracle": med("A_affine_absrel", ds),
                            "A_scale_oracle": med("A_scale_absrel", ds),
                            "B_raw": med("B_raw_absrel", ds)}
                       for ds in sorted({r["dataset"] for r in per})},
    }
    # monotone check on the 5 worst A-affine images (only if genuinely poor)
    worst = sorted(per, key=lambda r: -r["A_affine_absrel"])[:5]
    mono = []
    if worst and worst[0]["A_affine_absrel"] > 0.08:
        for r in worst:
            row = [x for x in rows if x["sample_id"] == r["sample_id"]][0]
            gt = np.load(pathlib.Path(row["dataset_root"]) / row["gt"]).astype(np.float64)
            mask = np.load(pathlib.Path(row["dataset_root"]) / row["mask"]).astype(bool)
            dA = np.load(cache / f"{r['sample_id']}__A.npz")["depth"].astype(np.float64)
            o = mono_oracle_report(dA, gt, mask, run_monotone=True)
            mono.append({"sample_id": r["sample_id"],
                         "affine": r["A_affine_absrel"],
                         "monotone_holdout": o["monotone_loglin"]["abs_rel_holdout"],
                         "knots": o["monotone_loglin"].get("knots")})
    summary["monotone_worst5"] = mono
    out = ROOT / "results" / "exploratory" / "mono_oracle_table.json"
    out.write_text(json.dumps({"summary": summary, "per_image": per}, indent=1),
                   encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
