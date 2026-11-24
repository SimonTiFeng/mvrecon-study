"""Full-pixel recomputation: no subsampling anywhere.

Reruns every headline number from the diagnostic round using *all* valid pixels for
both parameter fitting and metric evaluation, plus multi-seed subsample sweeps to
show how much the 50k-pixel subsample could have moved anything.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.optimize import minimize, minimize_scalar

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config                    # noqa: E402
from src.evaluation.mono_oracle import (fit_affine as fit_affine_sub,
                                        fit_scale as fit_scale_sub)


def valid(pred, gt, mask):
    m = mask & np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    return pred[m].astype(np.float64), gt[m].astype(np.float64)


def absrel(z, g):
    return float(np.mean(np.abs(z - g) / g))


def est_scale_all_pixels(p, g):
    """min_a mean |a*p - g| / g  (all pixels), closed form not used: bounded search."""
    r = minimize_scalar(lambda a: absrel(a * p, g) if a > 0 else 1e9,
                        bounds=(1e-4, 1e4), method="bounded",
                        options={"xatol": 1e-8})
    return float(r.x), absrel(r.x * p, g)


def est_affine_all_pixels(p, g):
    """min_{a>0,b} mean |a*p + b - g| / g over ALL pixels, multistart L-BFGS-B."""
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
    a = float(np.exp(best.x[0])); b = float(best.x[1] * med)
    return a, b, absrel(a * p + b, g)


def main() -> None:
    cfg = load_config(ROOT / "configs" / "pilot.yaml")
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
        mB = json.loads((cache / f"{sid}__B.json").read_text(encoding="utf-8"))
        K = np.asarray(row["K"], float); Hp, Wp = mB["processed_shape"]
        dB *= ((K[0, 0] * Wp / row["width"]) + (K[1, 1] * Hp / row["height"])) / 2 / 300.0

        pA, gA = valid(dA, gt, mask)
        pB, gB = valid(dB, gt, mask)
        if pA.size < 100:
            continue
        a_s, ar_s = est_scale_all_pixels(pA, gA)
        a_f, b_f, ar_f = est_affine_all_pixels(pA, gA)
        aB_s, arB_s = est_scale_all_pixels(pB, gB)
        # subsampled variants at the configured 50k, over 5 seeds
        sub_scale, sub_aff = [], []
        for sd in range(5):
            fs = fit_scale_sub(dA, gt, mask, seed=sd)
            fa_ = fit_affine_sub(dA, gt, mask, seed=sd)
            sub_scale.append(fs["abs_rel"]); sub_aff.append(fa_["abs_rel"])
        per.append({
            "sample_id": sid, "dataset": row["dataset"],
            "n_pixels": int(pA.size),
            "A_raw": absrel(pA, gA),
            "A_scale_oracle_FULL": ar_s,
            "A_affine_oracle_FULL": ar_f,
            "A_affine_a": a_f, "A_affine_b": b_f,
            "A_scale_oracle_sub50k_mean": float(np.mean(sub_scale)),
            "A_affine_oracle_sub50k_mean": float(np.mean(sub_aff)),
            "A_affine_oracle_sub50k_max_dev": float(np.max(np.abs(np.array(sub_aff) - ar_f))),
            "B_raw": absrel(pB, gB),
            "B_scale_oracle_FULL": arB_s,
        })
    out = ROOT / "results" / "exploratory" / "full_pixel_recompute.json"
    out.write_text(json.dumps(per, indent=1), encoding="utf-8")

    def med(k, ds=None):
        v = [r[k] for r in per if ds is None or r["dataset"] == ds]
        return float(np.median(v)) if v else float("nan")

    summary = {
        "n_images": len(per),
        "total_pixels_evaluated": int(sum(r["n_pixels"] for r in per)),
        "median": {
            "A_scale_oracle_FULL": med("A_scale_oracle_FULL"),
            "A_affine_oracle_FULL": med("A_affine_oracle_FULL"),
            "A_affine_oracle_sub50k_mean": med("A_affine_oracle_sub50k_mean"),
            "A_affine_oracle_max_abs_dev_between_seeds": max(
                r["A_affine_oracle_sub50k_max_dev"] for r in per),
            "B_raw": med("B_raw"),
            "B_scale_oracle_FULL": med("B_scale_oracle_FULL"),
        },
        "by_dataset": {ds: {"n": sum(1 for r in per if r["dataset"] == ds),
                            "A_affine_oracle_FULL": med("A_affine_oracle_FULL", ds),
                            "A_scale_oracle_FULL": med("A_scale_oracle_FULL", ds),
                            "B_raw": med("B_raw", ds)}
                       for ds in sorted({r["dataset"] for r in per})},
        "count_A_affine_beats_B_raw": sum(1 for r in per
                                          if r["A_affine_oracle_FULL"] < r["B_raw"]),
    }
    (ROOT / "results" / "exploratory" / "full_pixel_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
