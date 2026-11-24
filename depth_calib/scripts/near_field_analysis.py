"""Near-field analysis: how Mono-vs-Metric and the affine behave as depth shrinks.

No new images are invented. The datasets contain no frames whose 5th-percentile depth is
below ~0.7 m (NYUv2) / ~0.85 m (ETH3D), so "near field" here means: rank all frames by their
near-depth percentile and evaluate in bands. Within each frame we additionally report the
error restricted to the nearest pixels (<0.5 m, <1 m) where such pixels exist.

Reads config for every numeric setting; no hidden defaults.
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

GEOM_ROOT = {"nyuv2": "E:/depth_calib_data/processed/nyuv2",
             "eth3d": "E:/depth_calib_data/processed/eth3d_final"}


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


def main() -> None:
    cfg = load_config(ROOT / "configs" / "pilot.yaml")
    near_band = cfg.get("data", {}).get("near_field_band_m", [0.5, 1.0])
    cache = ROOT / "cache" / "hpc" / "depths"
    runs = [("pilot_f1_001", "manifests/pilot.jsonl"),
            ("explor_points_001", "manifests/explor_20.jsonl"),
            ("explor_direct_001", "manifests/explor_20.jsonl")]
    seen = set()
    recs = []
    for run_dir, manifest in runs:
        rows = {json.loads(l)["sample_id"]: json.loads(l) for l in
                (ROOT / manifest).read_text(encoding="utf-8").splitlines() if l.strip()}
        for sid, row in rows.items():
            if sid in seen:
                continue
            fa, fb = cache / f"{sid}__A.npz", cache / f"{sid}__B.npz"
            if not (fa.exists() and fb.exists()):
                continue
            gt = np.load(pathlib.Path(row["dataset_root"]) / row["gt"]).astype(np.float64)
            mask = np.load(pathlib.Path(row["dataset_root"]) / row["mask"]).astype(bool)
            K = np.asarray(row["K"], float)
            dA = np.load(fa)["depth"].astype(np.float64)
            dB = np.load(fb)["depth"].astype(np.float64)
            mB = json.loads((cache / f"{sid}__B.json").read_text(encoding="utf-8"))
            Hp, Wp = mB["processed_shape"]
            dB *= ((K[0, 0] * Wp / row["width"]) + (K[1, 1] * Hp / row["height"])) / 2 / 300.0
            m = mask & np.isfinite(gt) & (gt > 0) & np.isfinite(dA) & (dA > 0) \
                & np.isfinite(dB) & (dB > 0)
            if m.sum() < 100:
                continue
            g = gt[m]; pA = dA[m]; pB = dB[m]
            # near-field masks inside the frame
            near05 = g < near_band[0]
            near10 = g < near_band[1]
            _, _, ar_orc = oracle_affine(pA, g)
            rec = {
                "sample_id": sid, "dataset": row["dataset"],
                "near_p5": float(np.percentile(g, 5)),
                "near_p25": float(np.percentile(g, 25)),
                "median_depth": float(np.median(g)),
                "n_pix": int(g.size),
                "frac_lt_0p5": float(near05.mean()),
                "frac_lt_1p0": float(near10.mean()),
                "B_raw_ALL": absrel(pB, g),
                "A_oracle_ALL": ar_orc,
                "B_raw_lt1m": absrel(pB[near10], g[near10]) if near10.sum() > 50 else None,
                "A_oracle_lt1m": None,
                "n_lt1m": int(near10.sum()),
            }
            if near10.sum() > 50:
                a, b, _ = oracle_affine(pA[near10], g[near10])
                rec["A_oracle_lt1m"] = absrel(a * pA[near10] + b, g[near10])
            if near05.sum() > 50:
                rec["B_raw_lt0p5"] = absrel(pB[near05], g[near05])
                a, b, _ = oracle_affine(pA[near05], g[near05])
                rec["A_oracle_lt0p5"] = absrel(a * pA[near05] + b, g[near05])
                rec["n_lt0p5"] = int(near05.sum())
            recs.append(rec)
            seen.add(sid)

    recs.sort(key=lambda r: r["near_p5"])
    out = ROOT / "results" / "exploratory" / "near_field.json"
    out.write_text(json.dumps(recs, indent=1), encoding="utf-8")

    n = len(recs)
    terciles = [recs[: n // 3], recs[n // 3: 2 * n // 3], recs[2 * n // 3:]]

    def med(rs, k):
        v = [r[k] for r in rs if r.get(k) is not None and np.isfinite(r[k])]
        return float(np.median(v)) if v else float("nan")

    print(f"frames scored: {n}")
    print(f"{'band by near_p5':22s} {'n':>3s} {'p5 range':>16s} {'B raw':>7s} {'A oracle':>9s}")
    for name, band in zip(("nearest third", "middle third", "farthest third"), terciles):
        print(f"{name:22s} {len(band):3d} "
              f"{min(r['near_p5'] for r in band):.2f}-{max(r['near_p5'] for r in band):.2f} m"
              f"   {med(band,'B_raw_ALL'):7.4f} {med(band,'A_oracle_ALL'):9.4f}")

    lt1 = [r for r in recs if r.get("B_raw_lt1m") is not None]
    lt05 = [r for r in recs if r.get("B_raw_lt0p5") is not None]
    print(f"\npixels nearer than 1.0 m (present in {len(lt1)} frames):")
    print("  B raw     median AbsRel %.4f" % med(lt1, "B_raw_lt1m"))
    print("  A oracle  median AbsRel %.4f" % med(lt1, "A_oracle_lt1m"))
    if lt05:
        print(f"pixels nearer than 0.5 m (present in {len(lt05)} frames, "
              f"max coverage {max(r['frac_lt_0p5'] for r in lt05):.0%}):")
        print("  B raw     median AbsRel %.4f" % med(lt05, "B_raw_lt0p5"))
        print("  A oracle  median AbsRel %.4f" % med(lt05, "A_oracle_lt0p5"))
    else:
        print("pixels nearer than 0.5 m: NONE in any frame "
              "(min near_p5 = %.2f m)" % min(r["near_p5"] for r in recs))


if __name__ == "__main__":
    main()
