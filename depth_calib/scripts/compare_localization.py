"""Fixed-target endpoint-localisation comparison (points vs direct), evaluator-side.

For the four pre-declared targets: compare each run's reported endpoints against the
verifier reference endpoints (pixel distance, best of the two pairings), and report
candidate-point coverage (nearest registered candidate to each reference endpoint)
for the points condition.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib

import numpy as np


def load_anchor_endpoints(run: str, sid: str):
    base = pathlib.Path("results") / run / "samples" / sid / "mcp_out"
    f = base / "anchors" / f"{sid}.json"
    if not f.exists():
        return None
    norm = json.loads(f.read_text(encoding="utf-8"))
    if norm.get("status") != "ok":
        return {"status": norm.get("status"), "anchors": []}
    return {"status": "ok", "anchors": norm.get("anchors", [])}


def endpoint_pair_error(a, ref):
    p1 = np.asarray(a["p1_xy"], float); p2 = np.asarray(a["p2_xy"], float)
    r1 = np.asarray(ref[0], float); r2 = np.asarray(ref[1], float)
    d1 = np.linalg.norm(p1 - r1) + np.linalg.norm(p2 - r2)
    d2 = np.linalg.norm(p1 - r2) + np.linalg.norm(p2 - r1)
    return float(min(d1, d2) / 2.0)      # mean per-endpoint error


def candidate_coverage(run: str, sid: str, ref):
    f = pathlib.Path("results") / run / "samples" / sid / "mcp_out" / "state" / f"{sid}.usage.json"
    if not f.exists():
        return None
    u = json.loads(f.read_text(encoding="utf-8"))
    pts = u.get("point_registry", {})
    if not pts:
        return {"n_candidates": 0}
    P = np.array(list(pts.values()), float)
    out = {"n_candidates": len(P)}
    for i, r in enumerate(ref):
        d = np.linalg.norm(P - np.asarray(r, float), axis=1)
        out[f"nearest_candidate_px_ref{i+1}"] = float(d.min())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--targets", default="results/exploratory/fixed_targets.json")
    ap.add_argument("--out", default="results/exploratory/localization_comparison.json")
    args = ap.parse_args()
    t = json.loads(pathlib.Path(args.targets).read_text(encoding="utf-8"))
    refs = t["reference_endpoints_verifier_only"]
    out = {"targets": t["targets"], "reference": refs, "runs": {}}
    for run in args.runs:
        rows = {}
        for sid, ref in refs.items():
            a = load_anchor_endpoints(run, sid)
            if a is None:
                rows[sid] = {"status": "missing"}
                continue
            if a["status"] != "ok" or not a["anchors"]:
                rows[sid] = {"status": a["status"], "endpoint_error_px": None}
                continue
            errs = [endpoint_pair_error(x, ref) for x in a["anchors"]]
            rows[sid] = {
                "status": "ok",
                "n_anchors": len(a["anchors"]),
                "endpoint_error_px": float(min(errs)),
                "per_anchor_endpoint_error_px": errs,
                "claimed_lengths_m": [x["length_m"] for x in a["anchors"]],
                "candidate_coverage": candidate_coverage(run, sid, ref),
            }
        out["runs"][run] = rows
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    # compact print
    for run, rows in out["runs"].items():
        print(f"== {run}")
        for sid, r in rows.items():
            print(f"  {sid}: {r.get('status')} err_px={r.get('endpoint_error_px')} "
                  f"cov={r.get('candidate_coverage')}")


if __name__ == "__main__":
    main()
