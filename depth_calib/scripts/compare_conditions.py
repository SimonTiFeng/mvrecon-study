"""Paired condition comparison: candidate-points vs direct-coordinates.

Reports per condition: submission rate, anchors, budget usage, and (where anchors
exist) calibration outcome. Also the fixed-target endpoint-error comparison.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics

import numpy as np


def scan(run: str, ids: list[str]) -> dict:
    root = pathlib.Path("results") / run / "samples"
    rows = {"n": len(ids), "ok": 0, "abstain": 0, "missing": 0,
            "anchors": 0, "vis": [], "crops": [], "att": [], "per_sample": {}}
    for sid in ids:
        sd = root / sid
        st = sd / "mcp_out" / "state" / f"{sid}.state.json"
        us = sd / "mcp_out" / "state" / f"{sid}.usage.json"
        anc_f = sd / "mcp_out" / "anchors" / f"{sid}.json"
        rec = {"committed": st.exists(), "status": None, "n_anchors": 0,
               "vis": 0, "crops": 0, "att": 0}
        if us.exists():
            u = json.loads(us.read_text(encoding="utf-8"))
            rec.update(vis=u.get("visual_calls", 0), crops=u.get("crop_calls", 0),
                       att=u.get("submission_attempts", 0))
            rows["vis"].append(rec["vis"]); rows["crops"].append(rec["crops"])
            rows["att"].append(rec["att"])
        if anc_f.exists():
            a = json.loads(anc_f.read_text(encoding="utf-8"))
            rec["status"] = a.get("status")
            rec["n_anchors"] = len(a.get("anchors") or [])
            rows["anchors"] += rec["n_anchors"]
            if a.get("status") == "ok":
                rows["ok"] += 1
            else:
                rows["abstain"] += 1
        else:
            rows["missing"] += 1
        rows["per_sample"][sid] = rec
    return rows


def fmt(name: str, r: dict) -> str:
    n = max(r["n"], 1)
    vis = statistics.mean(r["vis"]) if r["vis"] else 0.0
    return (f"| {name} | {r['n']} | {r['ok']} ({r['ok']/n:.0%}) | {r['abstain']} | "
            f"{r['missing']} | {r['anchors']/n:.2f} | {vis:.2f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/explor_20.jsonl")
    ap.add_argument("--runs", nargs="+", default=["explor_points_001", "explor_direct_001"])
    ap.add_argument("--out", default="results/exploratory/condition_comparison.json")
    args = ap.parse_args()
    ids = [json.loads(l)["sample_id"] for l in
           pathlib.Path(args.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    res = {r: scan(r, ids) for r in args.runs}
    lines = ["| condition | n | submitted ok | abstain | missing | anchors/img | mean obs calls |",
             "|---|---|---|---|---|---|---|"]
    for r, v in res.items():
        lines.append(fmt(r, v))
    table = "\n".join(lines)
    print(table)
    pathlib.Path(args.out).write_text(
        json.dumps({"table": table, "runs": res}, indent=1), encoding="utf-8")
    # paired: both submitted ok
    if len(args.runs) == 2:
        a, b = args.runs
        both = [s for s in ids
                if res[a]["per_sample"][s]["status"] == "ok"
                and res[b]["per_sample"][s]["status"] == "ok"]
        only_a = [s for s in ids if res[a]["per_sample"][s]["status"] == "ok"
                  and res[b]["per_sample"][s]["status"] != "ok"]
        only_b = [s for s in ids if res[b]["per_sample"][s]["status"] == "ok"
                  and res[a]["per_sample"][s]["status"] != "ok"]
        print(f"\nboth ok: {len(both)} {both}")
        print(f"only {a}: {len(only_a)} {only_a}")
        print(f"only {b}: {len(only_b)} {only_b}")


if __name__ == "__main__":
    main()
