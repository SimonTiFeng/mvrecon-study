"""Compare two measurement runs (e.g. F0 direct vs F1 harness) on one manifest."""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics


def stats(run: str, manifest_ids: set[str]) -> dict:
    root = pathlib.Path("results") / run / "samples"
    sub = {"n": 0, "ok": 0, "abstain": 0, "anchors": 0,
           "visual_calls": [], "crops": [], "attempts": [], "missing": 0}
    for sid in sorted(manifest_ids):
        sd = root / sid
        st = sd / "mcp_out" / "state" / f"{sid}.state.json"
        us = sd / "mcp_out" / "state" / f"{sid}.usage.json"
        if not st.exists():
            sub["missing"] += 1
            continue
        s = json.loads(st.read_text(encoding="utf-8"))
        sub["n"] += 1
        if s.get("status") == "ok":
            sub["ok"] += 1
            sub["anchors"] += s.get("anchor_count", 0)
        else:
            sub["abstain"] += 1
        if us.exists():
            u = json.loads(us.read_text(encoding="utf-8"))
            sub["visual_calls"].append(u["visual_calls"])
            sub["crops"].append(u["crop_calls"])
            sub["attempts"].append(u["submission_attempts"])
    return sub


def summarize(name: str, s: dict) -> str:
    n = max(1, s["n"])
    return (f"| {name} | {s['n']} | {s['ok']} ({s['ok']/n:.0%}) | {s['abstain']} | "
            f"{s['anchors']/n:.2f} | "
            f"{statistics.mean(s['visual_calls']) if s['visual_calls'] else 0:.2f} | "
            f"{statistics.mean(s['attempts']) if s['attempts'] else 0:.2f} | {s['missing']} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ids = {json.loads(l)["sample_id"] for l in
           pathlib.Path(args.manifest).read_text(encoding="utf-8").splitlines() if l.strip()}
    lines = ["| run | committed | ok | abstain | anchors/img | visual calls | attempts | missing |",
             "|---|---|---|---|---|---|---|---|"]
    for r in args.runs:
        lines.append(summarize(r, stats(r, ids)))
    text = "\n".join(lines)
    print(text)
    if args.out:
        pathlib.Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
