"""Materialize a manifest split into a run dir: anonymized image + bound context."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.manifest import materialize_sample  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--mode", default="harness", choices=["harness", "direct"])
    ap.add_argument("--toolset", default=None,
                    choices=["all", "points", "direct_coords", "none"])
    ap.add_argument("--targets", default=None,
                    help="JSON file mapping sample_id -> fixed measurement target text")
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    args = ap.parse_args()

    run_dir = ROOT / "results" / args.run_id / "samples"
    rows = [json.loads(l) for l in pathlib.Path(args.manifest).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    targets = {}
    if args.targets:
        targets = json.loads(pathlib.Path(args.targets).read_text(encoding="utf-8"))
    index = []
    for row in rows:
        sid = row["sample_id"]
        sdir = run_dir / sid
        sample = materialize_sample(row, sdir)
        ctx = {"image_id": sid, "image_path": sample["image_path"],
               "width": sample["width"], "height": sample["height"],
               "output_dir": (sdir / "mcp_out").as_posix(),
               "run_id": args.run_id, "mode": args.mode}
        if args.toolset:
            ctx["toolset"] = args.toolset
            ctx["require_point_ids"] = (args.toolset == "points")
        if sid in targets:
            ctx["target"] = targets[sid]
        (sdir / "sample_context.json").write_text(json.dumps(ctx, indent=1),
                                                  encoding="utf-8")
        index.append({**sample, "context": (sdir / "sample_context.json").as_posix()})
    (ROOT / "results" / args.run_id / "samples_index.json").write_text(
        json.dumps(index, indent=1), encoding="utf-8")
    print(f"materialized {len(index)} samples -> {run_dir}")


if __name__ == "__main__":
    main()
