"""Build the FORMAL manifest: every image of every downloaded scene, no caps, no splitting.

Per config `data.formal_scope`, this emits one manifest covering the full processed
datasets (native resolution, no per-scene cap). Scene-disjoint dev/pilot/smoke splits
remain in the exploratory manifests and are not used for formal results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="name=processed_root")
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    ap.add_argument("--out", default=str(ROOT / "manifests" / "formal_all.jsonl"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    scope = cfg["data"]["formal_scope"]
    rows = []
    for item in args.datasets:
        name, path = item.split("=", 1)
        root = pathlib.Path(path)
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        keep = 0
        for e in meta["entries"]:
            r = dict(e)
            r["dataset_root"] = str(root)
            r["split"] = "formal"
            r["sample_id"] = f"{name}_{e['index']:04d}"
            if scope["resize_max_side"] and max(r["width"], r["height"]) > scope["resize_max_side"]:
                raise RuntimeError(
                    f"{r['sample_id']} is {r['width']}x{r['height']} but formal_scope forbids "
                    f"resizing; re-ingest at native resolution first")
            rows.append(r)
            keep += 1
        print(f"[{name}] {keep} images from {len({e['scene_id'] for e in meta['entries']})} scenes")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    info = {
        "n": len(rows),
        "by_dataset": {d: sum(1 for r in rows if r["dataset"] == d)
                       for d in sorted({r["dataset"] for r in rows})},
        "scenes": len({(r["dataset"], r["scene_id"]) for r in rows}),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "scope": scope,
        "note": "formal manifest: every processed image, native resolution, no per-scene cap",
    }
    (out.parent / "formal_index.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
