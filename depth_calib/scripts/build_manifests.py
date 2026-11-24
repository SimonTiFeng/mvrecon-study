"""Build scene-disjoint pilot/dev/smoke manifests from processed dataset roots."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config                     # noqa: E402
from src.data.manifest import assign_splits, write_manifests  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="name=processed_root, e.g. nyuv2=E:/.../nyuv2")
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    ap.add_argument("--out", default=str(ROOT / "manifests"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    datasets = {}
    for item in args.datasets:
        name, path = item.split("=", 1)
        datasets[name] = pathlib.Path(path)
    splits = assign_splits(datasets, cfg["data"], cfg["run"]["seed"],
                           cfg["data"]["max_images_per_scene"])
    info = write_manifests(splits, pathlib.Path(args.out))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
