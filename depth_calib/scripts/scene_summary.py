"""Per-scene aggregation of an eval per_image.csv."""
from __future__ import annotations

import argparse
import csv
import collections
import pathlib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(pathlib.Path(args.eval_dir) / "per_image.csv",
                                   encoding="utf-8")))
    systems = args.systems or sorted({r["system"] for r in rows})
    agg = collections.defaultdict(list)
    for r in rows:
        if r["system"] not in systems:
            continue
        try:
            v = float(r["abs_rel"])
        except (TypeError, ValueError):
            continue
        agg[(r["scene_id"], r["system"])].append(v)
    scenes = sorted({k[0] for k in agg})
    lines = ["| scene | " + " | ".join(systems) + " |",
             "|---|" + "---|" * len(systems)]
    for sc in scenes:
        cells = []
        for s in systems:
            vals = agg.get((sc, s), [])
            cells.append(f"{sum(vals)/len(vals):.4f}(n={len(vals)})" if vals else "")
        lines.append(f"| {sc} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        pathlib.Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
