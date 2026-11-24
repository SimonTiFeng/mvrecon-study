"""Renumber a processed dataset so `index` is unique dataset-wide.

`ingest_eth3d.py` numbers images per scene (0..N-1 for each scene), which collides when
scenes are combined. This rewrites the meta file and renames images/gt/mask to
`<dataset>_<global_index>` so sample ids stay unique and no file is overwritten.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="processed dataset dir with meta.json")
    ap.add_argument("--dataset", required=True, help="dataset name used in file prefixes")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    entries = meta["entries"]

    seen = set()
    collisions = [e["index"] for e in entries if e["index"] in seen or seen.add(e["index"])]
    print(f"entries={len(entries)} unique_index_before={len(set(e['index'] for e in entries))} "
          f"colliding_rows={len(collisions)}")

    # deterministic global order: scene order as first appearance, then local index
    scene_order: list[str] = []
    for e in entries:
        if e["scene_id"] not in scene_order:
            scene_order.append(e["scene_id"])
    ordered = sorted(entries, key=lambda e: (scene_order.index(e["scene_id"]), e["index"]))
    sample_ids = [f"{args.dataset}_{i:04d}" for i in range(len(ordered))]
    assert len(set(sample_ids)) == len(sample_ids)

    new_entries, renames = [], []
    for i, e in enumerate(ordered):
        new_base = sample_ids[i]
        for key, sub, ext in (("image", "images", ".png"), ("gt", "gt", ".npy"),
                              ("mask", "mask", ".npy")):
            old = root / e[key]
            new = root / sub / f"{new_base}{ext}"
            if old.resolve() != new.resolve():
                renames.append((old, new))
        r = dict(e)
        r["index"] = i
        r["sample_id"] = new_base
        r["image"] = f"images/{new_base}.png"
        r["gt"] = f"gt/{new_base}.npy"
        r["mask"] = f"mask/{new_base}.npy"
        new_entries.append(r)

    print(f"renames queued: {len(renames)}")
    if args.dry_run:
        for o, n in renames[:5]:
            print("  ", o.name, "->", n.name)
        return

    # two-phase rename to avoid clobbering when old and new names overlap
    tmp_map = []
    for old, new in renames:
        tmp = old.with_name(old.name + ".renumber_tmp")
        old.rename(tmp)
        tmp_map.append((tmp, new))
    for tmp, new in tmp_map:
        shutil.move(str(tmp), str(new))

    meta["entries"] = new_entries
    meta["n"] = len(new_entries)
    meta["renumbered"] = {
        "dataset": args.dataset,
        "rule": "scene-first-appearance then per-scene index, global 0..N-1",
        "unique_index_after": len({e["index"] for e in new_entries}),
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"renumbered {len(new_entries)} entries; unique ids now "
          f"{len({e['sample_id'] for e in new_entries})}")


if __name__ == "__main__":
    main()
