"""Merge ETH3D processed dirs into one dataset with globally unique indices.

Ingest numbers images per scene; manifests need unique sample ids dataset-wide,
so entries are renumbered and files renamed deterministically here.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    for sub in ("images", "gt", "mask"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    entries = []
    seen_scenes: dict[str, pathlib.Path] = {}
    skipped = {}
    for d in map(pathlib.Path, args.inputs):
        mf = d / "meta.json"
        if not mf.exists():
            continue
        meta = json.loads(mf.read_text(encoding="utf-8"))
        skipped.update(meta.get("skipped", {}))
        for e in meta["entries"]:
            scene = e["scene_id"]
            if scene in seen_scenes and seen_scenes[scene] != d:
                continue                      # first source wins per scene
            seen_scenes.setdefault(scene, d)
            entries.append((d, e))

    meta_out = []
    for new_i, (d, e) in enumerate(entries):
        old_base = pathlib.Path(e["image"]).stem            # e.g. courtyard_0003
        new_base = f"{e['scene_id']}_{new_i:04d}"
        for kind, ext in (("images", ".png"), ("gt", ".npy"), ("mask", ".npy")):
            src = d / kind / f"{old_base}{ext}"
            dst = out / kind / f"{new_base}{ext}"
            shutil.copy2(src, dst)
        rec = dict(e)
        rec["index"] = new_i
        rec["image"] = f"images/{new_base}.png"
        rec["gt"] = f"gt/{new_base}.npy"
        rec["mask"] = f"mask/{new_base}.npy"
        meta_out.append(rec)

    (out / "meta.json").write_text(json.dumps({
        "dataset": "eth3d", "n": len(meta_out), "scenes": sorted(seen_scenes),
        "skipped": skipped,
        "depth_note": "headerless float32 metres; invalid=+inf",
        "resize_note": "long side <= 1280; K scaled; depth/mask nearest",
        "entries": meta_out}, indent=1), encoding="utf-8")
    print(f"merged {len(meta_out)} entries from {len(seen_scenes)} scenes -> {out}")


if __name__ == "__main__":
    main()
