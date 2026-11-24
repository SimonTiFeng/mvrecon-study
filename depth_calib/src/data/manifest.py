"""Deterministic manifest construction with scene-disjoint dev/pilot/smoke splits.

Sampling rule (frozen in configs/pilot.yaml): scenes are ordered by
sha256(seed|dataset|scene_id); images inside a scene keep dataset order and at
most `max_images_per_scene` are taken. pilot/dev/smoke consume disjoint scenes.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

from PIL import Image, ImageOps


def _scene_key(seed: int, dataset: str, scene_id: str) -> str:
    return hashlib.sha256(f"{seed}|{dataset}|{scene_id}".encode()).hexdigest()


def load_dataset_meta(root: pathlib.Path) -> dict:
    return json.loads((root / "meta.json").read_text(encoding="utf-8"))


def _entries_by_scene(meta: dict) -> dict[str, list[dict]]:
    scenes: dict[str, list[dict]] = {}
    for e in meta["entries"]:
        scenes.setdefault(e["scene_id"], []).append(e)
    for v in scenes.values():
        v.sort(key=lambda x: x["index"])
    return scenes


def assign_splits(datasets: dict[str, pathlib.Path], quotas: dict, seed: int,
                  max_per_scene: int) -> dict[str, list[dict]]:
    """Return {'pilot': [...], 'dev': [...], 'smoke': [...]} of dataset entries."""
    splits: dict[str, list[dict]] = {"pilot": [], "dev": [], "smoke": []}
    need = {
        "pilot": quotas["pilot_quotas"],
        "dev": quotas["dev_quotas"],
        "smoke": quotas["smoke_quotas"],
    }
    for ds, root in datasets.items():
        meta = load_dataset_meta(root)
        scenes = _entries_by_scene(meta)
        order = sorted(scenes, key=lambda s: _scene_key(seed, ds, s))
        cursor = 0
        for split in ("pilot", "dev", "smoke"):
            want = need[split].get(ds, 0)
            if want == 0:
                continue
            taken = 0
            while taken < want and cursor < len(order):
                sid = order[cursor]
                cursor += 1
                imgs = scenes[sid][:max_per_scene]
                for e in imgs:
                    row = dict(e)
                    row["split"] = split
                    row["dataset_root"] = str(root)
                    row["sample_id"] = f"{ds}_{e['index']:04d}"
                    splits[split].append(row)
                    taken += 1
                    if taken >= want:
                        break
            if taken < want:
                raise RuntimeError(
                    f"{ds}: only {taken}/{want} {split} images available in disjoint scenes")
    return splits


def write_manifests(splits: dict[str, list[dict]], out_dir: pathlib.Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for split, rows in splits.items():
        p = out_dir / f"{split}.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        scenes = sorted({r["scene_id"] for r in rows})
        info[split] = {"n": len(rows), "scenes": scenes,
                       "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                       "by_dataset": {ds: sum(1 for r in rows if r["dataset"] == ds)
                                      for ds in {r["dataset"] for r in rows}}}
    (out_dir / "manifest_index.json").write_text(json.dumps(info, indent=1),
                                                 encoding="utf-8")
    return info


def materialize_sample(row: dict, sample_dir: pathlib.Path) -> dict:
    """Copy + re-encode (EXIF stripped, orientation applied) for the agent side."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    src = pathlib.Path(row["dataset_root"]) / row["image"]
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    dst = sample_dir / "input.png"
    img.save(dst, format="PNG")
    h = hashlib.sha256(dst.read_bytes()).hexdigest()
    return {"sample_id": row["sample_id"], "image_path": str(dst),
            "width": img.width, "height": img.height, "image_sha256": h}


def sample_id_for(row: dict) -> str:
    return f"{row['dataset']}_{row['index']:04d}"
