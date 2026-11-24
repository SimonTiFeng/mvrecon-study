"""NYUv2 labeled ingest (MATLAB v7.3 / HDF5): PNG + GT depth + mask + scene ids.

Storage layout verified from the official file:
  images (1449, 3, 640, 480) uint8     -> per image transpose(2,1,0) = (480,640,3)
  depths (1449, 640, 480) float32      -> transpose to (480,640), metres (z-depth)
  scenes (1, 1449) HDF5 refs           -> true scene names (464 unique)
"""
from __future__ import annotations

import argparse
import json
import pathlib

import h5py
import numpy as np
from PIL import Image

K = {"fx": 518.8579, "fy": 519.469, "cx": 325.5824, "cy": 253.7362}


def _decode_ref(f, ref) -> str:
    v = f[ref][()]
    if isinstance(v, np.ndarray):
        return "".join(chr(int(c)) for c in v.ravel())
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    for sub in ("images", "gt", "mask"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    entries = []
    with h5py.File(args.mat, "r") as f:
        n = f["images"].shape[0] if args.limit <= 0 else min(args.limit, f["images"].shape[0])
        refs = f["scenes"][0]
        for i in range(n):
            img = np.transpose(np.array(f["images"][i]), (2, 1, 0))
            dep = np.transpose(np.array(f["depths"][i]), (1, 0)).astype(np.float32)
            if img.shape[:2] != (480, 640) or dep.shape != (480, 640):
                raise ValueError(f"unexpected shapes img={img.shape} dep={dep.shape}")
            mask = np.isfinite(dep) & (dep > 0)
            Image.fromarray(img.astype(np.uint8)).save(out / "images" / f"{i:04d}.png")
            np.save(out / "gt" / f"{i:04d}.npy", dep)
            np.save(out / "mask" / f"{i:04d}.npy", mask)
            entries.append({
                "index": i, "dataset": "nyuv2",
                "scene_id": _decode_ref(f, refs[i]),
                "image": f"images/{i:04d}.png", "gt": f"gt/{i:04d}.npy",
                "mask": f"mask/{i:04d}.npy", "width": 640, "height": 480,
                "K": [[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]],
                "depth_convention": "z_depth_m", "indoor_outdoor": "indoor"})
    meta = {
        "dataset": "nyuv2", "source": args.mat, "n": n,
        "K_note": "official NYUv2 labeled intrinsics constants; per-image K not distributed",
        "scene_note": "scene ids from the 'scenes' variable (464 unique scenes)",
        "depth_note": "GT = 'depths' variable, metres, mask = finite & >0",
        "entries": entries,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote {n} entries to {out}")


if __name__ == "__main__":
    main()
