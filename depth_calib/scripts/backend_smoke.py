"""Backend engineering smoke: load a frozen DA3 checkpoint and infer one image.

Not a scientific result. Records model revision, checkpoint hash, device, timing.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config            # noqa: E402
from src.backends.da3 import DA3Backend      # noqa: E402


def synthetic_image(path: pathlib.Path) -> None:
    """Simple perspective scene: floor grid and two vertical posts."""
    W, H = 640, 480
    img = Image.new("RGB", (W, H), (200, 210, 220))
    d = ImageDraw.Draw(img)
    for i in range(0, W, 40):
        d.line([(W // 2, 240), (i, H)], fill=(140, 140, 140), width=1)
    for y in (300, 340, 390, 450):
        d.line([(0, y), (W, y)], fill=(140, 140, 140), width=1)
    d.rectangle([420, 180, 452, 400], fill=(120, 70, 40))
    d.rectangle([500, 210, 524, 386], fill=(120, 70, 40))
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="A")
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results" / "backend_smoke"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    img_path = out / "synthetic.png"
    if not img_path.exists():
        synthetic_image(img_path)

    spec = cfg["backends"][args.backend]
    be = DA3Backend(spec, cfg["backend_inference"]).load()
    t0 = time.time()
    res = be.predict(str(img_path), (640, 480))
    dt = time.time() - t0

    d = res["depth"]
    rec = {
        "backend": args.backend,
        "model": spec["name"],
        "hf_repo": spec["hf_repo"],
        "revision": spec["revision"],
        "checkpoint_sha256": res["checkpoint_sha256"],
        "vendor_commit": res["vendor_commit"],
        "device": str(be.device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "process_res": res["process_res"],
        "processed_shape": res["processed_shape"],
        "inference_seconds": round(dt, 2),
        "depth_min": float(np.nanmin(d)),
        "depth_max": float(np.nanmax(d)),
        "depth_median": float(np.nanmedian(d)),
        "is_metric_flag": res["is_metric"],
    }
    np.savez_compressed(out / f"depth_{args.backend}.npz", depth=d,
                        conf=res["conf"] if res["conf"] is not None else np.zeros_like(d))
    (out / f"smoke_{args.backend}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
