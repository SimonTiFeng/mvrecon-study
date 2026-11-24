"""Depth inference cache builder (frozen backends, no training).

Caches raw float depth per (sample, backend) with provenance. Resumable; skips
completed entries whose image hash and model revision match. Suitable for local
runs and for a single-card HPC job (same command, different manifest).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config          # noqa: E402
from src.backends.da3 import DA3Backend    # noqa: E402


def file_sha(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--backends", nargs="+", default=["A", "B"])
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    ap.add_argument("--out", default=str(ROOT / "cache" / "depths"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["backend_inference"]["device"] = args.device
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for m in args.manifest:
        rows += [json.loads(l) for l in pathlib.Path(m).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    devices = [DA3Backend(cfg["backends"][b], cfg["backend_inference"]).load()
               for b in args.backends]
    log = []
    for row in rows:
        img = pathlib.Path(row["dataset_root"]) / row["image"]
        h = file_sha(img)
        for be, letter in zip(devices, args.backends):
            dst = out / f"{row['sample_id']}__{letter}.npz"
            meta_path = out / f"{row['sample_id']}__{letter}.json"
            if dst.exists() and meta_path.exists():
                prev = json.loads(meta_path.read_text(encoding="utf-8"))
                if prev.get("image_sha256") == h and \
                        prev.get("model_revision") == be.spec["revision"]:
                    continue
            t0 = time.time()
            res = be.predict(str(img), (row["width"], row["height"]))
            cache_dtype = np.dtype(cfg["backend_inference"].get("cache_dtype", "float32"))
            include_conf = bool(cfg["backend_inference"].get("cache_include_conf", True))
            payload = {"depth": res["depth"].astype(cache_dtype)}
            if include_conf:
                payload["conf"] = (res["conf"] if res["conf"] is not None
                                   else np.zeros_like(res["depth"])).astype(cache_dtype)
            np.savez_compressed(dst, **payload)
            meta = {"sample_id": row["sample_id"], "backend": letter,
                    "image_sha256": h, "width": row["width"], "height": row["height"],
                    "processed_shape": res["processed_shape"],
                    "model_revision": res["model_revision"],
                    "checkpoint_sha256": res["checkpoint_sha256"],
                    "vendor_commit": res["vendor_commit"],
                    "process_res": res["process_res"],
                    "process_res_method": res["process_res_method"],
                    "is_metric_flag": res["is_metric"],
                    "depth_min": float(np.nanmin(res["depth"])),
                    "depth_max": float(np.nanmax(res["depth"])),
                    "cache_dtype": str(cache_dtype), "cache_include_conf": include_conf,
                    "precision_policy": res.get("precision_policy"),
                    "inference_seconds": round(time.time() - t0, 2)}
            (out / f"{row['sample_id']}__{letter}.json").write_text(
                json.dumps(meta, indent=1), encoding="utf-8")
            log.append(meta)
            print(f"[{row['sample_id']} {letter}] {meta['inference_seconds']}s "
                  f"min={meta['depth_min']:.3g} max={meta['depth_max']:.3g}", flush=True)
    (out / "inference_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"done: {len(log)} new entries in {out}")


if __name__ == "__main__":
    main()
