"""Official-path DA3 metric evaluation: pass dataset intrinsics, use model's own scaling.

Reproduces the paper's protocol (DepthAnything3.inference with intrinsics supplied and
align_to_input_ext_scale=True), then evaluates AbsRel on the valid GT mask.

Usage:
  python scripts/official_eval.py --manifest manifests/pilot.jsonl --split nyuv2 \
      --model depth-anything/DA3METRIC-LARGE --data-root /path --out results/official_nyu.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import types

import cv2
import numpy as np
import torch
from PIL import Image

VENDOR = pathlib.Path(__file__).resolve().parents[1] / "vendor" / "Depth-Anything-3" / "src"
sys.path.insert(0, str(VENDOR))
if "depth_anything_3.utils.export" not in sys.modules:
    shim = types.ModuleType("depth_anything_3.utils.export")
    shim.export = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("export disabled"))
    sys.modules["depth_anything_3.utils.export"] = shim


def absrel(z, g):
    return float(np.mean(np.abs(z - g) / g))


def load_model(repo: str, device: str):
    from depth_anything_3.api import DepthAnything3
    m = DepthAnything3.from_pretrained(repo)
    return m.to(torch.device(device)).eval()


@torch.no_grad()
def infer(model, img: Image.Image, K: np.ndarray, process_res: int):
    """Run the official single-view path with dataset intrinsics supplied."""
    Kb = K[None]  # (1,3,3)
    pred = model.inference([img], intrinsics=Kb,
                           align_to_input_ext_scale=True,
                           process_res=process_res,
                           process_res_method="upper_bound_resize")
    d = np.asarray(pred.depth)[0].astype(np.float64)
    return d, np.asarray(pred.intrinsics)[0] if pred.intrinsics is not None else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--data-root", default=None, help="override dataset_root")
    ap.add_argument("--model", default="depth-anything/DA3METRIC-LARGE")
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in
            pathlib.Path(a.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.split:
        rows = [r for r in rows if r["dataset"] == a.split]
    if a.limit:
        rows = rows[:a.limit]

    model = load_model(a.model, a.device)
    out = []
    for i, row in enumerate(rows):
        root = pathlib.Path(a.data_root or row["dataset_root"])
        img = Image.open(root / row["image"]).convert("RGB")
        K = np.asarray(row["K"], float)
        d, Kp = infer(model, img, K, a.process_res)
        W, H = row["width"], row["height"]
        if d.shape != (H, W):
            d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
        gt = np.load(root / row["gt"]).astype(np.float64)
        mask = np.load(root / row["mask"]).astype(bool)
        if d.shape != gt.shape:
            print(f"[skip {row['sample_id']}] shape {d.shape} vs {gt.shape}", flush=True)
            continue
        m = mask & np.isfinite(d) & (d > 0) & np.isfinite(gt) & (gt > 0)
        if m.sum() < 100:
            continue
        p, g = d[m], gt[m]
        # GT-optimal single scale (oracle diagnostic, not a method)
        best = (1.0, 1e9)
        for s in np.exp(np.linspace(np.log(0.5), np.log(2.0), 200)):
            v = absrel(s * p, g)
            if v < best[1]:
                best = (float(s), float(v))
        out.append({"sample_id": row["sample_id"], "dataset": row["dataset"],
                    "n_pix": int(g.size),
                    "absrel_raw": absrel(p, g),
                    "s_opt": best[0], "absrel_oracle_scale": best[1],
                    "pred_intrinsics_fx": float(Kp[0, 0]) if Kp is not None else None,
                    "input_fx": float(K[0, 0])})
        print(f"[{i+1}/{len(rows)}] {row['sample_id']} raw={out[-1]['absrel_raw']:.4f} "
              f"s*={best[0]:.4f}", flush=True)

    pathlib.Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    r = np.array([o["absrel_raw"] for o in out])
    o = np.array([o["absrel_oracle_scale"] for o in out])
    s = np.array([o["s_opt"] for o in out])
    print(f"\n=== n={len(out)} raw={np.median(r):.4f} oracle-scale={np.median(o):.4f} "
          f"median s*={np.median(s):.4f}")


if __name__ == "__main__":
    main()
