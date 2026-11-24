"""GT-free metric scale from cross-view camera-baseline agreement (ETH3D).

Idea: DA3's multi-view output includes predicted camera extrinsics. ETH3D ships
metric COLMAP poses. The model's reconstructed camera-centre separations scale
linearly with its depth scale error, so comparing predicted vs known-metric
baselines recovers the bias WITHOUT reading any ground-truth depth.

This is the only GT-free route that yields an ABSOLUTE (not merely relative)
scale, because COLMAP translations are in metres.

Outputs per scene:
  s_baseline = median over view-pairs of (COLMAP distance / predicted distance)
  and the same using the model's per-view depth scale directly (for cross-check).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import types

import numpy as np
import torch
from PIL import Image

VENDOR = pathlib.Path(__file__).resolve().parents[1] / "vendor" / "Depth-Anything-3" / "src"
sys.path.insert(0, str(VENDOR))
if "depth_anything_3.utils.export" not in sys.modules:
    shim = types.ModuleType("depth_anything_3.utils.export")
    shim.export = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("export disabled"))
    sys.modules["depth_anything_3.utils.export"] = shim

from crossview_scale import qvec_to_rot, read_colmap  # reused


def centers(exts: np.ndarray) -> np.ndarray:
    """Camera centres in world coords from world->cam 4x4s."""
    out = []
    for T in exts:
        R, t = T[:3, :3], T[:3, 3]
        out.append(-R.T @ t)
    return np.stack(out)


def scaled_centers(exts, s):
    """If depth is off by s, reconstruction is off by s -> centres scale by s."""
    return centers(exts) * s


def pair_baselines(c: np.ndarray) -> np.ndarray:
    n = len(c)
    return np.array([np.linalg.norm(c[i] - c[j])
                     for i in range(n) for j in range(i + 1, n)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-root", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--images", nargs="+", required=True,
                    help="image filenames (basenames) to feed as one multi-view set")
    ap.add_argument("--model", default="depth-anything/DA3METRIC-LARGE")
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--local-weights", default=None,
                    help="dir containing model.safetensors + config.json (offline)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    root = pathlib.Path(a.scene_root)
    calib = read_colmap(root / "dslr_calibration_jpg")

    imgs, ks, gt_centres = [], [], []
    for name in a.images:
        p = root / "images" / name
        if not p.exists():
            for cand in root.rglob(name):
                p = cand
                break
        imgs.append(Image.open(p).convert("RGB"))
        pose = calib.get(pathlib.Path(name).name)
        if pose is None:
            raise SystemExit(f"no COLMAP pose for {name}")
        R, t, K, W, H = pose
        ks.append(K)
        gt_centres.append(-R.T @ t)

    gt_c = np.stack(gt_centres)
    gt_bl = pair_baselines(gt_c)
    print(f"COLMAP metric baselines (m): min={gt_bl.min():.3f} med={np.median(gt_bl):.3f} "
          f"max={gt_bl.max():.3f}")

    from depth_anything_3.api import DepthAnything3
    local = pathlib.Path(a.local_weights) if a.local_weights else None
    if local and (local / "model.safetensors").exists():
        # Offline path: this cluster cannot reach huggingface.co. Build from the
        # vendored preset registry (no network) and load the checkpoint manually.
        from safetensors.torch import load_file
        m = DepthAnything3(model_name="da3metric-large")
        sd = load_file(str(local / "model.safetensors"))
        missing, unexpected = m.load_state_dict(sd, strict=False)
        print(f"offline weights: {len(sd)} tensors, "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        m = DepthAnything3.from_pretrained(a.model)
    m = m.to(torch.device(a.device)).eval()
    with torch.no_grad():
        pred = m.inference(imgs, process_res=a.process_res,
                           process_res_method="upper_bound_resize")
    exts = np.asarray(pred.extrinsics)
    print(f"predicted extrinsics: {exts.shape}")
    pc = centers(exts)
    pred_bl = pair_baselines(pc)

    # ratio of known-metric baseline to model-reconstructed baseline IS the scale factor
    # (model shrinks everything by 1/s if its depths are biased by s)
    ratios = gt_bl / np.maximum(pred_bl, 1e-9)
    print(f"per-pair COLMAP/predicted baseline ratio: "
          f"min={ratios.min():.3f} med={np.median(ratios):.3f} max={ratios.max():.3f}")
    rec = {"scene": a.scene, "images": a.images,
           "gt_baselines": gt_bl.tolist(), "pred_baselines": pred_bl.tolist(),
           "ratios": ratios.tolist(), "s_baseline": float(np.median(ratios)),
           "n_pairs": int(gt_bl.size)}
    pathlib.Path(a.out).write_text(json.dumps(rec, indent=1))
    print(f"\ns_baseline (GT-FREE) = {rec['s_baseline']:.4f}")


if __name__ == "__main__":
    main()
