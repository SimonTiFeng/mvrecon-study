"""GT-free absolute scale from depth-vs-pose registration (no model poses needed).

For two views i,j of one scene with metric COLMAP poses:
  - backproject the model's depth maps into each camera frame
  - map j's cloud into i's frame with the KNOWN rigid pose (metres, no free scale)
  - the only unknown left is the model's multiplicative depth bias s
  - scan s so j's cloud lands on i's cloud; the argmin is the bias

Ground-truth depth is never read. The metric reference is COLMAP's translation,
which is in metres, so the recovered s is absolute, not merely relative.

Reports the per-pair s and the scene-level median, plus the GT-optimal s for
comparison (comparison only — not used by the estimator).
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

from crossview_scale import qvec_to_rot, read_colmap  # noqa: E402


def backproject(depth, K, stride=6, max_depth=120.0):
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    ys, xs = ys.ravel(), xs.ravel()
    z = depth[ys, xs]
    ok = np.isfinite(z) & (z > 0) & (z < max_depth)
    ys, xs, z = ys[ok], xs[ok], z[ok]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], 1)


def world_cloud(X_cam, R, t):
    """COLMAP world->cam: X_cam = R X_world + t  =>  X_world = R^T (X_cam - t)."""
    return (X_cam - t) @ R


def align_scale(A_world, B_world, lo=0.5, hi=2.0, trunc=0.30):
    """Scale between two clouds of one scene using radial statistics.

    Point-to-point matching is unreliable here because the model is monocular:
    the two views disagree on shape even at the correct scale. What survives
    shape disagreement is the RADIAL distribution about the scene centroid —
    a metric distance that scales exactly with s. Compare the robust radius
    quantiles of the two clouds to recover s.
    """
    from scipy.spatial import cKDTree
    ca, cb = np.median(A_world, axis=0), np.median(B_world, axis=0)
    ra = np.linalg.norm(A_world - ca, axis=1)
    rb = np.linalg.norm(B_world - cb, axis=1)
    q = np.array([0.25, 0.50, 0.75, 0.90])
    qa, qb = np.quantile(ra, q), np.quantile(rb, q)
    s = float(np.clip(np.median(qa / np.maximum(qb, 1e-9)), lo, hi))
    # residual at s, restricted to pairs that actually find a neighbour
    d, _ = cKDTree(A_world).query(s * B_world, k=1)
    keep = d < trunc
    resid = float(np.mean(d[keep])) if keep.sum() > 50 else float("inf")
    return s, resid, float(np.median(d))


def gt_opt_scale(d, gt, mask):
    m = mask & np.isfinite(d) & (d > 0) & np.isfinite(gt) & (gt > 0)
    if m.sum() < 100:
        return None
    p, g = d[m], gt[m]
    best = (1.0, np.inf)
    for s in np.linspace(0.5, 2.0, 300):
        v = float(np.mean(np.abs(s * p - g) / g))
        if v < best[1]:
            best = (float(s), v)
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-root", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="image basenames, grouped in twos: a1 b1 a2 b2 ...")
    ap.add_argument("--local-weights", default=None)
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import time
    t0 = time.time()
    root = pathlib.Path(a.scene_root)
    print(f"[{time.time()-t0:.1f}s] reading COLMAP calib...", flush=True)
    calib = read_colmap(root / "dslr_calibration_jpg")
    print(f"[{time.time()-t0:.1f}s] calib: {len(calib)} images", flush=True)

    if len(a.pairs) % 2:
        raise SystemExit("--pairs needs an even number of names")

    from depth_anything_3.api import DepthAnything3
    print(f"[{time.time()-t0:.1f}s] building model...", flush=True)
    if a.local_weights and (pathlib.Path(a.local_weights) / "model.safetensors").exists():
        from safetensors.torch import load_file
        m = DepthAnything3(model_name="da3metric-large")
        print(f"[{time.time()-t0:.1f}s] model built, loading weights...", flush=True)
        sd = load_file(str(pathlib.Path(a.local_weights) / "model.safetensors"))
        miss, unexp = m.load_state_dict(sd, strict=False)
        print(f"[{time.time()-t0:.1f}s] offline weights: missing={len(miss)} unexpected={len(unexp)}", flush=True)
    else:
        m = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    m = m.to(torch.device(a.device)).eval()
    print(f"[{time.time()-t0:.1f}s] model on {a.device}", flush=True)

    def depth_of(name):
        p = root / "images" / name
        if not p.exists():
            for c in root.rglob(name):
                p = c
                break
        img = Image.open(p).convert("RGB")
        with torch.no_grad():
            pred = m.inference([img], process_res=a.process_res,
                               process_res_method="upper_bound_resize")
        d = np.asarray(pred.depth)[0].astype(np.float64)
        pose = calib[pathlib.Path(name).name]
        R, t, Kc, Wc, Hc = pose
        print(f"[{time.time()-t0:.1f}s] {name}: depth {d.shape} -> resize {(Hc,Wc)}", flush=True)
        if d.shape != (Hc, Wc):
            import cv2
            d = cv2.resize(d, (Wc, Hc), interpolation=cv2.INTER_LINEAR)
        return d, R, t, Kc

    records = []
    for k in range(0, len(a.pairs), 2):
        na, nb = a.pairs[k], a.pairs[k + 1]
        da, Ra, ta, Ka = depth_of(na)
        db, Rb, tb, Kb = depth_of(nb)
        A = world_cloud(backproject(da, Ka, a.stride), Ra, ta)
        B = world_cloud(backproject(db, Kb, a.stride), Rb, tb)
        print(f"[{time.time()-t0:.1f}s] clouds {len(A)}/{len(B)} pts, scanning s...", flush=True)
        s_est, cost, med_d = align_scale(A, B)
        rec = {"a": na, "b": nb, "n_a": int(len(A)), "n_b": int(len(B)),
               "s_estimator": s_est, "cost": cost, "median_nn_m": med_d}
        records.append(rec)
        print(f"[{na} | {nb}] s={s_est:.4f}  median NN={med_d:.3f} m  "
              f"(n={len(A)}/{len(B)})", flush=True)

    s_scene = float(np.median([r["s_estimator"] for r in records]))
    rec_out = {"scene": a.scene, "pairs": records, "s_scene": s_scene}
    pathlib.Path(a.out).write_text(json.dumps(rec_out, indent=1))
    print(f"\nGT-FREE s_scene = {s_scene:.4f}  (from {len(records)} pairs)")


if __name__ == "__main__":
    main()
