"""KITTI scale estimator: model depth vs Velodyne LiDAR, both in metres.

KITTI ships metric LiDAR (velodyne_points) with a calibrated velo->cam transform.
Project the LiDAR cloud into the rectified camera, read its depth along the optical
axis at a pixel, and compare with the model's predicted depth there. The ratio is
the model's scale, measured against LiDAR — no ground-truth depth involved.

This is the KITTI analogue of scripts/landmark_scale.py, with one important
difference: KITTI's metric reference is raw LiDAR in metres, not SfM up to scale,
so no benchmark-level SfM/GT constant is expected here.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np


def _load_calib(calib_dir: pathlib.Path):
    def read_pairs(path):
        out = {}
        for l in path.read_text().splitlines():
            if ":" not in l:
                continue
            k, v = l.split(":", 1)
            out[k.strip()] = v.split()
        return out

    c2c = read_pairs(calib_dir / "calib_cam_to_cam.txt")
    v2c = read_pairs(calib_dir / "calib_velo_to_cam.txt")

    P = np.array(list(map(float, c2c["P_rect_02"])), float).reshape(3, 4)
    R0 = np.array(list(map(float, c2c["R_rect_00"])), float).reshape(3, 3)
    Rv = np.array(list(map(float, v2c["R"])), float).reshape(3, 3)
    Tv = np.array(list(map(float, v2c["T"])), float).reshape(3)
    return P, R0, Rv, Tv


def velo_to_cam_points(velo_xyz: np.ndarray, R0, Rv, Tv) -> np.ndarray:
    """Velodyne (x,y,z,refl) -> rectified camera 0 coordinates."""
    X = velo_xyz[:, :3]
    # velodyne -> camera 0 (unrectified)
    Xc = X @ Rv.T + Tv
    # apply rectifying rotation
    Xr = Xc @ R0.T
    return Xr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--drive", required=True)
    ap.add_argument("--root", default="/d/kitti")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--map-json", required=True, help='{"0000000000.png": "kitti_0000"}')
    ap.add_argument("--min-points", type=int, default=200)
    ap.add_argument("--max-log-sd", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    root = pathlib.Path(a.root) / a.date
    sync = root / f"{a.date}_drive_{a.drive}_sync"
    calib_dir = root
    P, R0, Rv, Tv = _load_calib(calib_dir)
    fx, fy, cx, cy = P[0, 0], P[1, 1], P[0, 2], P[1, 2]
    cache = pathlib.Path(a.cache)
    name2sid = json.loads(pathlib.Path(a.map_json).read_text())

    imgs = sorted((sync / "image_02" / "data").glob("*.png"))
    vels = sorted((sync / "velodyne_points" / "data").glob("*.bin"))
    print(f"{len(imgs)} images, {len(vels)} velodyne scans; intrinsics fx={fx:.1f}")

    recs = []
    for img_path, vel_path in zip(imgs, vels):
        sid = name2sid.get(img_path.name)
        if sid is None:
            continue
        npz = cache / f"{sid}__B.npz"
        side = cache / f"{sid}__B.json"
        if not npz.exists():
            continue
        d = np.load(npz)["depth"].astype(np.float64)
        sd = json.loads(side.read_text())
        Hp, Wp = sd["processed_shape"]
        # kitti image_02 is 1242x375
        H, W = 375, 1242
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)

        velo = np.fromfile(vel_path, dtype=np.float32).reshape(-1, 4)
        Xr = velo_to_cam_points(velo, R0, Rv, Tv)
        z = Xr[:, 2]
        ok = z > 1.0
        Xr = Xr[ok]

        rows = []
        for X in Xr:
            zz = X[2]
            u = fx * X[0] / zz + cx
            v = fy * X[1] / zz + cy
            iu, iv = int(round(u)), int(round(v))
            if not (2 <= iu < W - 2 and 2 <= iv < H - 2):
                continue
            patch = d[iv-1:iv+2, iu-1:iu+2]
            zm = float(np.median(patch))
            if not np.isfinite(zm) or zm <= 0:
                continue
            rows.append((float(zz), zm))
        if len(rows) < a.min_points:
            print(f"  {img_path.name} ({sid}): {len(rows)} points, skip")
            continue
        zt = np.array([r[0] for r in rows])
        zm = np.array([r[1] for r in rows])
        lr = np.log(zt / zm)
        med = np.median(lr)
        lr = lr[np.abs(lr - med) < 3 * 1.4826 * np.median(np.abs(lr - med)) + 1e-9]
        s = float(np.exp(np.median(lr)))
        sd_l = float(np.std(lr))
        good = sd_l <= a.max_log_sd
        recs.append({"image": img_path.name, "sid": sid, "s": s,
                     "n_points": int(len(rows)), "log_sd": sd_l, "ok": bool(good)})
        print(f"  {img_path.name} ({sid}): s={s:.4f} from {len(rows)} pts "
              f"(log sd {sd_l:.3f}){'' if good else '  [REJECTED]'}")

    good = [r for r in recs if r["ok"]]
    if good:
        arr = np.array([r["s"] for r in good])
        print(f"\ndrive median s = {np.median(arr):.4f} sd={arr.std():.4f} "
              f"(accepted {len(good)}/{len(recs)})")
    pathlib.Path(a.out).write_text(json.dumps(recs, indent=1))


if __name__ == "__main__":
    main()
