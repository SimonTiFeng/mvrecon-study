"""KITTI drive_0002 evaluation with the corrected GT crop offset (du=7, dv=23).

Recomputes: per-frame landmark scale from Velodyne, the GT-optimal scale, and the
AbsRel each delivers. Uses the offset found by scripts/kitti_crop_offset.py, which
maximises LiDAR/GT agreement (84.7% of points within 2%).
"""
from __future__ import annotations

import glob
import json
import pathlib

import cv2
import numpy as np

CACHE = pathlib.Path("D:/kitti2_cache")
ROOT = pathlib.Path("D:/kitti/2011_09_26")
GT_DIR = "D:/kitti/sel/depth_selection/val_selection_cropped/groundtruth_depth"
DU, DV = 7, 23


def ab(z, g):
    return float(np.mean(np.abs(z - g) / g))


def read_calib():
    calib = {}
    for line in (ROOT / "calib_cam_to_cam.txt").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            calib[k.strip()] = v.split()
    v2c = {}
    for line in (ROOT / "calib_velo_to_cam.txt").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            v2c[k.strip()] = v.split()
    Rv = np.array([float(x) for x in v2c["R"]], float).reshape(3, 3)
    Tv = np.array([float(x) for x in v2c["T"]], float)
    R0 = np.array([float(x) for x in calib["R_rect_00"]], float).reshape(3, 3)
    P = np.array([float(x) for x in calib["P_rect_02"]], float).reshape(3, 4)
    return Rv, Tv, R0, P


def main():
    Rv, Tv, R0, P = read_calib()
    rows = [json.loads(l) for l in open("work/kitti2.jsonl")]
    sync = ROOT / "2011_09_26_drive_0002_sync"

    print(f"{'sid':<9}{'n_pts':>7}{'s_lidar':>9}{'raw':>9}{'@s_l':>8}{'@s_gt':>8}{'px':>8}")
    recs = []
    for r in rows:
        sid, fr = r["sample_id"], r["frame"]
        npz = CACHE / f"{sid}__B.npz"
        side = CACHE / f"{sid}__B.json"
        if not (npz.exists() and side.exists()):
            continue
        d = np.load(npz)["depth"].astype(np.float64)
        sd = json.loads(side.read_text())
        Hp, Wp = sd["processed_shape"]
        K = np.asarray(r["K"], float)
        f = ((K[0, 0] * Wp / r["width"]) + (K[1, 1] * Hp / r["height"])) / 2 / 300.0
        dm = cv2.resize(d, (1242, 375)) * f

        gpath = glob.glob(f"{GT_DIR}/2011_09_26_drive_0002_sync_groundtruth_depth_"
                          f"{fr}_image_02.png")
        if not gpath:
            continue
        gt = cv2.imread(gpath[0], cv2.IMREAD_UNCHANGED).astype(np.float64) / 256.0

        vel = np.fromfile(str(sync / "velodyne_points" / "data" / f"{fr}.bin"),
                          dtype=np.float32).reshape(-1, 4)
        X = (vel[:, :3] @ Rv.T + Tv) @ R0.T
        z = X[:, 2]
        X = X[z > 1]
        z = z[z > 1]
        u = P[0, 0] * X[:, 0] / z + P[0, 2]
        v = P[1, 1] * X[:, 1] / z + P[1, 2]
        iu = np.round(u).astype(int) - DU
        iv = np.round(v).astype(int) - DV
        inb = (iu >= 0) & (iu < 1216) & (iv >= 0) & (iv < 352)
        if inb.sum() < 1000:
            continue
        g2 = gt[iv[inb], iu[inb]]
        L = z[inb]
        m = g2 > 0
        if m.sum() < 500:
            continue
        lr = np.log(L[m] / g2[m])
        med = np.median(lr)
        lr = lr[np.abs(lr - med) < 3 * 1.4826 * np.median(np.abs(lr - med)) + 1e-9]
        s_lidar = float(np.exp(np.median(lr)))

        # AbsRel on the GT's own valid pixels (full crop), model vs GT
        gv = gt > 0
        sub = dm[DV:DV + 352, DU:DU + 1216]
        mg = gv & np.isfinite(sub) & (sub > 0)
        p, q = sub[mg], gt[mg]
        grid = np.arange(0.80, 1.25, 0.002)
        vals = [ab(s * p, q) for s in grid]
        s_gt = float(grid[int(np.argmin(vals))])
        a0 = ab(p, q)
        a1 = ab(s_lidar * p, q)
        a2 = float(np.min(vals))
        recs.append((sid, len(L[m]), s_lidar, a0, a1, a2, int(mg.sum()), s_gt))
        print(f"{sid:<9}{len(L[m]):>7}{s_lidar:>9.4f}{a0:>9.4f}{a1:>8.4f}{a2:>8.4f}"
              f"{int(mg.sum()):>8}")

    if not recs:
        print("no frames evaluated")
        return
    arr = np.array([r[2:7] for r in recs], float)
    print(f"\n  n={len(recs)}  s_lidar={np.median(arr[:, 0]):.4f}  raw={np.median(arr[:, 1]):.4f}"
          f"  @s_lidar={np.median(arr[:, 2]):.4f}  @s_gt={np.median(arr[:, 3]):.4f}")
    print(f"  wins @s_lidar {int((arr[:, 2] < arr[:, 1]).sum())}/{len(recs)}")
    print(f"  GT-optimal s median = {np.median([r[7] for r in recs]):.4f}")
    print(f"  ratio s_lidar / s_gt = "
          f"{np.median(arr[:, 0]) / np.median([r[7] for r in recs]):.4f}")
    print("  published DA3-metric KITTI = 0.086")


if __name__ == "__main__":
    main()
