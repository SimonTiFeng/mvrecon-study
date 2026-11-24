"""KITTI per-frame ratio: is the estimator/GT-optimal offset stable within one drive?

Section 12 found s_lidar / s_GT = 0.9557 on KITTI against 0.951 on ETH3D, using two
unrelated metric references. This checks that the KITTI ratio holds frame by frame
rather than being an aggregate coincidence, and reports its spread.
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


def calib():
    c = {}
    for line in (ROOT / "calib_cam_to_cam.txt").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            c[k.strip()] = v.split()
    v2 = {}
    for line in (ROOT / "calib_velo_to_cam.txt").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            v2[k.strip()] = v.split()
    Rv = np.array([float(x) for x in v2["R"]], float).reshape(3, 3)
    Tv = np.array([float(x) for x in v2["T"]], float)
    R0 = np.array([float(x) for x in c["R_rect_00"]], float).reshape(3, 3)
    P = np.array([float(x) for x in c["P_rect_02"]], float).reshape(3, 4)
    return Rv, Tv, R0, P


def main():
    Rv, Tv, R0, P = calib()
    rows = [json.loads(l) for l in open("work/kitti2.jsonl")]
    sync = ROOT / "2011_09_26_drive_0002_sync"
    print(f"{'sid':<9}{'s_lidar':>9}{'s_gt':>8}{'ratio':>8}")
    rr = []
    for r in rows:
        sid, fr = r["sample_id"], r["frame"]
        npz = CACHE / f"{sid}__B.npz"
        if not npz.exists():
            continue
        d = np.load(npz)["depth"].astype(np.float64)
        sd = json.loads((CACHE / f"{sid}__B.json").read_text())
        Hp, Wp = sd["processed_shape"]
        K = np.asarray(r["K"], float)
        f = ((K[0, 0] * Wp / 1242) + (K[1, 1] * Hp / 375)) / 2 / 300.0
        dm = cv2.resize(d, (1242, 375)) * f
        gp = glob.glob(f"{GT_DIR}/2011_09_26_drive_0002_sync_groundtruth_depth_"
                       f"{fr}_image_02.png")
        if not gp:
            continue
        gt = cv2.imread(gp[0], cv2.IMREAD_UNCHANGED).astype(np.float64) / 256.0
        sub = dm[DV:DV + 352, DU:DU + 1216]
        m = (gt > 0) & np.isfinite(sub) & (sub > 0)
        p, q = sub[m], gt[m]
        grid = np.arange(0.85, 1.25, 0.002)
        vals = [ab(x * p, q) for x in grid]
        s_gt = float(grid[int(np.argmin(vals))])

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
        g2 = gt[iv[inb], iu[inb]]
        m2 = g2 > 0
        s_l = float(np.exp(np.median(np.log(z[inb][m2] / g2[m2]))))
        rr.append(s_l / s_gt)
        print(f"{sid:<9}{s_l:>9.4f}{s_gt:>8.4f}{s_l / s_gt:>8.4f}")
    rr = np.array(rr)
    print(f"\n  per-frame ratio median={np.median(rr):.4f} sd={rr.std():.4f} "
          f"range=[{rr.min():.4f},{rr.max():.4f}]")


if __name__ == "__main__":
    main()
