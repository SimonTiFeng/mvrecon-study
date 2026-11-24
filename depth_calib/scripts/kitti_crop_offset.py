"""Locate the exact crop offset of KITTI depth-selection GT within the full frame.

The GT images in `data_depth_selection/val_selection_cropped` are 352x1216 while the
raw frame is 375x1242, so a crop offset exists. Rather than guess it, score every
candidate offset by how well projected Velodyne depths agree with the GT.
"""
from __future__ import annotations

import glob
import pathlib

import cv2
import numpy as np

ROOT = pathlib.Path("D:/kitti/2011_09_26")
GT_DIR = "D:/kitti/sel/depth_selection/val_selection_cropped/groundtruth_depth"


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
    gt_path = sorted(glob.glob(f"{GT_DIR}/*0005_image_02*"))[0]
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float64) / 256.0
    vel = np.fromfile(str(ROOT / "2011_09_26_drive_0002_sync" / "velodyne_points" / "data"
                       / "0000000005.bin"), dtype=np.float32).reshape(-1, 4)
    X = (vel[:, :3] @ Rv.T + Tv) @ R0.T
    z = X[:, 2]
    X = X[z > 1]
    z = z[z > 1]
    u = P[0, 0] * X[:, 0] / z + P[0, 2]
    v = P[1, 1] * X[:, 1] / z + P[1, 2]

    print("GT shape", gt.shape, " full frame 375x1242")
    print(f"{'du':>4}{'dv':>5}{'n':>7}{'medratio':>11}{'agree2%':>10}")
    best = (0, 0, -1.0, 0.0)
    for du in range(-4, 8):
        for dv in range(16, 30):
            iu = np.round(u).astype(int) - du
            iv = np.round(v).astype(int) - dv
            inb = (iu >= 0) & (iu < 1216) & (iv >= 0) & (iv < 352)
            if inb.sum() < 3000:
                continue
            g2 = gt[iv[inb], iu[inb]]
            L = z[inb]
            m = g2 > 0
            if m.sum() < 1500:
                continue
            rel = np.abs(L[m] - g2[m]) / g2[m]
            agree = float((rel < 0.02).mean())
            mr = float(np.exp(np.median(np.log(L[m] / g2[m]))))
            if agree > best[2]:
                best = (du, dv, agree, mr)
            mark = " *" if agree >= best[2] - 1e-9 else ""
            print(f"{du:>4}{dv:>5}{int(m.sum()):>7}{mr:>11.3f}{agree * 100:>9.1f}%{mark}")
    print(f"\nBEST du={best[0]} dv={best[1]} agree={best[2] * 100:.1f}% ratio={best[3]:.3f}")


if __name__ == "__main__":
    main()
