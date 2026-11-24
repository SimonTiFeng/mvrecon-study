"""KITTI full evaluation across all depth-selection drives with image_02 GT.

For each frame: build the manifest entry, run the LiDAR scale estimator with the
verified GT crop offset (du=7, dv=23), and compare raw vs estimator-calibrated AbsRel.
Requires depth caches produced by scripts/infer_depths.py and the raw drives on disk.
"""
from __future__ import annotations

import glob
import json
import pathlib
import re

import cv2
import numpy as np

KITTI = pathlib.Path("D:/kitti")
GT_DIR = "D:/kitti/sel/depth_selection/val_selection_cropped/groundtruth_depth"
CACHE = pathlib.Path("D:/kitti_cache_full")
DU, DV = 7, 23
W, H = 1242, 375


def ab(z, g):
    return float(np.mean(np.abs(z - g) / g))


def read_calib(date_dir: pathlib.Path):
    c = {}
    for line in (date_dir / "calib_cam_to_cam.txt").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            c[k.strip()] = v.split()
    v2 = {}
    for line in (date_dir / "calib_velo_to_cam.txt").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            v2[k.strip()] = v.split()
    Rv = np.array([float(x) for x in v2["R"]], float).reshape(3, 3)
    Tv = np.array([float(x) for x in v2["T"]], float)
    R0 = np.array([float(x) for x in c["R_rect_00"]], float).reshape(3, 3)
    P = np.array([float(x) for x in c["P_rect_02"]], float).reshape(3, 4)
    return Rv, Tv, R0, P


def build_manifest(out_jsonl: pathlib.Path) -> int:
    gts = sorted(glob.glob(f"{GT_DIR}/*_image_02.png"))
    rows = []
    for g in gts:
        name = pathlib.Path(g).name
        m = re.match(r"(\d{4}_\d{2}_\d{2})_drive_([0-9]+)_sync_groundtruth_depth_"
                     r"([0-9]+)_image_02", name)
        if not m:
            continue
        date, drive, frame = m.group(1), m.group(2), m.group(3)
        d = KITTI / date
        sync = d / f"{date}_drive_{drive}_sync"
        img = sync / "image_02" / "data" / f"{frame}.png"
        if not (img.exists() and (d / "calib_cam_to_cam.txt").exists()
                and (sync / "velodyne_points" / "data" / f"{frame}.bin").exists()):
            continue
        P = read_calib(d)[3]
        sid = f"k_{date}_{drive}_{frame}"
        rows.append({"index": len(rows), "dataset": "kitti",
                     "scene_id": f"drive_{date}_{drive}", "image": str(img),
                     "gt": "", "mask": "", "width": W, "height": H,
                     "K": P[:, :3].tolist(), "depth_convention": "z_depth_m",
                     "indoor_outdoor": "outdoor", "split": "kitti",
                     "dataset_root": str(sync), "sample_id": sid,
                     "date": date, "drive": drive, "frame": frame,
                     "gt_path": g})
    out_jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return len(rows)


def evaluate(manifest: pathlib.Path):
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    caches = {}
    for date in sorted({r["date"] for r in rows}):
        caches[date] = read_calib(KITTI / date)

    print(f"{'scene':<26}{'sid':<24}{'s_l':>8}{'raw':>8}{'@s_l':>8}{'s_gt':>8}")
    out = []
    for r in rows:
        sid = r["sample_id"]
        npz = CACHE / f"{sid}__B.npz"
        side = CACHE / f"{sid}__B.json"
        if not (npz.exists() and side.exists()):
            continue
        Rv, Tv, R0, P = caches[r["date"]]
        d = np.load(npz)["depth"].astype(np.float64)
        sd = json.loads(side.read_text())
        Hp, Wp = sd["processed_shape"]
        K = np.asarray(r["K"], float)
        f = ((K[0, 0] * Wp / W) + (K[1, 1] * Hp / H)) / 2 / 300.0
        dm = cv2.resize(d, (W, H)) * f

        gt = cv2.imread(r["gt_path"], cv2.IMREAD_UNCHANGED).astype(np.float64) / 256.0
        sub = dm[DV:DV + 352, DU:DU + 1216]
        m = (gt > 0) & np.isfinite(sub) & (sub > 0)
        if m.sum() < 2000:
            continue
        p, q = sub[m], gt[m]
        grid = np.arange(0.80, 1.30, 0.002)
        vals = [ab(x * p, q) for x in grid]
        s_gt = float(grid[int(np.argmin(vals))])

        vel = np.fromfile(str(pathlib.Path(r["dataset_root"]) / "velodyne_points"
                              / "data" / f"{r['frame']}.bin"),
                          dtype=np.float32).reshape(-1, 4)
        X = (vel[:, :3] @ Rv.T + Tv) @ R0.T
        z = X[:, 2]
        X, z = X[z > 1], z[z > 1]
        u = P[0, 0] * X[:, 0] / z + P[0, 2]
        v = P[1, 1] * X[:, 1] / z + P[1, 2]
        # crop indices for GT, full-frame indices for the model depth
        iu_c = np.round(u).astype(int) - DU
        iv_c = np.round(v).astype(int) - DV
        inb = (iu_c >= 0) & (iu_c < 1216) & (iv_c >= 0) & (iv_c < 352)
        iu_f = np.round(u).astype(int)
        iv_f = np.round(v).astype(int)
        g2 = gt[iv_c[inb], iu_c[inb]]
        mz = dm[iv_f[inb], iu_f[inb]]
        m2 = (g2 > 0) & np.isfinite(mz) & (mz > 0)
        if m2.sum() < 500:
            continue
        # the estimator compares LiDAR against the MODEL's depth; comparing it to GT
        # would measure the LiDAR/GT agreement and say nothing about the model
        lr = np.log(z[inb][m2] / mz[m2])
        if sid.startswith("k_2011_09_26_0002_0000000005"):
            print(f"[cmp] z_med={np.median(z[inb][m2]):.3f} mz_med={np.median(mz[m2]):.3f} "
                  f"gz_med={np.median(g2[m2]):.3f} lr={med if False else np.median(lr):.4f}", flush=True)
        med = np.median(lr)
        lr = lr[np.abs(lr - med) < 3 * 1.4826 * np.median(np.abs(lr - med)) + 1e-9]
        s_l = float(np.exp(np.median(lr)))
        if len(out) == 0:
            print(f"[diag] {sid} f={f:.4f} Hp={Hp} Wp={Wp} n_raw={m2.sum()} lr_med={med:.4f} s_l={s_l:.4f}", flush=True)

        a0, a1 = ab(p, q), ab(s_l * p, q)
        out.append({"sid": sid, "scene": r["scene_id"], "s_l": s_l, "s_gt": s_gt,
                    "raw": a0, "cal": a1, "n_pts": int(m2.sum())})
        print(f"{r['scene_id']:<26}{sid:<24}{s_l:>8.3f}{a0:>8.4f}{a1:>8.4f}{s_gt:>8.3f}")

    if not out:
        print("nothing evaluated")
        return
    arr = np.array([[o["raw"], o["cal"], o["s_l"], o["s_gt"]] for o in out])
    print(f"\n=== KITTI: n={len(out)} frames, "
          f"{len({o['scene'] for o in out})} drives ===")
    print(f"  raw       = {np.median(arr[:, 0]):.4f}")
    print(f"  @s_lidar  = {np.median(arr[:, 1]):.4f}   "
          f"wins {int((arr[:, 1] < arr[:, 0]).sum())}/{len(out)}")
    print(f"  s_lidar / s_gt = {np.median(arr[:, 2] / arr[:, 3]):.4f}")
    pathlib.Path("work/kitti_full_eval.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    mf = pathlib.Path("work/kitti_full.jsonl")
    n = build_manifest(mf)
    print(f"manifest: {n} frames")
    evaluate(mf)
