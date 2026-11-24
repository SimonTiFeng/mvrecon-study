"""Trace a single KITTI frame through the estimator to find why s_l is low."""
import json
import pathlib

import cv2
import numpy as np

import importlib.util

spec = importlib.util.spec_from_file_location(
    "k", pathlib.Path(__file__).resolve().parent / "kitti_full.py")
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)

rows = [json.loads(l) for l in open("work/kitti_full.jsonl")]
r = [x for x in rows if x["drive"] == "0002"][0]
Rv, Tv, R0, P = k.read_calib(pathlib.Path("D:/kitti/2011_09_26"))
C = k.CACHE

d = np.load(C / f"{r['sample_id']}__B.npz")["depth"].astype(np.float64)
sd = json.loads((C / f"{r['sample_id']}__B.json").read_text())
Hp, Wp = sd["processed_shape"]
K = np.asarray(r["K"], float)
f = ((K[0, 0] * Wp / k.W) + (K[1, 1] * Hp / k.H)) / 2 / 300.0
print(f"factor={f:.4f}  processed={Hp}x{Wp}  frame={k.W}x{k.H}")

dm = cv2.resize(d, (k.W, k.H)) * f
gt = cv2.imread(r["gt_path"], cv2.IMREAD_UNCHANGED).astype(np.float64) / 256.0
sub = dm[k.DV:k.DV + 352, k.DU:k.DU + 1216]
mm = (gt > 0) & np.isfinite(sub) & (sub > 0)
print("M/GT (full crop):", round(float(np.median(sub[mm] / gt[mm])), 4))

vel = np.fromfile(str(pathlib.Path(r["dataset_root"]) / "velodyne_points" / "data"
                   / f"{r['frame']}.bin"), dtype=np.float32).reshape(-1, 4)
X = (vel[:, :3] @ Rv.T + Tv) @ R0.T
z = X[:, 2]
X, z = X[z > 1], z[z > 1]
u = P[0, 0] * X[:, 0] / z + P[0, 2]
v = P[1, 1] * X[:, 1] / z + P[1, 2]
iu = np.round(u).astype(int) - k.DU
iv = np.round(v).astype(int) - k.DV
inb = (iu >= 0) & (iu < 1216) & (iv >= 0) & (iv < 352)
print("pts inside crop:", int(inb.sum()), "of", len(z))

g2 = gt[iv[inb], iu[inb]]
mz = sub[iv[inb], iu[inb]]
ok = (g2 > 0) & np.isfinite(mz) & (mz > 0)
print("pts usable:", int(ok.sum()))
lr = np.log(z[inb][ok] / mz[ok])
print("s_l (L/model):", round(float(np.exp(np.median(lr))), 4))
print("s vs GT (L/GT):", round(float(np.exp(np.median(np.log(z[inb][ok] / g2[ok])))), 4))
print("model/GT on those pts:", round(float(np.median(mz[ok] / g2[ok])), 4))
