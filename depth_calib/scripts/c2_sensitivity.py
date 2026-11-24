"""C2 sensitivity test: does the radial scale estimator track a KNOWN multiplier?

Take two cached views of one ETH3D scene, apply a known factor k to one side's
depth (simulating a model whose bias changed by k), and check that the estimator
returns s ~ 1/k (or k, depending on which side is scaled). A degenerate estimator
returns ~1.0 regardless of k.

Uses the frozen local cache and the COLMAP poses. No new inference, no GT.
"""
from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def q2r(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                     [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                     [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]])


def read_colmap(calib_dir):
    cams = {}
    for l in (calib_dir / "cameras.txt").read_text().splitlines():
        if l.startswith("#") or not l.strip():
            continue
        p = l.split()
        K = np.array([[float(p[4]), 0, float(p[6])],
                      [0, float(p[5]), float(p[7])], [0, 0, 1]])
        cams[p[0]] = (K, int(p[2]), int(p[3]))
    imgs = {}
    lines = [l for l in (calib_dir / "images.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        if len(p) != 10:
            continue
        _, qw, qx, qy, qz, tx, ty, tz, cid, name = p
        R = q2r(np.array([float(qw), float(qx), float(qy), float(qz)]))
        t = np.array([float(tx), float(ty), float(tz)])
        K, W, H = cams[cid]
        imgs[pathlib.Path(name).name] = (R, t, K, W, H)
    return imgs


def backproject(d, K, stride=24, maxd=120.0):
    H, W = d.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    ys, xs = ys.ravel(), xs.ravel()
    z = d[ys, xs]
    ok = np.isfinite(z) & (z > 0) & (z < maxd)
    ys, xs, z = ys[ok], xs[ok], z[ok]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], 1)


def radial_scale(A, B):
    ca, cb = np.median(A, 0), np.median(B, 0)
    ra = np.linalg.norm(A - ca, axis=1)
    rb = np.linalg.norm(B - cb, axis=1)
    q = np.array([0.25, 0.5, 0.75, 0.9])
    return float(np.median(np.quantile(ra, q) / np.maximum(np.quantile(rb, q), 1e-9)))


def main():
    calib_dir = pathlib.Path(sys.argv[1])
    cache = ROOT / "cache/hpc/depths"
    manifest = [json.loads(l) for l in
                (ROOT / "manifests/formal_all.jsonl").read_text().splitlines() if l.strip()]
    imgs = read_colmap(calib_dir)

    # map source jpg -> sample_id via the manifest's own provenance field
    src2sid = {}
    for r in manifest:
        if r.get("dataset") == "eth3d" and r.get("source_image"):
            src2sid[pathlib.Path(r["source_image"]).name] = r["sample_id"]

    pairs = [("DSC_0286.JPG", "DSC_0287.JPG"), ("DSC_0288.JPG", "DSC_0289.JPG")]
    print(f"{'pair':<26}{'k':>7}{'s_est':>9}{'1/k':>8}{'tracks?':>9}")
    for na, nb in pairs:
        sa = src2sid.get(na); sb = src2sid.get(nb)
        if not (sa and sb):
            print(f"  no cache for {na}/{nb}: looked up {sa}/{sb}")
            continue
        da = np.load(cache / f"{sa}__B.npz")["depth"].astype(np.float64)
        db = np.load(cache / f"{sb}__B.npz")["depth"].astype(np.float64)
        Ra, ta, Ka, Wa, Ha = imgs[na]
        Rb, tb, _, Wb, Hb = imgs[nb]
        da = cv2.resize(da, (Wa, Ha), interpolation=cv2.INTER_LINEAR)
        db = cv2.resize(db, (Wb, Hb), interpolation=cv2.INTER_LINEAR)
        for k in (0.80, 0.90, 1.00, 1.10, 1.20):
            # scale B's depth by k; the fitted s should move by ~1/k
            A_cam = backproject(da, Ka)
            B_cam = backproject(db * k, imgs[nb][2])
            A_w = (A_cam - ta) @ Ra
            B_w = (B_cam - tb) @ Rb
            s = radial_scale(A_w, B_w)
            print(f"{na[:11]+'|'+nb[:11]:<26}{k:>7.2f}{s:>9.4f}{1/k:>8.4f}"
                  f"{'YES' if abs(s*1/k-1)<0.25 else 'no':>9}")


if __name__ == "__main__":
    main()
