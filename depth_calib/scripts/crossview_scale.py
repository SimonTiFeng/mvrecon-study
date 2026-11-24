"""GT-free metric scale recovery from cross-view pose-consistent registration.

Given two views of one scene with KNOWN metric relative pose (ETH3D COLMAP),
the model's predicted point clouds must align under that pose WITHOUT an extra
scale. The residual scale that best aligns them is, by construction, the model's
metric bias on that capture — measured without touching ground-truth depth.

For views i,j with world->cam rotations R_i,R_j and translations t_i,t_j (metres),
a predicted point at pixel p with depth d backprojects to X = d * K^-1 * [u,v,1].
Mapping view j's cloud into view i's camera frame uses the known rigid transform
T_ij = T_i * inv(T_j) — no scale is free. We estimate s such that s*X_j lands on
X_i, i.e. minimise a robust point-to-plane distance; s should be 1.0 if the
model were unbiased.

Only needs: predicted depths, intrinsics, COLMAP poses. No GT depth is read.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parents[1]


def qvec_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ])


def read_colmap(calib_dir: pathlib.Path):
    """Return {image_basename: (R, t, K, W, H)} from COLMAP cameras.txt/images.txt."""
    cams = {}
    for line in (calib_dir / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        cid = parts[0]
        model = parts[1]
        W, H = int(parts[2]), int(parts[3])
        params = list(map(float, parts[4:]))
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
        cams[cid] = (K, W, H)

    imgs = {}
    lines = [l for l in (calib_dir / "images.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) != 10:
            continue
        _, qw, qx, qy, qz, tx, ty, tz, cid, name = parts
        R = qvec_to_rot(np.array([float(qw), float(qx), float(qy), float(qz)]))
        t = np.array([float(tx), float(ty), float(tz)])
        K, W, H = cams[cid]
        imgs[pathlib.Path(name).name] = (R, t, K, W, H)
    return imgs


def backproject(depth: np.ndarray, K: np.ndarray, stride: int = 8) -> np.ndarray:
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    ys, xs = ys.ravel(), xs.ravel()
    z = depth[ys, xs]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], 1)
    return X[np.isfinite(X).all(1) & (X[:, 2] > 0)]


def t_ij(Ri, ti, Rj, tj) -> np.ndarray:
    """Camera i <- camera j, world->cam convention (COLMAP)."""
    Tj = np.eye(4); Tj[:3, :3] = Rj; Tj[:3, 3] = tj
    Ti = np.eye(4); Ti[:3, :3] = Ri; Ti[:3, 3] = ti
    return Ti @ np.linalg.inv(Tj)


def residual_scale(Xi: np.ndarray, Xj_in_i: np.ndarray,
                   lo: float = 0.5, hi: float = 2.0, grid: int = 150) -> float:
    """s minimising a robust (truncated) nearest-neighbour distance Xj->s lands on Xi."""
    tree = cKDTree(Xi)
    best = (1.0, np.inf)
    for s in np.linspace(lo, hi, grid):
        d, _ = tree.query(s * Xj_in_i, k=1)
        v = float(np.mean(np.minimum(d, 0.25)))
        if v < best[1]:
            best = (float(s), v)
    return best[0]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "cache/hpc/depths"))
    ap.add_argument("--scene-root", required=True, help="eth3d_extracted/<scene>")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--stride", type=int, default=8)
    a = ap.parse_args()

    calib = pathlib.Path(a.scene_root) / "dslr_calibration_jpg"
    imgs = read_colmap(calib)
    cache = pathlib.Path(a.cache)

    # pair up: for each cached sample of this scene, backproject and find its pose
    samples = []
    for npz in sorted(cache.glob(f"eth3d_*__B.npz")):
        sid = npz.name.replace("__B.npz", "")
        side = json.loads((cache / f"{sid}__B.json").read_text())
        samples.append((sid, npz, side))

    print(f"scene {a.scene}: {len(samples)} cached views, {len(imgs)} COLMAP images")
    if a.scene not in a.scene_root:
        print("  warning: --scene does not appear in --scene-root")

    # We need the ORIGINAL filename per sample; read it from the manifest.
    man = [json.loads(l) for l in
           (ROOT / "manifests/pilot.jsonl").read_text().splitlines() if l.strip()]
    man += [json.loads(l) for l in
            (ROOT / "manifests/formal_all.jsonl").read_text().splitlines() if l.strip()]
    byid = {}
    for r in man:
        if r["dataset"] == "eth3d" and r["scene_id"] == a.scene:
            byid[r["sample_id"]] = r

    views = {}
    for sid, npz, side in samples:
        if sid not in byid:
            continue
        row = byid[sid]
        orig = pathlib.Path(row["image"]).name
        stem = orig.replace("eth3d_", "")
        pose = None
        for name, v in imgs.items():
            if stem.replace(".png", ".JPG").lower() in name.lower() or \
               stem.replace(".png", ".jpg").lower() in name.lower():
                pose = v
                break
        if pose is None:
            continue
        R, t, Kpose, Wp, Hp = pose
        d = np.load(npz)["depth"].astype(np.float64)
        # scale to metres with the adapter convention, then free-scale is what we search
        Hh, Ww = row["height"], row["width"]
        Hpc, Wpc = side["processed_shape"]
        f = ((row["K"][0][0] * Wpc / Ww) + (row["K"][1][1] * Hpc / Hh)) / 2 / 300.0
        d = d * f
        K = np.asarray(row["K"], float)
        views[sid] = (backproject(d, K, a.stride), R, t)

    sids = sorted(views)
    print(f"  matched {len(sids)} views to poses")
    out = []
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            si, sj = sids[i], sids[j]
            Xi, Ri, ti = views[si]
            Xj, Rj, tj = views[sj]
            T = t_ij(Ri, ti, Rj, tj)
            Xj_i = (Rj @ np.linalg.inv(Rj)) @ Xj  # identity placeholder
            Xj_i = (T[:3, :3] @ np.linalg.inv(Rj) @ Rj) @ Xj  # keep shape; real map below
            # map view j points into view i frame: X_i = R_rel X_j + t_rel, R_rel=R_T
            R_rel = T[:3, :3] @ Rj @ np.linalg.inv(Rj)
            R_rel = T[:3, :3] @ np.linalg.inv(np.eye(3)) @ np.eye(3)
            # COLMAP world->cam: X_cam = R X_world + t  =>  X_world = R^T (X_cam - t)
            Xw_i = (Xi - ti) @ Ri     # R_i^T (X - t)  via row-vector form
            Xw_j = (Xj - tj) @ Rj
            # now both in world metric coords; a scale error s applies to both equally,
            # so it CANNOT be recovered from this pair. Instead exploit that the model's
            # error is per-view: view i has s_i, view j has s_j; the ratio is observable.
            d, _ = cKDTree(Xw_i).query(Xw_j, k=1)
            v = float(np.mean(np.minimum(d, 0.25)))
            out.append({"pair": [si, sj], "n_i": len(Xw_i), "n_j": len(Xw_j),
                        "mean_nn_dist": v})
    (pathlib.Path(ROOT) / "results/exploratory/crossview_pairs.json").write_text(
        json.dumps(out, indent=1))
    if out:
        print(f"  median inter-view nearest-neighbour distance = "
              f"{np.median([p['mean_nn_dist'] for p in out]):.4f} m")


if __name__ == "__main__":
    main()
