"""GT-free scale from COLMAP 3-D landmarks (Option B).

COLMAP's points3D.txt holds SfM landmark positions in metres together with the
image tracks that observed them. For a landmark observed in a view we know its
metric depth along that camera's optical axis. Reading the model's predicted
depth at the same pixel and taking the ratio gives the model's scale bias
DIRECTLY -- a measured quantity against a metric ruler, not a ratio of two
things that both carry the bias.

No ground-truth depth is read. The metric reference is the SfM landmark cloud.
"""
from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np


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
    imgs, by_id = {}, {}
    lines = [l for l in (calib_dir / "images.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        if len(p) != 10:
            continue
        iid, qw, qx, qy, qz, tx, ty, tz, cid, name = p
        R = q2r(np.array([float(qw), float(qx), float(qy), float(qz)]))
        t = np.array([float(tx), float(ty), float(tz)])
        K, W, H = cams[cid]
        entry = (R, t, K, W, H)
        imgs[pathlib.Path(name).name] = entry
        by_id[int(iid)] = pathlib.Path(name).name
    return imgs, by_id


def read_points(points3d: pathlib.Path):
    """Yield (xyz_metres, [(image_id, point2d_idx), ...])."""
    out = []
    for l in points3d.read_text().splitlines():
        if l.startswith("#") or not l.strip():
            continue
        p = l.split()
        xyz = np.array([float(p[1]), float(p[2]), float(p[3])])
        err = float(p[7])
        rest = p[8:]
        track = [(int(rest[i]), int(rest[i + 1])) for i in range(0, len(rest) - 1, 2)]
        out.append((xyz, err, track))
    return out


def main():
    scene_root = pathlib.Path(sys.argv[1])
    sample_json = pathlib.Path(sys.argv[2])   # {"image_name": "sid", ...}
    cache = pathlib.Path(sys.argv[3])         # dir with <sid>__B.npz
    root_proc = pathlib.Path(sys.argv[4])     # dir with processed/<...> GT? not needed

    calib = scene_root / "dslr_calibration_jpg"
    imgs, by_id = read_colmap(calib)
    pts = read_points(calib / "points3D.txt")
    name2sid = json.loads(sample_json.read_text())
    print(f"calibration: {len(imgs)} images, {len(pts)} landmarks")

    # Build: for each image_id, which points project where.
    per_image = {}
    for xyz, err, track in pts:
        for iid, p2d in track:
            per_image.setdefault(iid, []).append((xyz, err, p2d))

    sids_by_name = {}
    for n, sid in name2sid.items():
        sids_by_name[pathlib.Path(n).name] = sid

    all_s = []
    for iid, name in sorted(by_id.items()):
        sid = sids_by_name.get(name)
        if sid is None:
            continue
        npz = cache / f"{sid}__B.npz"
        side = cache / f"{sid}__B.json"
        if not npz.exists():
            print(f"  {name}: no cached depth, skip")
            continue
        R, t, K, W, H = imgs[name]
        d = np.load(npz)["depth"].astype(np.float64)
        sd = json.loads(side.read_text())
        Hp, Wp = sd["processed_shape"]
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
        # model depth is in metres only after the focal conversion; the ratio we
        # want is model-vs-SfM, so use the SAME conversion the pipeline uses.
        # Here we report the raw ratio and let the caller normalise.

        rows = []
        for xyz, err, p2d in per_image.get(iid, []):
            if err > 2.0:
                continue
            Xc = R @ xyz + t          # camera frame, metres
            z_metric = float(Xc[2])
            if z_metric <= 0.05 or z_metric > 60:
                continue
            u = K[0, 0] * Xc[0] / Xc[2] + K[0, 2]
            v = K[1, 1] * Xc[1] / Xc[2] + K[1, 2]
            iu, iv = int(round(u)), int(round(v))
            if not (2 <= iu < W - 2 and 2 <= iv < H - 2):
                continue
            patch = d[iv-1:iv+2, iu-1:iu+2]
            z_model = float(np.median(patch))
            if not np.isfinite(z_model) or z_model <= 0:
                continue
            rows.append((z_metric, z_model))

        if len(rows) < 20:
            print(f"  {name} ({sid}): only {len(rows)} usable landmarks, skip")
            continue
        zt = np.array([r[0] for r in rows])
        zm = np.array([r[1] for r in rows])
        # robust ratio, in log space, trimmed
        lr = np.log(zt / zm)
        lr = lr[np.abs(lr - np.median(lr)) < 3 * 1.4826 * np.median(np.abs(lr - np.median(lr)) + 1e-9)]
        s = float(np.exp(np.median(lr)))
        sd = float(np.std(lr))
        med_z = float(np.median(zt))
        # (a) few points or wide scatter -> unreliable match
        ok = (len(rows) >= 100) and (sd <= 0.15)
        all_s.append((name, sid, s, int(len(rows)), sd, bool(ok), med_z))
        print(f"  {name} ({sid}): s={s:.4f} from {len(rows)} landmarks "
              f"(log sd {sd:.3f}, med z {med_z:.1f} m)"
              f"{'' if ok else '   [REJECTED]'}")

    if all_s:
        good = [x for x in all_s if x[5]]
        arr = np.array([x[2] for x in good]) if good else np.array([x[2] for x in all_s])
        print(f"\nscene median s = {np.median(arr):.4f}  sd={arr.std():.4f}  "
              f"n={len(arr)} (accepted {len(good)}/{len(all_s)})")
        pathlib.Path(sys.argv[5]).write_text(json.dumps(
            [{"name": a, "sid": b, "s": c, "n": d, "log_sd": e, "ok": f, "med_z": g}
             for a, b, c, d, e, f, g in all_s], indent=1))


if __name__ == "__main__":
    main()
