"""ETH3D DSLR ingest: JPG + raw float32 ground-truth depth + COLMAP intrinsics.

Archive facts verified 2026-11-17:
  <scene>_dslr_jpg.7z    : images/dslr_images/*.JPG + dslr_calibration_jpg/{cameras,images}.txt
  <scene>_dslr_depth.7z  : ground_truth_depth/dslr_images/*.JPG, headerless float32 (H,W),
                           invalid = +inf, values in metres (z-depth).
Images are resized so the long side is <= --max-side (default 1280) and K is scaled
accordingly; GT depth/mask are resized with nearest neighbour.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import tempfile

import numpy as np
import cv2
import py7zr
from PIL import Image

SCENE_TYPE = {
    "courtyard": "outdoor", "delivery_area": "outdoor", "facade": "outdoor",
    "terrace": "outdoor", "terrains": "outdoor", "bridge": "outdoor",
    "boulders": "outdoor", "playground": "outdoor", "pipes": "outdoor",
    "electro": "indoor", "office": "indoor", "kicker": "indoor",
    "living_room": "indoor", "loungeroom": "indoor", "relief": "indoor",
    "relief_2": "indoor", "door": "indoor", "university": "indoor",
}


def parse_cameras(path: pathlib.Path) -> dict[int, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cid, model, w, h = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        p = [float(x) for x in parts[4:]]
        if model == "PINHOLE":
            fx, fy, cx, cy = p[:4]
        elif model in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE", "RADIAL"):
            f, cx, cy = p[0], p[1], p[2]
            fx = fy = f
        elif model == "OPENCV":
            fx, fy, cx, cy = p[:4]
        elif model == "THIN_PRISM_FISHEYE":
            fx, fy, cx, cy = p[:4]
        else:
            raise ValueError(f"unsupported camera model {model}")
        out[cid] = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "w": w, "h": h,
                    "model": model, "params": p}
    return out


SCALE_IDX = {
    "PINHOLE": [0, 1, 2, 3],
    "SIMPLE_RADIAL": [0, 1, 2],
    "SIMPLE_PINHOLE": [0, 1, 2],
    "RADIAL": [0, 1, 2],
    "OPENCV": [0, 1, 2, 3],
    "THIN_PRISM_FISHEYE": [0, 1, 2, 3],
}


def scale_params(model: str, params: list[float], sx: float, sy: float) -> list[float]:
    out = list(params)
    if model in ("PINHOLE", "OPENCV", "THIN_PRISM_FISHEYE"):
        out[0] *= sx; out[1] *= sy; out[2] *= sx; out[3] *= sy
    else:  # SIMPLE_RADIAL / SIMPLE_PINHOLE / RADIAL
        out[0] *= (sx + sy) / 2.0; out[1] *= sx; out[2] *= sy
    return out


def parse_images(path: pathlib.Path) -> dict[str, int]:
    """COLMAP images.txt: metadata lines have 10 tokens; points2D lines follow each."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 10:
            continue
        try:
            int(parts[0]); int(parts[8])
        except ValueError:
            continue
        out[pathlib.Path(parts[9]).name] = int(parts[8])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", help="dir with <scene>_dslr_{jpg,depth}.7z (legacy mode)")
    ap.add_argument("--extracted", help="pre-extracted dir containing <scene>/...")
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-side", type=int, default=0,
                    help="0 = keep native resolution (formal scope)")
    ap.add_argument("--max-per-scene", type=int, default=0,
                    help="keep only the first N images per scene (0 = all)")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    for sub in ("images", "gt", "mask"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    entries = []
    skipped = {}
    for scene in args.scenes:
        try:
            if args.extracted:
                base = pathlib.Path(args.extracted) / scene
                if not base.exists():
                    print(f"[skip] missing extracted scene {scene}")
                    skipped[scene] = "missing_extracted"
                    continue
                cams = parse_cameras(base / "dslr_calibration_jpg" / "cameras.txt")
                imgs = parse_images(base / "dslr_calibration_jpg" / "images.txt")
                img_dir = base / "images" / "dslr_images"
                dep_dir = base / "ground_truth_depth" / "dslr_images"
            else:
                jpg7 = pathlib.Path(args.archives) / f"{scene}_dslr_jpg.7z"
                dep7 = pathlib.Path(args.archives) / f"{scene}_dslr_depth.7z"
                if not jpg7.exists() or not dep7.exists():
                    print(f"[skip] missing archives for {scene}")
                    skipped[scene] = "missing_archives"
                    continue
                tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"eth_{scene}_"))
                with py7zr.SevenZipFile(jpg7) as z:
                    z.extractall(path=tmp)
                with py7zr.SevenZipFile(dep7) as z:
                    z.extractall(path=tmp)
                cams = parse_cameras(tmp / scene / "dslr_calibration_jpg" / "cameras.txt")
                imgs = parse_images(tmp / scene / "dslr_calibration_jpg" / "images.txt")
                img_dir = tmp / scene / "images" / "dslr_images"
                dep_dir = tmp / scene / "ground_truth_depth" / "dslr_images"
        except ValueError as e:
            print(f"[skip] {scene}: {e}")
            skipped[scene] = f"unsupported_camera: {e}"
            continue
        idx = 0
        for name, cam_id in sorted(imgs.items()):
            if args.max_per_scene and idx >= args.max_per_scene:
                break
            fimg = img_dir / name
            fdep = dep_dir / name
            if not fimg.exists() or not fdep.exists():
                continue
            W0, H0 = Image.open(fimg).size
            raw = np.fromfile(fdep, dtype=np.float32)
            if raw.size != W0 * H0:
                raise ValueError(f"{scene}/{name}: depth {raw.size} != {W0*H0}")
            dep = raw.reshape(H0, W0)
            valid = np.isfinite(dep) & (dep > 0)
            if valid.mean() < 0.02:
                continue
            scale = 1.0 if args.max_side <= 0 else min(1.0, args.max_side / max(W0, H0))
            W, H = int(round(W0 * scale)), int(round(H0 * scale))
            pil = Image.open(fimg)
            pil.draft("RGB", (W, H))            # JPEG DCT downscale before decode
            rgb = pil.convert("RGB")
            img = np.asarray(rgb if (W, H) == rgb.size else rgb.resize((W, H), Image.BILINEAR))
            dep_r = cv2.resize(dep, (W, H), interpolation=cv2.INTER_NEAREST)
            msk_r = cv2.resize(valid.astype(np.uint8), (W, H),
                               interpolation=cv2.INTER_NEAREST) > 0
            c = cams[cam_id]
            cam_params = scale_params(c["model"], c["params"], W / W0, H / H0)
            K = [[c["fx"] * W / W0, 0, c["cx"] * W / W0],
                 [0, c["fy"] * H / H0, c["cy"] * H / H0], [0, 0, 1]]
            Image.fromarray(np.asarray(img)).save(out / "images" / f"{scene}_{idx:04d}.png")
            np.save(out / "gt" / f"{scene}_{idx:04d}.npy", dep_r.astype(np.float32))
            np.save(out / "mask" / f"{scene}_{idx:04d}.npy", msk_r)
            entries.append({
                "index": idx, "dataset": "eth3d", "scene_id": scene,
                "image": f"images/{scene}_{idx:04d}.png",
                "gt": f"gt/{scene}_{idx:04d}.npy",
                "mask": f"mask/{scene}_{idx:04d}.npy",
                "width": W, "height": H, "K": K,
                "camera_model": c["model"], "camera_params": cam_params,
                "depth_convention": "z_depth_m",
                "indoor_outdoor": SCENE_TYPE.get(scene, "unknown"),
                "source_image": name, "original_size": [W0, H0]})
            idx += 1
        print(f"[{scene}] {idx} images")
    (out / "meta.json").write_text(json.dumps({
        "dataset": "eth3d", "source": args.extracted or args.archives,
        "scenes": args.scenes, "skipped": skipped,
        "depth_note": "headerless float32 metres; invalid=+inf; valid fraction sparse (laser GT)",
        "resize_note": ("native resolution (no resize)" if args.max_side <= 0
                        else f"long side <= {args.max_side}; K scaled; depth/mask nearest"),
        "entries": entries}, indent=1), encoding="utf-8")
    print(f"wrote {len(entries)} entries to {out}")


if __name__ == "__main__":
    main()
