"""Package the MV-Recon data for handoff: point clouds, images, depth, poses, GT."""
import json, pathlib, shutil
import numpy as np, cv2, open3d as o3d

SRC = pathlib.Path('mvrecon')
OUT = pathlib.Path('mvrecon_pkg/data')
OUT.mkdir(parents=True, exist_ok=True)

TASK = json.loads((SRC/'eval_data/task.json').read_text())
GT   = json.loads((SRC/'eval_data/gt_transforms.json').read_text())

def build_ply(scene, sid, meta, stride=32):
    d = np.load(SRC/'agent_data'/scene/f'{sid}_depth.npy').astype(np.float64)
    K = np.asarray(meta['K'])
    H, W = d.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    z = d[ys, xs]
    ok = np.isfinite(z) & (z > 0) & (z < 200)
    ys, xs, z = ys[ok], xs[ok], z[ok]
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    P = np.stack([(xs-cx)/fx*z, (ys-cy)/fy*z, z], 1)
    img = cv2.imread(str(SRC/'agent_data'/scene/f'{sid}.png'))
    ih, iw = img.shape[:2]
    us = (xs * iw / W).astype(int).clip(0, iw-1)
    vs = (ys * ih / H).astype(int).clip(0, ih-1)
    bgr = img[vs, us][:, ::-1].astype(np.float64)/255.0
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P)
    pc.colors = o3d.utility.Vector3dVector(bgr)
    dst = OUT/scene/'clouds'; dst.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(dst/f'{sid}.ply'), pc, write_ascii=False, compressed=True)
    return len(P)

index = {'description': 'MV-Recon handoff package', 'scenes': {}}
for scene in sorted(TASK):
    meta = {x['sid']: x for x in
            json.loads((SRC/'agent_data'/scene/'meta.json').read_text())['images']}
    n_total = 0
    for sid, m in meta.items():
        # images: half-resolution JPEG
        img = cv2.imread(str(SRC/'agent_data'/scene/f'{sid}.png'))
        h, w = img.shape[:2]
        small = cv2.resize(img, (w//4, h//4), interpolation=cv2.INTER_AREA)
        d = OUT/scene/'images'; d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d/f'{sid}.jpg'), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
        # depth: float16 at quarter res (metres)
        dep = np.load(SRC/'agent_data'/scene/f'{sid}_depth.npy').astype(np.float32)
        Dh, Dw = dep.shape
        dep_s = cv2.resize(dep, (Dw//4, Dh//4), interpolation=cv2.INTER_NEAREST)
        dd = OUT/scene/'depth16'; dd.mkdir(parents=True, exist_ok=True)
        np.save(dd/f'{sid}.npy', dep_s.astype(np.float16))
        n_total += build_ply(scene, sid, m)
    # per-scene metadata
    (OUT/scene/'meta.json').write_text(json.dumps({
        'scene': scene,
        'camera_intrinsics': {s: {'K': m['K'], 'W': m['W'], 'H': m['H']} for s, m in meta.items()},
        'note': 'intrinsics are for the ORIGINAL resolution; images here are 1/4 scale, '
                'depth16 is 1/4 scale; clouds are at full metric scale in each camera frame',
        'depth_units': 'metres, z along optical axis, model focal/300 convention applied',
    }, indent=1))
    # task + gt for this scene
    (OUT/scene/'task.json').write_text(json.dumps({'scene': scene, 'subsets': TASK[scene]}))
    (OUT/scene/'gt_transforms.json').write_text(json.dumps({'scene': scene, 'gt': GT[scene]}))
    index['scenes'][scene] = {'views': len(meta), 'cloud_points_total': n_total}

(OUT/'index.json').write_text(json.dumps(index, indent=1))
print(json.dumps(index, indent=1))
