"""Generic NESTED pose-baseline calibration across ETH3D scenes.

For each scene: feed N views to the nested model, derive a scale from predicted
camera baselines vs COLMAP metric baselines, then evaluate the depth before/after.
No depth ground truth is used to derive the scale.
"""
import sys
import types
import pathlib
import itertools
import json

import numpy as np
import torch
import cv2
from PIL import Image

V = str(pathlib.Path(__file__).resolve().parents[1] / 'vendor/Depth-Anything-3/src')
sys.path.insert(0, V)
shim = types.ModuleType('depth_anything_3.utils.export')
shim.export = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('x'))
sys.modules['depth_anything_3.utils.export'] = shim
from depth_anything_3.api import DepthAnything3
from safetensors.torch import load_file
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from crossview_scale import read_colmap


def ab(z, g):
    return float(np.mean(np.abs(z - g) / g))


def gt_opt(p, g):
    if p.size > 4000:
        i = np.linspace(0, p.size - 1, 4000).astype(np.int64)
        p, g = p[i], g[i]
    r = p / g
    best = (1.0, np.inf)
    for s in np.arange(0.6, 1.6, 0.01):
        v = float(np.mean(np.abs(s * r - 1.0)))
        if v < best[1]:
            best = (float(s), v)
    return best[0]


def main(scene, names, rows, model):
    root = pathlib.Path.home()/'jhspoolers/dc_data/eth3d_extracted'/scene
    calib = read_colmap(root / 'dslr_calibration_jpg')
    imgs = [Image.open(root / 'images' / 'dslr_images' / n).convert('RGB') for n in names]
    with torch.no_grad():
        pred = model.inference(imgs, process_res=336,
                               process_res_method='upper_bound_resize')
    dep = np.asarray(pred.depth).astype(np.float64)
    ex = np.asarray(pred.extrinsics)

    def baselines(c):
        return np.array([np.linalg.norm(c[i] - c[j])
                         for i, j in itertools.combinations(range(len(c)), 2)])
    pb = baselines(ex[:, :3, 3])
    gc = []
    for n in names:
        R, t = calib[n][0], calib[n][1]
        gc.append(-R.T @ t)
    gb = baselines(np.stack(gc))
    s_pose = float(np.median(gb / np.maximum(pb, 1e-9)))

    print(f"\n=== {scene} ===")
    print(f"  pose scale {s_pose:.4f}  (from {len(pb)} view pairs)")
    print(f"  {'view':<14}{'raw':>9}{'@pose':>9}{'s_gt':>7}{'px':>9}")
    res = []
    for i, n in enumerate(names):
        sid = rows[n]
        r = [json.loads(l) for l in open(str(pathlib.Path(__file__).resolve().parents[1]/'manifests/formal_all.jsonl'))
             if json.loads(l)['sample_id'] == sid][0]
        gro = pathlib.Path(r['dataset_root'])
        gp = np.load(gro / r['gt']).astype(np.float64)
        mk = np.load(gro / r['mask']).astype(bool)
        d2 = cv2.resize(dep[i], (r['width'], r['height']))
        m = mk & np.isfinite(d2) & (d2 > 0) & np.isfinite(gp) & (gp > 0)
        if m.sum() < 500:
            continue
        p, g = d2[m], gp[m]
        sgt = gt_opt(p, g)
        res.append((ab(p, g), ab(s_pose * p, g), sgt))
        print(f"  {n:<14}{res[-1][0]:>9.4f}{res[-1][1]:>9.4f}{sgt:>7.3f}{int(m.sum()):>9}")
    if res:
        a = np.array(res)
        print(f"  median raw {np.median(a[:,0]):.4f} -> {np.median(a[:,1]):.4f}"
              f"   wins {int((a[:,1]<a[:,0]).sum())}/{len(a)}")
        print(f"  s_gt median {np.median(a[:,2]):.3f}   s_pose/s_gt {s_pose/np.median(a[:,2]):.4f}")
    return s_pose, res


if __name__ == '__main__':
    rows_map = {}
    for l in open(str(pathlib.Path(__file__).resolve().parents[1]/'manifests/formal_all.jsonl')):
        r = json.loads(l)
        if r['dataset'] == 'eth3d' and r.get('source_image'):
            rows_map[pathlib.Path(r['source_image']).name] = r['sample_id']

    m = DepthAnything3(model_name='da3nested-giant-large')
    m.load_state_dict(load_file(str(pathlib.Path.home() / 'agent/depth_calib/models/DA3NESTED/model.safetensors')), strict=False)
    m = m.to(torch.device('cuda')).eval()

    import os
    ONLY = os.environ.get('SC')
    ALL = [('facade', ['DSC_0324.JPG','DSC_0325.JPG','DSC_0326.JPG','DSC_0327.JPG']),
           ('playground', ['DSC_0567.JPG','DSC_0568.JPG','DSC_0569.JPG','DSC_0570.JPG']),
           ('delivery_area', ['DSC_0675.JPG','DSC_0676.JPG','DSC_0677.JPG','DSC_0678.JPG'])]
    plan = [p for p in ALL if ONLY is None or p[0] == ONLY]
    out = {}
    for sc, names in plan:
        s, res = main(sc, names, rows_map, m)
        out[sc] = {'s_pose': s, 'results': res,
                   'median_raw': float(np.median([r[0] for r in res])) if res else None,
                   'median_cal': float(np.median([r[1] for r in res])) if res else None}
    (pathlib.Path(__file__).resolve().parents[1]/'work/nested_scenes.json').write_text(json.dumps(out, indent=1))
    print("\nwrote work/nested_scenes.json")
