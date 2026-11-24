"""DA3 on LIBERO with CORRECT metric depth. The definitive robot-domain test."""
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, '.')
from src.config import load_config
from src.backends.da3 import DA3Backend

OUT = pathlib.Path('D:/libero_v3')
meta = json.loads((OUT / 'meta.json').read_text())
cfg = load_config('configs/pilot.yaml')
be = DA3Backend(cfg['backends']['B'], cfg['backend_inference']).load()


def ab(z, g):
    return float(np.mean(np.abs(z - g) / g))


for cam, fx in (('img_ag', meta['fx_ag']), ('img_wr', meta['fx_wr'])):
    print('\n=== %s (fx=%.1f) ===' % (cam, fx))
    print('%-10s%9s%8s%8s%9s%9s' % ('sid', 'raw_med', 'gt_med', 's_opt', 'cal', 'rawAbs'))
    rows = []
    tag = 'age' if cam == 'img_ag' else 'rob'
    for rec in meta['frames']:
        if rec['sid'].split('_')[0] != tag:
            continue
        sid = rec['sid']
        d = np.asarray(be.predict(str(OUT / (sid + '.png')), (256, 256))['depth'], float)
        d = cv2.resize(d, (256, 256))
        g = np.load(OUT / ('%s_depth.npy' % sid)).astype(float)
        m = (g > 0) & np.isfinite(d) & (d > 0)
        if m.sum() < 1000:
            continue
        p, gt = d[m], g[m]
        grid = np.logspace(-1, 2, 500)
        vals = [ab(x * p, gt) for x in grid]
        s_opt = float(grid[int(np.argmin(vals))])
        raw = ab(p, gt)
        cal = ab(s_opt * p, gt)
        rows.append((raw, s_opt, cal))
        print('%-10s%9.4f%8.3f%8.3f%9.4f%9.4f'
              % (sid, float(np.median(p)), float(np.median(gt)), s_opt, cal, raw))
    if rows:
        r = np.array(rows)
        print('  -> raw %.4f  s_opt %.3f  cal %.4f'
              % (float(np.median(r[:, 0])), float(np.median(r[:, 1])), float(np.median(r[:, 2]))))
