"""Run DA3-metric on LIBERO wrist frames and score against exact sim depth."""
import json, pathlib, numpy as np, cv2, torch
from PIL import Image
ROOT = pathlib.Path('.')
import sys; sys.path.insert(0, str(ROOT))
from src.config import load_config
from src.backends.da3 import DA3Backend

OUT = pathlib.Path('D:/libero_out2')
meta = json.loads((OUT/'meta.json').read_text())
cfg = load_config(ROOT/'configs/pilot.yaml')
be = DA3Backend(cfg['backends']['B'], cfg['backend_inference']).load()

fx_true = meta['fx']
def ab(z,g): return float(np.mean(np.abs(z-g)/g))
print(f"{'frame':<12}{'raw':>9}{'s_opt':>8}{'cal':>9}{'med_gt':>9}")
rows=[]
for rec in meta['frames']:
    sid = rec['sid']
    png = OUT/f'{sid}.png'
    dep_gt = np.load(OUT/f'{sid}_depth.npy').astype(np.float64)
    r = be.predict(str(png), (256,256))
    d = np.asarray(r['depth'], dtype=np.float64)
    # DA3 metric needs the focal/300 conversion; use the TRUE intrinsics here
    # because the sim gives them exactly
    fac = fx_true/300.0
    d2 = cv2.resize(d,(256,256))*fac
    m = np.isfinite(d2)&(d2>0)&np.isfinite(dep_gt)&(dep_gt>0)
    if m.sum()<1000: print(sid,'skip'); continue
    p,g = d2[m], dep_gt[m]
    grid=np.arange(0.3,3.0,0.005)
    s_opt=float(grid[int(np.argmin([ab(x*p,g) for x in grid]))])
    raw=ab(p,g)
    rows.append((raw,s_opt,ab(s_opt*p,g),float(np.median(g))))
    print(f"{sid:<12}{raw:>9.4f}{s_opt:>8.3f}{rows[-1][2]:>9.4f}{np.median(g):>9.3f}")
r=np.array(rows)
print(f"\nmedian raw {np.median(r[:,0]):.4f}  s_opt {np.median(r[:,1]):.3f}  calibrated {np.median(r[:,2]):.4f}")
