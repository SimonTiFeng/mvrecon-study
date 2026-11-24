"""DA3 on LIBERO with a proper scale search (no focal assumption)."""
import json, pathlib, numpy as np, cv2, sys
sys.path.insert(0,'.')
from src.config import load_config
from src.backends.da3 import DA3Backend

OUT = pathlib.Path('D:/libero_out2')
meta = json.loads((OUT/'meta.json').read_text())
cfg = load_config('configs/pilot.yaml')
be = DA3Backend(cfg['backends']['B'], cfg['backend_inference']).load()

def ab(z,g): return float(np.mean(np.abs(z-g)/g))
print(f"{'frame':<10}{'raw_med':>9}{'gt_med':>8}{'s_opt':>8}{'cal':>9}")
rows=[]
for rec in meta['frames']:
    sid=rec['sid']
    d=np.asarray(be.predict(str(OUT/f'{sid}.png'),(256,256))['depth'],float)
    d=cv2.resize(d,(256,256))
    g=np.load(OUT/f'{sid}_depth.npy').astype(float)
    m=(g>0)&np.isfinite(d)&(d>0)
    p,gt=d[m],g[m]
    # wide search because LIBERO is a hard case
    grid=np.logspace(-1,2,400)
    s_opt=float(grid[int(np.argmin([ab(x*p,gt) for x in grid]))])
    raw=ab(p,gt); cal=ab(s_opt*p,gt)
    rows.append((raw,s_opt,cal,float(np.median(p))))
    print(f"{sid:<10}{np.median(p):>9.4f}{np.median(gt):>8.3f}{s_opt:>8.3f}{cal:>9.4f}")
r=np.array(rows)
print(f"\nDA3 on LIBERO: raw median {np.median(r[:,0]):.4f}   s_opt {np.median(r[:,1]):.3f}"
      f"   calibrated {np.median(r[:,2]):.4f}")
