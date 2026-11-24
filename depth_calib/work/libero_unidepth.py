"""UniDepth on LIBERO wrist frames with exact sim GT. The robot-domain test."""
import json, torch, pathlib, numpy as np, cv2
from safetensors.torch import load_file
from unidepth.models import UniDepthV2
from omegaconf import DictConfig

D = pathlib.Path('D:/models/UNIDEPTH-V2-L')
m = UniDepthV2(DictConfig(json.loads((D/'config.json').read_text())))
m.load_state_dict(load_file(str(D/'model.safetensors')), strict=False)
m = m.to('cuda').eval()
OUT = pathlib.Path('D:/libero_out2')
meta = json.loads((OUT/'meta.json').read_text())
fx = meta['fx']

def ab(z,g): return float(np.mean(np.abs(z-g)/g))
print(f"{'frame':<10}{'raw_guess':>11}{'raw_trueK':>11}{'s_opt':>8}{'med_gt':>8}{'px':>7}")
rows=[]
for rec in meta['frames'][:8]:
    sid=rec['sid']
    img=cv2.cvtColor(cv2.imread(str(OUT/f'{sid}.png')),cv2.COLOR_BGR2RGB)
    t=torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()/255.0
    gt=np.load(OUT/f'{sid}_depth.npy').astype(np.float64)
    out=[]
    for mode in ('guess','trueK'):
        cam=None
        if mode=='trueK':
            K=np.array([[fx,0,128],[0,fx,128],[0,0,1]],float)
            cam=torch.from_numpy(K).float().unsqueeze(0).cuda()
        with torch.no_grad():
            o=m.infer(t.cuda(), camera=cam)
        d=cv2.resize(o['depth'][0,0].cpu().numpy().astype(np.float64),(256,256))
        mm=(gt>0)&np.isfinite(d)&(d>0)
        p,g=d[mm],gt[mm]
        out.append((ab(p,g),p,g))
    (rg,pg,gg),(rt,pt,gtk)=out
    grid=np.arange(0.3,4.0,0.005)
    s_opt=float(grid[int(np.argmin([ab(x*pt,gtk) for x in grid]))])
    rows.append((rg,rt,s_opt,float(np.median(gtk))))
    print(f"{sid:<10}{rg:>11.4f}{rt:>11.4f}{s_opt:>8.3f}{np.median(gtk):>8.3f}{int(mm.sum()):>7}")
r=np.array(rows)
print(f"\nmedian: raw(guess K) {np.median(r[:,0]):.4f}  raw(true K) {np.median(r[:,1]):.4f}")
print(f"        s_opt median {np.median(r[:,2]):.3f}")
