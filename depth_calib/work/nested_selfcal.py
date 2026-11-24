"""Can the NESTED model's own pose baseline calibrate its own depth? (fully self-contained)"""
import sys, types, pathlib, numpy as np, torch, itertools
from PIL import Image
V='vendor/Depth-Anything-3/src'; sys.path.insert(0,V)
shim=types.ModuleType('depth_anything_3.utils.export')
shim.export=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('x'))
sys.modules['depth_anything_3.utils.export']=shim
from depth_anything_3.api import DepthAnything3
from safetensors.torch import load_file
sys.path.insert(0,'scripts')
from crossview_scale import read_colmap

def ab(z,g): return float(np.mean(np.abs(z-g)/g))
scene='courtyard'
root=pathlib.Path(f'E:/depth_calib_data/eth3d_extracted/{scene}')
calib=read_colmap(root/'dslr_calibration_jpg')
names=['DSC_0286.JPG','DSC_0287.JPG','DSC_0288.JPG','DSC_0289.JPG']
m=DepthAnything3(model_name='da3nested-giant-large')
m.load_state_dict(load_file('D:/models/DA3NESTED/model.safetensors'),strict=False)
m=m.to(torch.device('cuda')).eval()
imgs=[Image.open(root/'images'/'dslr_images'/n).convert('RGB') for n in names]
with torch.no_grad():
    pred=m.inference(imgs,process_res=504,process_res_method='upper_bound_resize')
dep=np.asarray(pred.depth).astype(np.float64)
ex=np.asarray(pred.extrinsics)
cin=np.asarray(pred.intrinsics) if pred.intrinsics is not None else None
print('depth',dep.shape,'intrinsics',None if cin is None else cin.shape)

def baselines(c):
    return np.array([np.linalg.norm(c[i]-c[j]) for i,j in itertools.combinations(range(len(c)),2)])
pb=baselines(ex[:,:3,3])
gb=[]
for n in names:
    R,t,K,W,H=calib[n]
    gb.append(-R.T@t)
gb=baselines(np.stack(gb))
s_pose=float(np.median(gb/pb))
print(f'pose-derived scale = {s_pose:.4f}')

# evaluate depth against GT, scaled by a) nothing b) s_pose
import cv2, json
gt_rows={json.loads(l)['sample_id']:json.loads(l) for l in open('manifests/formal_all.jsonl') if l.strip()}
mapn={'DSC_0286.JPG':'eth3d_0000','DSC_0287.JPG':'eth3d_0001','DSC_0288.JPG':'eth3d_0002','DSC_0289.JPG':'eth3d_0003'}
print(f"{'view':<16}{'raw':>9}{'@s_pose':>9}{'s_gt':>7}{'px':>8}")
res=[]
for i,n in enumerate(names):
    r=gt_rows[mapn[n]]
    gro=pathlib.Path(r['dataset_root'])
    gp=np.load(gro/r['gt']).astype(np.float64); mk=np.load(gro/r['mask']).astype(bool)
    d=dep[i]
    # pred depth is at processed res; GT is native
    d2=cv2.resize(d,(r['width'],r['height']))
    m=mk&np.isfinite(d2)&(d2>0)&np.isfinite(gp)&(gp>0)
    if m.sum()<500: print(n,'skip'); continue
    p,g=d2[m],gp[m]
    grid=np.arange(0.6,1.6,0.005)
    vals=[ab(x*p,g) for x in grid]
    sgt=float(grid[int(np.argmin(vals))])
    res.append((ab(p,g),ab(s_pose*p,g),sgt))
    print(f"{n:<16}{ab(p,g):>9.4f}{ab(s_pose*p,g):>9.4f}{sgt:>7.3f}{int(m.sum()):>8}")
r=np.array(res)
print(f"\nmedian raw {np.median(r[:,0]):.4f} -> @s_pose {np.median(r[:,1]):.4f}  (s_gt median {np.median(r[:,2]):.3f})")
