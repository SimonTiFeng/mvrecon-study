"""NESTED gives predicted c2w extrinsics. Compare their baselines to COLMAP metric ones."""
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

m=DepthAnything3(model_name='da3nested-giant-large')
sd=load_file('D:/models/DA3NESTED/model.safetensors')
m.load_state_dict(sd,strict=False)
m=m.to(torch.device('cuda')).eval()

scene='courtyard'
root=pathlib.Path(f'E:/depth_calib_data/eth3d_extracted/{scene}')
calib=read_colmap(root/'dslr_calibration_jpg')
names=['DSC_0286.JPG','DSC_0287.JPG','DSC_0288.JPG','DSC_0289.JPG']
imgs=[Image.open(root/'images'/'dslr_images'/n).convert('RGB') for n in names]
with torch.no_grad():
    pred=m.inference(imgs,process_res=504,process_res_method='upper_bound_resize')
ex=np.asarray(pred.extrinsics)  # (N,3,4) c2w
print('extrinsics',ex.shape,'is_metric',getattr(pred,'is_metric',None))
centres=ex[:,:3,3]
print('predicted camera centres:'); print(np.round(centres,3))

gc=[]
for n in names:
    R,t,K,W,H=calib[n]
    gc.append(-R.T@t)
gc=np.stack(gc)
print('\nCOLMAP metric centres:'); print(np.round(gc,3))

def baselines(c):
    return np.array([np.linalg.norm(c[i]-c[j]) for i,j in itertools.combinations(range(len(c)),2)])
pb=baselines(centres); gb=baselines(gc)
print(f'\n{"pair":<12}{"pred":>9}{"colmap":>9}{"ratio":>8}')
for (i,j),p,g in zip(itertools.combinations(range(len(names)),2),pb,gb):
    print(f'{names[i][4:8]+"-"+names[j][4:8]:<12}{p:>9.3f}{g:>9.3f}{g/p:>8.3f}')
print(f'\nmedian ratio COLMAP/predicted = {np.median(gb/pb):.4f}')
print('If this is ~1.0 the model pose is metric; >1 means it under-scales.')
