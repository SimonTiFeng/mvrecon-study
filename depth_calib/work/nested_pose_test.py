import sys, types, pathlib, numpy as np, torch
from PIL import Image
V = 'vendor/Depth-Anything-3/src'; sys.path.insert(0, V)
shim = types.ModuleType('depth_anything_3.utils.export')
shim.export = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('x'))
sys.modules['depth_anything_3.utils.export'] = shim
from depth_anything_3.api import DepthAnything3
from safetensors.torch import load_file

m = DepthAnything3(model_name='da3nested-giant-large')
sd = load_file('D:/models/DA3NESTED/model.safetensors')
mm, uu = m.load_state_dict(sd, strict=False)
print('load:', len(mm), 'missing', len(uu), 'unexpected')
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m = m.to(dev).eval()
print('device', dev)

base = pathlib.Path('E:/depth_calib_data/eth3d_extracted/courtyard/images/dslr_images')
names = ['DSC_0286.JPG', 'DSC_0287.JPG', 'DSC_0288.JPG']
imgs = [Image.open(base / n).convert('RGB') for n in names]
print('images', [i.size for i in imgs])
with torch.no_grad():
    pred = m.inference(imgs, process_res=504, process_res_method='upper_bound_resize')
print('depth', np.asarray(pred.depth).shape)
print('extrinsics shape:', None if pred.extrinsics is None else np.asarray(pred.extrinsics).shape)
print('is_metric:', getattr(pred, 'is_metric', None))
if pred.extrinsics is not None:
    ex = np.asarray(pred.extrinsics)
    print('extrinsic[0]:'); print(np.round(ex[0], 3))
