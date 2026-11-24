import json, torch, pathlib, numpy as np, cv2
from safetensors.torch import load_file
from unidepth.models import UniDepthV2
from omegaconf import DictConfig
D = pathlib.Path('D:/models/UNIDEPTH-V2-L')
m = UniDepthV2(DictConfig(json.loads((D/'config.json').read_text())))
m.load_state_dict(load_file(str(D/'model.safetensors')), strict=False)
m = m.to('cuda').eval()
img = cv2.imread('D:/kitti/2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000005.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float() / 255.0
with torch.no_grad():
    out = m.infer(t.cuda())
print('output keys:', list(out.keys()))
print('depth shape:', out['depth'].shape, 'min', float(out['depth'].min()), 'max', float(out['depth'].max()))
print('confidence:', None if out.get('confidence') is None else out['confidence'].shape)
print('intrinsics:', out['intrinsics'].shape)
print(np.round(out['intrinsics'][0].cpu().numpy(), 2))
