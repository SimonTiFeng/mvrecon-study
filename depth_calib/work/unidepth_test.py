import json, torch, pathlib
from safetensors.torch import load_file
from unidepth.models import UniDepthV2
from omegaconf import OmegaConf, DictConfig

D = pathlib.Path('D:/models/UNIDEPTH-V2-L')
cfg = json.loads((D/'config.json').read_text())
print('config keys:', list(cfg.keys())[:8])
# build a DictConfig from the plain json
params = DictConfig(cfg)
m = UniDepthV2(params)
sd = load_file(str(D/'model.safetensors'))
miss, unexp = m.load_state_dict(sd, strict=False)
print('missing', len(miss), 'unexpected', len(unexp))
if miss: print('first missing:', miss[:3])
m = m.to('cuda').eval()
print('loaded on cuda')
