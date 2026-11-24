"""Is the model's own camera frame already a common frame?

If two clouds overlap heavily at IDENTITY, the registration task is trivial and the
subset is not measuring what the study intends to measure.
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, 'mvrecon/workbench')
from wb import Workbench

GT = json.loads(pathlib.Path('mvrecon/eval_data/gt_transforms.json').read_text())


def rotdeg(M):
    c = (np.trace(M[:3, :3]) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))

print('%-14s%-22s%16s%10s%9s' % ('scene', 'pair', 'overlap@identity', 'true|t|', 'true rot'))
for scene in sorted(GT):
    d = GT[scene]['v2']
    ids = d['ids']
    wb = Workbench('mvrecon/agent_data/%s' % scene, stride=32)
    a, b = ids[1], ids[0]
    M = np.asarray(d['T'][a])
    ov = wb.overlap_stats(a)[0]
    print('%-14s%-22s%16.3f%10.3f%9.1f'
          % (scene, a + '|' + b, ov, float(np.linalg.norm(M[:3, 3])), rotdeg(M)))
