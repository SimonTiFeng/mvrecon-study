import sys, numpy as np, pathlib
sys.path.insert(0,'mvrecon/workbench')
from agent_tools import AgentTools
t = AgentTools('mvrecon/agent_data/office', stride=24)
print('views', t.list_views()['views'][:3], 'ref', t.list_views()['reference'])
img = t.render('front')
print('render ok:', None if 'error' in img else img['image'].shape)
m = t.measure(t.list_views()['views'][0], (100,80,180,160))
print('measure:', m.get('diag_m'), 'm')
r = t.transform(t.list_views()['views'][1], t=np.array([0.2,0,0]))
print('transform ok:', r.get('ok'), 'overlap', round(r.get('overlap') or 0,3))
u = t.undo(); print('undo:', u)
print('budget:', t.report()['used'])
