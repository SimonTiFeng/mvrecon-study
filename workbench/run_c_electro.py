"""Baseline C, first real run: closed-loop alignment using ONLY tool feedback.

Points are electrode; the agent (me) may not read GT until after it stops.
"""
import sys, json, pathlib, numpy as np
sys.path.insert(0,'mvrecon/workbench')
from agent_tools import AgentTools

SCENE='electro'
t = AgentTools(f'mvrecon/agent_data/{SCENE}', stride=24)
views = t.list_views()['views']
print('views:', views, 'ref:', t.list_views()['reference'])
ref = t.list_views()['reference']
tgt = [v for v in views if v!=ref][0]
print(f'binding {tgt} to {ref}\n')

steps=[]
def step(name, **kw):
    r = kw.pop('fn')()
    steps.append({'step':name, **{k:v for k,v in r.items() if k!='image'}})
    print(f'  [{name}] ' + json.dumps({k:v for k,v in r.items() if k!='image'})[:150])
    return r

print('1. baseline state')
step('stats_init', fn=lambda: t.stats(tgt))

print('2. coarse: align centroids')
c_t = t.wb.clouds[tgt].transformed(); c_r = t.wb.clouds[ref].transformed()
d = np.median(c_r,0) - np.median(c_t,0)
step('centroid', fn=lambda: t.transform(tgt, t=d))

print('3. try solver with two picked regions (same uv across both, since overlap is high)')
step('solve_uv', fn=lambda: t.solve_correspondences(tgt,(400,300,1200,900),(400,300,1200,900)))

print('4. refine with ICP')
step('icp', fn=lambda: t.refine_icp(tgt, radius=0.5))

print('5. final stats + render')
step('stats_final', fn=lambda: t.stats(tgt))
img = t.render('front', exclude=())
print('render shape', img['image'].shape)
print('\nbudget:', t.report()['used'])

pathlib.Path(f'mvrecon/out/run_c_{SCENE}.json').write_text(
    json.dumps({'scene':SCENE,'ref':ref,'target':tgt,'steps':steps,'trace':t.trace},indent=1))
