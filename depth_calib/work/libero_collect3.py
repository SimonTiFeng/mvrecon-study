"""Collect LIBERO wrist frames with CORRECT mujoco-rendered metric depth."""
import sys, os, json, pathlib
sys.path.insert(0, r'D:/libero_src')
os.environ.setdefault('MUJOCO_GL','wgl')
import numpy as np, cv2, mujoco
from libero.libero.envs import OffScreenRenderEnv
BDDL = r'D:/libero_src/libero/libero/bddl_files/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl'
OUT = pathlib.Path('D:/libero_v3'); OUT.mkdir(exist_ok=True)
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256,
                         camera_names=['agentview','robot0_eye_in_hand'], camera_depths=False)
env.reset()
m = env.sim.model._model; d = env.sim.data._data
ren = {}
for cname in ('agentview','robot0_eye_in_hand'):
    r = mujoco.Renderer(m, 256, 256); r.enable_depth_rendering(); ren[cname]=r
recs=[]
for step in range(10):
    act=np.array([0.5,0.3 if step%2 else -0.3,0.3 if step%3==0 else 0,0,0,0,1.0])
    env.step(act)
    for cname,key in (('agentview','img_ag'),('robot0_eye_in_hand','img_wr')):
        r=ren[cname]; r.update_scene(d, camera=cname)
        dep=r.render()
        sid=f'{cname[:3]}_{step:02d}'
        np.save(OUT/f'{sid}_depth.npy', dep.astype(np.float32))
        cv2.imwrite(str(OUT/f'{sid}.png'), cv2.cvtColor(env.sim.render(256,256,camera_name=cname), cv2.COLOR_RGB2BGR))
        recs.append({'sid':sid,'cam':cname,'dmin':float(dep.min()),'dmax':float(dep.max()),
                     'dmed':float(np.median(dep))})
    print(f'step {step}: ag {recs[-2]["dmin"]:.2f}-{recs[-2]["dmax"]:.2f}  wr {recs[-1]["dmin"]:.2f}-{recs[-1]["dmax"]:.2f}', flush=True)
meta={'frames':recs,'H':256,'W':256,
      'fx_ag':256/2/np.tan(np.deg2rad(float(env.sim.model.cam_fovy[env.sim.model.camera_name2id('agentview')])/2)),
      'fx_wr':256/2/np.tan(np.deg2rad(float(env.sim.model.cam_fovy[env.sim.model.camera_name2id('robot0_eye_in_hand')])/2))}
(OUT/'meta.json').write_text(json.dumps(meta,indent=1))
print('fx ag',round(meta['fx_ag'],1),'wr',round(meta['fx_wr'],1))
env.close()
