"""Render side-by-side previews: start state (with D injected) vs true alignment."""
import json, pathlib, numpy as np, open3d as o3d, cv2

PKG = pathlib.Path('mvrecon_pkg/data')
OUT = pathlib.Path('mvrecon_pkg/preview'); OUT.mkdir(exist_ok=True)

def load(scene, sid):
    return o3d.io.read_point_cloud(str(PKG/scene/'clouds'/f'{sid}.ply'))

def render(pcs, size=(520,380)):
    pts=np.concatenate([np.asarray(p.points) for p in pcs],0)
    cen=pts.mean(0); rad=float(np.linalg.norm(pts-cen,axis=1).max())
    vis=o3d.visualization.Visualizer(); vis.create_window(visible=False,width=size[0],height=size[1])
    for p in pcs: vis.add_geometry(p)
    vc=vis.get_view_control()
    vc.set_front(cen+np.array([0,-3*rad,0.4*rad])-cen); vc.set_up(np.array([0,0,1])); vc.set_zoom(0.65)
    vis.poll_events(); vis.update_renderer()
    img=np.asarray(vis.capture_screen_float_buffer(False))
    vis.destroy_window()
    return (img*255).astype(np.uint8)

for scene in ('electro','office','terrace','playground'):
    task=json.loads((PKG/scene/'task.json').read_text())['subsets']['v3']
    ids=task['ids']; ref=task['ref']
    base=[load(scene,ref)]
    for sid in ids[1:]:
        c=load(scene,sid)
        M=np.asarray(task['D_injected'][sid])
        c.transform(M); base.append(c)
    A=[load(scene,ref)]
    for sid in ids[1:]:
        c=load(scene,sid)
        D=np.asarray(task['D_injected'][sid]); T=np.asarray(task['T_true'][sid])
        c.transform(T@np.linalg.inv(D)); A.append(c)
    row=np.hstack([render(base), render(A)])
    cv2.imwrite(str(OUT/f'{scene}_v3.png'), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
    print('wrote', scene)
