"""Render start-vs-GT-aligned previews for the corrected task (v3 subsets)."""
import json,pathlib,numpy as np,open3d as o3d,cv2
P=pathlib.Path('mvrecon_pkg_corrected'); OUT=P/'preview'; OUT.mkdir(exist_ok=True)

def load(sc,kind,tag,sid):
    return o3d.io.read_point_cloud(str(P/'evaluation_only'/sc/tag/kind/f'{sid}.ply'))

def render(pcs,size=(520,380)):
    pts=np.concatenate([np.asarray(p.points) for p in pcs],0)
    cen=pts.mean(0); rad=float(np.linalg.norm(pts-cen,axis=1).max())
    vis=o3d.visualization.Visualizer(); vis.create_window(visible=False,width=size[0],height=size[1])
    for p in pcs: vis.add_geometry(p)
    vc=vis.get_view_control(); vc.set_front(cen+np.array([0,-3*rad,0.4*rad])-cen)
    vc.set_up(np.array([0,0,1])); vc.set_zoom(0.65)
    vis.poll_events(); vis.update_renderer()
    img=np.asarray(vis.capture_screen_float_buffer(False)); vis.destroy_window()
    return (img*255).astype(np.uint8)

for scene in ('electro','office','terrace','playground','facade','delivery_area','courtyard'):
    task=json.loads((P/'evaluation_only'/scene/'v3'/'task_gt.json').read_text())
    ids=task['ids']
    row=np.hstack([render([load(scene,'start_clouds','v3',s) for s in ids]),
                   render([load(scene,'gt_aligned_clouds','v3',s) for s in ids])])
    cv2.imwrite(str(OUT/f'{scene}_v3_start_vs_aligned.png'),cv2.cvtColor(row,cv2.COLOR_RGB2BGR))
    print('wrote',scene)
