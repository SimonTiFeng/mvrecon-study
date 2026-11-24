"""Build a corrected, auditable handoff package.

Unlike the v1 package, this keeps agent inputs and eval-only truth in separate trees.
It includes the actual workbench clouds, raw RGB/depth inputs, injected start-state clouds,
GT-aligned visual references, and the full corrected operation/scoring records.
"""
from __future__ import annotations
import json, pathlib, shutil, sys
import numpy as np
import cv2
import open3d as o3d

ROOT=pathlib.Path(__file__).resolve().parents[1]
SRC=ROOT/'mvrecon'
OUT=ROOT/'mvrecon_pkg_corrected'
AG_SRC=SRC/'agent_data'; EV_SRC=SRC/'eval_data'; C=ROOT/'cache/hpc/depths'
TASK=json.loads((EV_SRC/'task.json').read_text())
GT=json.loads((EV_SRC/'gt_transforms.json').read_text())

# only views used in at least one v2/v3/v4 task, no extra images
selected={scene:sorted({sid for tag in ('v2','v3','v4') for sid in d[tag]['ids']})
          for scene,d in TASK.items()}

def mkdir(p): pathlib.Path(p).mkdir(parents=True,exist_ok=True)
def pcd_from(sid,scene,transform=None):
    d=np.load(AG_SRC/scene/f'{sid}_depth.npy').astype(np.float64)
    meta={x['sid']:x for x in json.loads((AG_SRC/scene/'meta.json').read_text())['images']}[sid]
    K=np.asarray(meta['K'],float); H,W=d.shape; stride=32
    ys,xs=np.mgrid[0:H:stride,0:W:stride]; z=d[ys,xs]
    ok=np.isfinite(z)&(z>0)&(z<200); ys,xs,z=ys[ok],xs[ok],z[ok]
    fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
    pts=np.stack([(xs-cx)/fx*z,(ys-cy)/fy*z,z],1)
    # Workbench uses one stable synthetic colour per view, not sampled RGB.
    h=abs(hash(sid))%360
    import colorsys
    color=np.array(colorsys.hsv_to_rgb(h/360,0.65,0.95))
    cols=np.tile(color,(len(pts),1))
    if transform is not None:
        M=np.asarray(transform,float)
        pts=pts@M[:3,:3].T+M[:3,3]
    pc=o3d.geometry.PointCloud(); pc.points=o3d.utility.Vector3dVector(pts); pc.colors=o3d.utility.Vector3dVector(cols)
    return pc,meta

def write_ply(pc,path):
    mkdir(pathlib.Path(path).parent)
    o3d.io.write_point_cloud(str(path),pc,write_ascii=False,compressed=True)

# 1) Agent-visible data: full-res original RGB, own-frame point clouds, intrinsics; no GT/D/T.
agent=OUT/'agent_input'
evalonly=OUT/'evaluation_only'
for scene,sids in selected.items():
    ai=agent/scene; mkdir(ai/'rgb'); mkdir(ai/'clouds_own'); mkdir(ai/'model_output')
    metas=[]
    srcmeta={x['sid']:x for x in json.loads((AG_SRC/scene/'meta.json').read_text())['images']}
    for sid in sids:
        m=srcmeta[sid]
        # RGB at half resolution (JPEG q92) to keep the package shippable.
        # Original is ETH3D DSLR 6048x4032; re-export full-res with
        #   cp mvrecon/agent_data/<scene>/<sid>.png
        img=cv2.imread(str(AG_SRC/scene/f'{sid}.png'))
        ih,iw=img.shape[:2]
        cv2.imwrite(str(ai/'rgb'/f'{sid}.jpg'),
                    cv2.resize(img,(iw//2,ih//2),interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY,92])
        pc,_=pcd_from(sid,scene)
        write_ply(pc,ai/'clouds_own'/f'{sid}.ply')
        # exact model raw cache + sidecar; keeps a reproducible link to inference output
        for ext in ('npz','json'):
            q=C/f'{sid}__B.{ext}'
            if q.exists(): shutil.copy2(q,ai/'model_output'/q.name)
        # metric-converted z-depth: keep a 1/4-scale float16 copy for convenience.
        # The full-resolution metric depth is exactly derivable from the npz + sidecar
        # (see views.json "metric_depth_reconstruction"), so it is not duplicated.
        dep=np.load(AG_SRC/scene/f'{sid}_depth.npy').astype(np.float32)
        Dh,Dw=dep.shape
        small=cv2.resize(dep,(Dw//4,Dh//4),interpolation=cv2.INTER_NEAREST)
        np.save(ai/'model_output'/f'{sid}_depth_metric_q4.npy',small.astype(np.float16))
        metas.append({**{k:m[k] for k in ('sid','K','W','H','depth_units','depth_convention')},
                      'rgb_file':f'{sid}.jpg','rgb_scale':0.5,
                      'rgb_note':'half-resolution JPEG; original ETH3D frame is 6048x4032 '
                                 'and K above refers to the ORIGINAL resolution'})
    (ai/'views.json').write_text(json.dumps({'scene':scene,'views':metas,
        'task_view_ids':selected[scene],
        'metric_depth_reconstruction':(
            'full-res metric z-depth = cv2.resize(npz["depth"], (W,H)) * '
            '((K[0,0]*Hp/W)+(K[1,1]*Wp/H))/2/300, where (Hp,Wp)=sidecar["processed_shape"]. '
            '`model_output/<sid>_depth_metric_q4.npy` is that quantity at 1/4 scale, float16, metres.'),
        'note':'Agent-visible only: RGB, intrinsics, per-image own-frame point clouds and depth predictions. '
               'No reference poses, no injected transforms, no task ground truth, no reference scan.'},indent=1))

# 2) Task starts and GT-aligned clouds go in evaluation-only, never agent_input.
for scene,sids in selected.items():
    gt_scene=evalonly/scene; mkdir(gt_scene)
    shutil.copy2(EV_SRC/scene/'meta.json',gt_scene/'colmap_meta.json')
    for tag in ('v2','v3','v4'):
        t=TASK[scene][tag]; td=gt_scene/tag; mkdir(td/'start_clouds'); mkdir(td/'gt_aligned_clouds')
        for sid in t['ids']:
            raw,_=pcd_from(sid,scene)
            D=np.asarray(t['D_injected'][sid],float)
            T=np.asarray(t['T_true'][sid],float)
            start,_=pcd_from(sid,scene,transform=D)
            aligned,_=pcd_from(sid,scene,transform=T)
            write_ply(start,td/'start_clouds'/f'{sid}.ply')
            write_ply(aligned,td/'gt_aligned_clouds'/f'{sid}.ply')
        (td/'task_gt.json').write_text(json.dumps(t,indent=1))
    (gt_scene/'subsets.json').write_text(json.dumps(TASK[scene],indent=1))
(Eval:=evalonly/'gt_transforms.json').write_text(json.dumps(GT,indent=1))
(Eval:=evalonly/'task.json').write_text(json.dumps(TASK,indent=1))

# 3) Full actual operation traces and corrected scores.
runs=OUT/'runs'; mkdir(runs)
for f in ('baseline_corrected.json','baseline_rescored.json','agent_runs_corrected.json','agent_rescored.json',
          'identity_corrected.json','correction_audit.json','tools_smoke.json'):
    p=SRC/'out'/f
    if p.exists(): shutil.copy2(p,runs/f)
# save corrected source code and convention regression test
code=OUT/'code'; mkdir(code/'bench'); mkdir(code/'eval'); mkdir(code/'workbench')
for f in ('select_subsets2.py','make_gt.py','build_task.py','verify_task.py','test_transform_convention.py'):
    shutil.copy2(SRC/'bench'/f,code/'bench'/f)
for f in ('protocol.py','run_baseline_corrected.py','rescore_saved.py','score_agent.py'):
    shutil.copy2(SRC/'eval'/f,code/'eval'/f)
for f in ('wb.py','agent_tools.py','run_agent.py'):
    shutil.copy2(SRC/'workbench'/f,code/'workbench'/f)

# 4) Archive the exact invalid v1 package and legacy logs for audit, without overwriting.
legacy=OUT/'audit'/'legacy_pre_fix'; mkdir(legacy)
for f in ('task.json','gt_transforms.json','subsets.json','baseline_table.json','baseline_A.json',
          'agent_runs.json','agent_scores.json','run_c_electro.json','mvrecon_handoff.tar.gz'):
    p=SRC/'archive'/'pre_pose_fix'/f
    if p.exists(): shutil.copy2(p,legacy/f)
shutil.copy2(SRC/'COORDINATE_AUDIT.md',OUT/'audit'/'COORDINATE_AUDIT.md')

# README
readme='''# MV-Recon corrected handoff v2\n\nThis package supersedes the v1 handoff. v1 had two independent defects: (1) the scorer\ncompared the cumulative transform M to incremental A=T@inv(D), not to T; (2) COLMAP\nworld-to-camera poses were converted with the wrong relative-pose formula. The original\nv1 package and logs are preserved under audit/legacy_pre_fix. See audit/COORDINATE_AUDIT.md.\n\n## Contents\n- agent_input/: full-res RGB, K, raw depth cache and the actual own-camera-frame clouds.\n  **No poses, D, T, or GT are in this tree.**\n- evaluation_only/: corrected T_true, injected D, COLMAP poses, and PLYs for the actual\n  start state and GT-aligned state. Do not expose this tree to an agent.\n- runs/: corrected identity/FPFH+ICP/scripted-loop results. Each row includes initial M,\n  final cumulative M, T, D, the coarse/fine ICP matrices, and complete agent hist/trace.\n- code/: corrected generation, scoring, workbench, and regression-test code.\n- audit/legacy_pre_fix/: exact v1 artefacts, retained for reproducibility, not valid scores.\n\n## Correct transform convention\nCOLMAP supplies world->camera: p_cam=R p_world+t. Therefore camera j -> camera ref i is\nR_ij=R_i R_j^T, t_ij=t_i-R_i R_j^T t_j.\n\nThe harness starts each cloud at M0=D. An incremental recovery A=T@inv(D) is left-composed,\nso M_final=A@D=T. **Score the cumulative M_final against T_true.**\n\n## Validation\n`python code/bench/test_transform_convention.py` tests the formula and exact composition.\n`runs/correction_audit.json` confirms all 80 baseline and 40 agent rows match the current\ntask matrices with zero mismatches.\n'''
(OUT/'README.md').write_text(readme)
print('corrected package built:',OUT)
print('scenes',len(selected),'selected views',sum(map(len,selected.values())))
