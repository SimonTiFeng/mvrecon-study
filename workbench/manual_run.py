"""Manual control loop: does the workbench actually allow a successful alignment?

This is ME acting as the agent, using only the tool feedback (no GT in the loop).
"""
import sys, pathlib, numpy as np, json
sys.path.insert(0,'mvrecon/workbench')
from wb import Workbench
import open3d as o3d

wb = Workbench('mvrecon/agent_data/office', stride=24)
GT = json.loads(pathlib.Path('mvrecon/eval_data/gt_transforms.json').read_text())
ids = GT['office']['v2']['ids']
ref, tgt = ids[0], ids[1]
wb.set_reference(ref)
M_gt = np.asarray(GT['office']['v2']['T'][tgt])

print(f'ref={ref}  target={tgt}')
print(f'true relative transform: rot '
      f'{np.degrees(np.arccos(np.clip((np.trace(M_gt[:3,:3])-1)/2,-1,1))):.1f} deg, '
      f'trans {np.linalg.norm(M_gt[:3,3]):.3f} m')
print()
print('--- what the agent can see (no GT) ---')
print(f'initial overlap {wb.overlap_stats(tgt)[0]:.3f}  residual {wb.residual(tgt):.3f}')
print(f'{ref} extent: {np.round(wb.clouds[ref].transformed().min(0),2)} .. '
      f'{np.round(wb.clouds[ref].transformed().max(0),2)}')
print(f'{tgt} extent: {np.round(wb.clouds[tgt].transformed().min(0),2)} .. '
      f'{np.round(wb.clouds[tgt].transformed().max(0),2)}')

# Step 1: coarsely centre the target on the reference (pure geometry, allowed)
tgt_c = wb.clouds[tgt].transformed()
ref_c = wb.clouds[ref].transformed()
delta = np.median(ref_c,0) - np.median(tgt_c,0)
wb.apply(tgt, t=np.array(delta))
print(f'\nafter centroid alignment: overlap {wb.overlap_stats(tgt)[0]:.3f} '
      f'residual {wb.residual(tgt):.3f}')

# Step 2: try ICP with the workbench's own clouds (local refine)
src = wb.clouds[tgt].as_o3d()
dst = wb.clouds[ref].as_o3d()
for name, est in (('point2point', o3d.pipelines.registration.TransformationEstimationPointToPoint()),
                  ('point2plane', o3d.pipelines.registration.TransformationEstimationPointToPlane())):
    if name=='point2plane':
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.4,max_nn=20))
        dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.4,max_nn=20))
    r = o3d.pipelines.registration.registration_icp(src, dst, 0.6, np.eye(4), est,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80))
    M = wb.clouds[tgt].matrix() @ np.linalg.inv(np.eye(4))
    # apply the ICP result in the workbench's world frame
    wb._snap()
    T = r.transformation
    c = wb.clouds[tgt]
    newM = T @ c.matrix()
    # decompose back to (R,t,s)
    Sc = np.cbrt(abs(np.linalg.det(newM[:3,:3])))
    Rc = newM[:3,:3]/Sc
    c.R, c.t, c.s = Rc, newM[:3,3], float(c.s*Sc)
    wb.log.append({'op':'icp','variant':name,'fitness':float(r.fitness),'rmse':float(r.inlier_rmse)})
    print(f'  ICP {name:12s}: fitness {r.fitness:.3f} rmse {r.inlier_rmse:.3f} '
          f'-> overlap {wb.overlap_stats(tgt)[0]:.3f}')
