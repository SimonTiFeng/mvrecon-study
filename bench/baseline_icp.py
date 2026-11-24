"""Baseline A: global registration + local ICP refine, no semantics.

Reports success rate, rotation error, translation error, for v2/v3/v4 subsets.
"""
import json, pathlib, numpy as np, open3d as o3d

AG=pathlib.Path('mvrecon/agent_data')
GT=json.loads(pathlib.Path('mvrecon/eval_data/gt_transforms.json').read_text())

def load_cloud(scene, sid, stride=24):
    d=np.load(AG/scene/f'{sid}_depth.npy').astype(np.float64)
    K=np.asarray(json.loads((AG/scene/'meta.json').read_text())['images'][0]['K']) if False else None
    meta={x['sid']:x for x in json.loads((AG/scene/'meta.json').read_text())['images']}
    K=np.asarray(meta[sid]['K']); H,W=d.shape
    ys,xs=np.mgrid[0:H:stride,0:W:stride]
    z=d[ys,xs]; ok=(z>0)&np.isfinite(z)&(z<200)
    ys,xs,z=ys[ok],xs[ok],z[ok]
    fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
    P=np.stack([(xs-cx)/fx*z,(ys-cy)/fy*z,z],1)
    pc=o3d.geometry.PointCloud()
    pc.points=o3d.utility.Vector3dVector(P)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,max_nn=20))
    return pc

def rot_err(R_est, R_gt):
    c=(np.trace(R_est.T@R_gt)-1)/2
    return float(np.degrees(np.arccos(np.clip(c,-1,1))))

def run_pair(src, tgt, gtM):
    # coarse: global FPFH+RANSAC, then ICP
    src_f=src.voxel_down_sample(0.3); tgt_f=tgt.voxel_down_sample(0.3)
    src_f.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.6,max_nn=30))
    tgt_f.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.6,max_nn=30))
    f_s=o3d.pipelines.registration.compute_fpfh_feature(src_f,
        o3d.geometry.KDTreeSearchParamHybrid(radius=1.2,max_nn=100))
    f_t=o3d.pipelines.registration.compute_fpfh_feature(tgt_f,
        o3d.geometry.KDTreeSearchParamHybrid(radius=1.2,max_nn=100))
    dist=1.5
    coarse=o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_f,tgt_f,f_s,f_t,True,dist,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),3,[],
        o3d.pipelines.registration.RANSACConvergenceCriteria(400000,0.999))
    M=coarse.transformation
    fine=o3d.pipelines.registration.registration_icp(src,tgt,0.5,M,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return fine.transformation, coarse.fitness, fine.fitness

import sys
ONLY = sys.argv[1] if len(sys.argv)>1 else None
print(f"{'scene':<15}{'tag':<4}{'pair':<26}{'fitness':>9}{'rotErr':>9}{'transErr':>10}{'ok':>4}")
results=[]
for scene in sorted(GT):
    if ONLY and scene != ONLY: continue
    for tag in ('v2','v3','v4'):
        if tag not in GT[scene]: continue
        d=GT[scene][tag]; ids=d['ids']
        for j in range(1,len(ids)):
            a,b = ids[j], ids[0]
            src=load_cloud(scene, a); tgt=load_cloud(scene, b)
            gtM=np.asarray(d['T'][a])
            M,f1,f2=run_pair(src,tgt,gtM)
            re=rot_err(M[:3,:3], gtM[:3,:3])
            te=float(np.linalg.norm(M[:3,3]-gtM[:3,3]))
            ok = (re<10) and (te<0.5)
            results.append({'scene':scene,'tag':tag,'pair':f'{a}|{b}','rot_err':re,
                            'trans_err':te,'fitness':float(f2),'ok':bool(ok)})
            print(f"{scene:<15}{tag:<4}{a+'|'+b:<26}{f2:>9.3f}{re:>9.1f}{te:>10.3f}{'Y' if ok else 'N':>4}")
r=np.array([[x['rot_err'],x['trans_err'],1.0 if x['ok'] else 0.0] for x in results])
print(f"\nBaseline A: n={len(results)}  success {int(r[:,2].sum())}/{len(r)}"
      f"  median rot {np.median(r[:,0]):.1f}deg  median trans {np.median(r[:,1]):.3f}m")
pathlib.Path('mvrecon/out/baseline_A.json').write_text(json.dumps(results,indent=1))
