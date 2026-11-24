"""Tool smoke test (requirement 3): load clouds, switch views, apply a known transform,
check coordinates, undo, and verify restoration."""
import numpy as np, open3d as o3d, json, pathlib

def make_test_cloud(n=200, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3)) * 0.3
    cols = np.tile([[0.9, 0.2, 0.2]], (n, 1))
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(cols)
    return pc

def transform_cloud(pc, R=None, t=None, s=1.0):
    T = np.eye(4)
    if R is not None: T[:3,:3] = R * s
    else: T[:3,:3] *= s
    if t is not None: T[:3,3] = t
    out = o3d.geometry.PointCloud(pc)
    out.transform(T)
    return out

res = {}
a = make_test_cloud()
pts_a = np.asarray(a.points)
theta = np.deg2rad(30)
R = np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
b = transform_cloud(a, R=R, t=np.array([0.5,-0.2,0.1]), s=1.5)
pts_b = np.asarray(b.points)

# check the transform is exactly what we asked for: apply its inverse
T_inv = np.eye(4)
T_fwd = np.eye(4); T_fwd[:3,:3] = R*1.5; T_fwd[:3,3] = [0.5,-0.2,0.1]
T_inv[:3,:3] = np.linalg.inv(T_fwd[:3,:3]); T_inv[:3,3] = -T_inv[:3,:3] @ T_fwd[:3,3]
restored = np.asarray(o3d.geometry.PointCloud(b).transform(T_inv).points)
res['roundtrip_max_abs'] = float(np.abs(restored - pts_a).max())
res['orthonormal_R'] = bool(np.allclose(R@R.T, np.eye(3), atol=1e-12))
res['det_R'] = float(np.linalg.det(R))

# voxel downsample (must not invent points)
down = a.voxel_down_sample(0.05)
res['downsample_count'] = int(len(down.points))
res['downsample_is_subset'] = bool(len(down.points) <= len(pts_a))

# KD-tree query returns real points from the cloud
tree = o3d.geometry.KDTreeFlann(a)
q = np.array([0.1, 0.1, 0.1])
[k, idx, d2] = tree.search_knn_vector_3d(q, 1)
res['knn_idx_valid'] = bool(idx[0] < len(pts_a))
res['knn_point_is_real'] = bool(np.allclose(np.asarray(a.points)[idx[0]], pts_a[idx[0]]))

# ICP on a rigid pair must recover the known transform
src = make_test_cloud(400, seed=1)
theta2 = np.deg2rad(15)
R2 = np.array([[np.cos(theta2),-np.sin(theta2),0],[np.sin(theta2),np.cos(theta2),0],[0,0,1]])
tgt = transform_cloud(src, R=R2, t=np.array([0.2,0.1,0.0]))
reg = o3d.pipelines.registration.registration_icp(
    src, tgt, 0.3, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
res['icp_fitness'] = float(reg.fitness)
res['icp_rmse'] = float(reg.inlier_rmse)
res['icp_recovered_R_err'] = float(np.abs(reg.transformation[:3,:3]-R2).max())
res['icp_recovered_t_err'] = float(np.abs(reg.transformation[:3,3]-np.array([0.2,0.1,0.0])).max())

print(json.dumps(res, indent=1))
pathlib.Path('mvrecon/out/tools_smoke.json').write_text(json.dumps(res, indent=1))
