"""Verify the per-image clouds are sane, and measure how much they overlap in the reference frame."""
import json, pathlib, numpy as np, cv2

AG = pathlib.Path('mvrecon/agent_data'); EV = pathlib.Path('mvrecon/eval_data')

def cloud_own_frame(sid, meta):
    d = np.load(AG/meta['scene']/f'{sid}_depth.npy').astype(np.float64)
    K = np.asarray(meta['K'], float); H, W = d.shape
    ys, xs = np.mgrid[0:H:24, 0:W:24]
    z = d[ys, xs]
    ok = np.isfinite(z) & (z > 0) & (z < 200)
    ys, xs, z = ys[ok], xs[ok], z[ok]
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    X = (xs-cx)/fx*z; Y = (ys-cy)/fy*z
    return np.stack([X, Y, z], 1)

print(f"{'scene':<15}{'img':<13}{'npts':>7}{'medZ':>7}{'z5':>7}{'z95':>7}")
for scene in sorted(p.name for p in AG.iterdir() if p.is_dir()):
    ag = json.loads((AG/scene/'meta.json').read_text())
    for im in ag['images']:
        p = cloud_own_frame(im['sid'], {'scene':scene, **im})
        if len(p)==0: continue
        print(f"{scene:<15}{im['sid']:<13}{len(p):>7}{np.median(p[:,2]):>7.2f}"
              f"{np.percentile(p[:,2],5):>7.2f}{np.percentile(p[:,2],95):>7.2f}")
    break  # one scene for the table
print()
print('=== overlap in the reference frame (do the 4 views share structure?) ===')
for scene in sorted(p.name for p in AG.iterdir() if p.is_dir()):
    ag = json.loads((AG/scene/'meta.json').read_text())
    ev = json.loads((EV/scene/'meta.json').read_text())
    pose = {x['sid']: x for x in ev['images']}
    worlds = []
    for im in ag['images']:
        if im['sid'] not in pose: continue
        p = cloud_own_frame(im['sid'], {'scene':scene, **im})
        if len(p)==0: continue
        R = np.asarray(pose[im['sid']]['R']); t = np.asarray(pose[im['sid']]['t'])
        worlds.append(p @ R.T + t)   # cam->world
    if len(worlds)<2: continue
    # pairwise nearest-neighbour distance in world coords (sampled)
    from scipy.spatial import cKDTree
    pairs=[]
    for i in range(len(worlds)):
        for j in range(i+1,len(worlds)):
            A=worlds[i]; B=worlds[j]
            if len(A)<50 or len(B)<50: continue
            d,_=cKDTree(A).query(B,k=1)
            frac=float((d<0.35).mean())
            pairs.append(frac)
    print(f'{scene:<15} views={len(worlds)}  pairwise overlap (<0.35m): '
          f'{np.round(pairs,2) if pairs else "n/a"}')
