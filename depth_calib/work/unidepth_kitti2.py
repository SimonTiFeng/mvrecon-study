"""Corrected cross-model test: UniDepth with TRUE intrinsics + our calibration."""
import json, torch, pathlib, numpy as np, cv2, glob
from safetensors.torch import load_file
from unidepth.models import UniDepthV2
from omegaconf import DictConfig

D = pathlib.Path('D:/models/UNIDEPTH-V2-L')
m = UniDepthV2(DictConfig(json.loads((D/'config.json').read_text())))
m.load_state_dict(load_file(str(D/'model.safetensors')), strict=False)
m = m.to('cuda').eval()

KIT = pathlib.Path("D:/kitti/2011_09_26/2011_09_26_drive_0002_sync")
GTD = "D:/kitti/sel/depth_selection/val_selection_cropped/groundtruth_depth"
DU, DV = 7, 23
c = {}
for l in pathlib.Path("D:/kitti/2011_09_26/calib_cam_to_cam.txt").read_text().splitlines():
    if ":" in l: k,v=l.split(":",1); c[k.strip()]=v.split()
v2 = {}
for l in pathlib.Path("D:/kitti/2011_09_26/calib_velo_to_cam.txt").read_text().splitlines():
    if ":" in l: k,v=l.split(":",1); v2[k.strip()]=v.split()
P = np.array([float(x) for x in c["P_rect_02"]],float).reshape(3,4)
Kt = P[:,:3]
Rv = np.array([float(x) for x in v2["R"]],float).reshape(3,3)
Tv = np.array([float(x) for x in v2["T"]],float)
R0 = np.array([float(x) for x in c["R_rect_00"]],float).reshape(3,3)

def ab(z,g): return float(np.mean(np.abs(z-g)/g))
frames = ["0000000005","0000000011","0000000017","0000000023","0000000029","0000000035",
          "0000000041","0000000047","0000000053","0000000059","0000000065","0000000071"]
print(f"{'frame':<13}{'raw':>9}{'s_lidar':>9}{'cal':>9}{'s_gt':>8}{'wins':>6}")
R=[]
for fr in frames:
    gp = glob.glob(f"{GTD}/2011_09_26_drive_0002_sync_groundtruth_depth_{fr}_image_02.png")
    if not gp: continue
    gt = cv2.imread(gp[0],cv2.IMREAD_UNCHANGED).astype(np.float64)/256.0
    img = cv2.cvtColor(cv2.imread(str(KIT/"image_02"/"data"/f"{fr}.png")),cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()/255.0
    with torch.no_grad():
        out = m.infer(t.cuda(), camera=torch.from_numpy(Kt).float().unsqueeze(0).cuda())
    d = cv2.resize(out['depth'][0,0].cpu().numpy().astype(np.float64),(1242,375))
    sub = d[DV:DV+352, DU:DU+1216]
    msk = (gt>0)&np.isfinite(sub)&(sub>0)
    if msk.sum()<2000: continue
    p,g = sub[msk],gt[msk]
    vel = np.fromfile(str(KIT/"velodyne_points"/"data"/f"{fr}.bin"),dtype=np.float32).reshape(-1,4)
    X = (vel[:,:3]@Rv.T+Tv)@R0.T; z=X[:,2]; X,z=X[z>1],z[z>1]
    u=P[0,0]*X[:,0]/z+P[0,2]; v=P[1,1]*X[:,1]/z+P[1,2]
    iu=np.round(u).astype(int)-DU; iv=np.round(v).astype(int)-DV
    inb=(iu>=0)&(iu<1216)&(iv>=0)&(iv<352)
    g2=gt[iv[inb],iu[inb]]; mz=sub[iv[inb],iu[inb]]
    ok=(g2>0)&np.isfinite(mz)&(mz>0)
    lr=np.log(z[inb][ok]/mz[ok]); med=np.median(lr)
    lr=lr[np.abs(lr-med)<3*1.4826*np.median(np.abs(lr-med))+1e-9]
    s_l=float(np.exp(np.median(lr)))
    grid=np.arange(0.4,3.0,0.005)
    s_gt=float(grid[int(np.argmin([ab(x*p,g) for x in grid]))])
    raw=ab(p,g); cal=ab(s_l*p,g)
    R.append((raw,cal,s_l,s_gt))
    print(f"{fr:<13}{raw:>9.4f}{s_l:>9.4f}{cal:>9.4f}{s_gt:>8.3f}{'Y' if cal<raw else 'N':>6}")
r=np.array(R)
print(f"\nUniDepth(true K): raw {np.median(r[:,0]):.4f} -> calibrated {np.median(r[:,1]):.4f}"
      f"   wins {int((r[:,1]<r[:,0]).sum())}/{len(r)}")
print(f"s_lidar/s_gt = {np.median(r[:,2]/r[:,3]):.4f}")
print(f"published UniDepth v2 KITTI = 0.076")
