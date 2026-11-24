"""Does UniDepth improve on KITTI when given the TRUE intrinsics instead of guessing?"""
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
    if ":" in l: k, v = l.split(":", 1); c[k.strip()] = v.split()
Ktrue = np.array([float(x) for x in c["P_rect_02"]], float).reshape(3, 4)[:, :3]

def ab(z, g): return float(np.mean(np.abs(z - g) / g))

frames = ["0000000005","0000000011","0000000017"]
print(f"{'frame':<13}{'mode':<12}{'raw':>9}{'fx':>8}{'px':>8}")
for fr in frames:
    gp = glob.glob(f"{GTD}/2011_09_26_drive_0002_sync_groundtruth_depth_{fr}_image_02.png")
    if not gp: continue
    gt = cv2.imread(gp[0], cv2.IMREAD_UNCHANGED).astype(np.float64)/256.0
    img = cv2.cvtColor(cv2.imread(str(KIT/"image_02"/"data"/f"{fr}.png")), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()/255.0
    for mode in ("guessed", "true_K"):
        cam = None
        if mode == "true_K":
            cam = torch.from_numpy(Ktrue).float().unsqueeze(0).cuda()
        with torch.no_grad():
            out = m.infer(t.cuda(), camera=cam)
        d = out['depth'][0,0].cpu().numpy().astype(np.float64)
        d = cv2.resize(d, (1242, 375))
        fx = float(out['intrinsics'][0,0,0])
        sub = d[DV:DV+352, DU:DU+1216]
        msk = (gt>0)&np.isfinite(sub)&(sub>0)
        if msk.sum()<2000: continue
        p,g = sub[msk], gt[msk]
        print(f"{fr:<13}{mode:<12}{ab(p,g):>9.4f}{fx:>8.1f}{int(msk.sum()):>8}")
