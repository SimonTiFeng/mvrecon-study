"""Does the estimate depend on how landmarks are filtered? Test error thresholds."""
import numpy as np, json, pathlib, cv2
ROOT = pathlib.Path("."); C = ROOT/"cache/hpc/depths"
CAL = pathlib.Path("E:/depth_calib_data/eth3d_extracted/courtyard/dslr_calibration_jpg")
rows = {json.loads(l)["sample_id"]: json.loads(l)
        for l in open("manifests/formal_all.jsonl") if l.strip()}

def q2r(q):
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],
                     [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
                     [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]])
cams={}
for l in open(CAL/"cameras.txt"):
    if l.startswith("#") or not l.strip(): continue
    p=l.split(); cams[p[0]]=(np.array([[float(p[4]),0,float(p[6])],[0,float(p[5]),float(p[7])],[0,0,1]]),int(p[2]),int(p[3]))
imgs={}; byid={}
L=[l for l in open(CAL/"images.txt") if l.strip() and not l.startswith("#")]
for i in range(0,len(L),2):
    p=L[i].split()
    if len(p)!=10: continue
    iid,qw,qx,qy,qz,tx,ty,tz,cid,name=p
    R=q2r(np.array([float(qw),float(qx),float(qy),float(qz)])); t=np.array([float(tx),float(ty),float(tz)])
    imgs[pathlib.Path(name).name]=(R,t)+cams[cid]; byid[int(iid)]=pathlib.Path(name).name
pts=[]
for l in open(CAL/"points3D.txt"):
    if l.startswith("#") or not l.strip(): continue
    p=l.split()
    xyz=np.array([float(p[1]),float(p[2]),float(p[3])]); err=float(p[7])
    tr=[(int(p[8+2*j]),int(p[9+2*j])) for j in range((len(p)-8)//2)]
    pts.append((xyz,err,tr))

print("=== Sensitivity to the landmark reprojection-error threshold ===")
print(f"{'err<':<6}{'imgs':>6}{'median s':>10}{'sd':>8}")
for thr in (0.5, 1.0, 2.0, 4.0, 8.0):
    ss=[]
    for sid,r in rows.items():
        if r["dataset"]!="eth3d" or r.get("scene_id")!="courtyard": continue
        src=pathlib.Path(r["source_image"]).name
        if src not in imgs: continue
        npz=C/f"{sid}__B.npz"
        if not npz.exists(): continue
        iid=[k for k,v in byid.items() if v==src][0]
        R,t,K,W,H=imgs[src]
        d=np.load(npz)["depth"].astype(np.float64)
        sd=json.loads((C/f"{sid}__B.json").read_text()); Hp,Wp=sd["processed_shape"]
        f=((K[0,0]*Wp/W)+(K[1,1]*Hp/H))/2/300.0
        # NOTE: no focal conversion -> compare against SfM directly at native scale
        d2=cv2.resize(d,(W,H))
        rec=[]
        for xyz,err,tr in pts:
            if err>thr: continue
            if not any(t2[0]==iid for t2 in tr): continue
            Xc=R@xyz+t
            z_=float(Xc[2])
            if z_<=0.05 or z_>60: continue
            u=K[0,0]*Xc[0]/Xc[2]+K[0,2]; v=K[1,1]*Xc[1]/Xc[2]+K[1,2]
            iu,iv=int(round(u)),int(round(v))
            if not(2<=iu<W-2 and 2<=iv<H-2): continue
            zm=float(np.median(d2[iv-1:iv+2,iu-1:iu+2]))
            if zm>0 and np.isfinite(zm): rec.append(np.log(z_/zm))
        if len(rec)>=100: ss.append(float(np.exp(np.median(rec))))
    if ss:
        print(f"{thr:<6}{len(ss):>6}{np.median(ss):>10.4f}{np.std(ss):>8.4f}")
