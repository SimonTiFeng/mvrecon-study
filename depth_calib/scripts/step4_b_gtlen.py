"""B-backend length-substitution diagnostic on the pilot anchors (evaluator-only)."""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.geometry.camera import Camera, sample_depth_bilinear
from src.geometry.metric_scale import fit_metric_scale, apply_metric_scale
from src.geometry.types import AnchorGeom
from src.evaluation.metrics import evaluate_depth
from src.evaluation.oracle import optimal_global_scale_absrel

ROOT = pathlib.Path(__file__).resolve().parents[1]
cfg = load_config(ROOT / "configs" / "pilot.yaml")
rows = [json.loads(l) for l in (ROOT/"manifests"/"pilot.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
cache = ROOT/"cache"/"hpc"/"depths"
out = []
for row in rows:
    sid = row["sample_id"]
    nf = ROOT/"results"/"pilot_f1_001"/"samples"/sid/"mcp_out"/"anchors"/f"{sid}.json"
    fb = cache/f"{sid}__B.npz"
    if not nf.exists() or not fb.exists(): continue
    norm = json.loads(nf.read_text(encoding="utf-8"))
    if norm.get("status")!="ok" or not norm.get("anchors"): continue
    gt = np.load(pathlib.Path(row["dataset_root"])/row["gt"]).astype(np.float64)
    mask = np.load(pathlib.Path(row["dataset_root"])/row["mask"]).astype(bool)
    dB = np.load(fb)["depth"].astype(np.float64)
    meta = json.loads((cache/f"{sid}__B.json").read_text(encoding="utf-8"))
    K = np.asarray(row["K"], float); Hp, Wp = meta["processed_shape"]
    fB = ((K[0,0]*Wp/row["width"]) + (K[1,1]*Hp/row["height"]))/2/300.0
    dB = dB*fB
    cam = Camera(K[0,0],K[1,1],K[0,2],K[1,2],row["width"],row["height"],
                 model=row.get("camera_model","PINHOLE"), params=row.get("camera_params"))
    def make(Ls):
        out=[]
        for a_,L in zip(norm["anchors"], Ls):
            p1=np.asarray(a_["p1_xy"],float); p2=np.asarray(a_["p2_xy"],float)
            dd,ok=sample_depth_bilinear(dB,np.stack([p1,p2]),eps=cfg["quality"]["depth_valid_eps"])
            g=AnchorGeom(a_["anchor_id"],a_["object_group_id"],a_["knowledge_group_id"],p1,p2,
                         cam.q(p1)[0],cam.q(p2)[0],float(L))
            if ok.all(): g.d1,g.d2=float(dd[0]),float(dd[1])
            out.append(g)
        return out
    def gtlen(a_):
        p1=np.asarray(a_["p1_xy"],float); p2=np.asarray(a_["p2_xy"],float)
        dd,ok=sample_depth_bilinear(gt,np.stack([p1,p2]),mask)
        if not ok.all(): return None
        return float(np.linalg.norm(dd[0]*cam.q(p1)[0]-dd[1]*cam.q(p2)[0]))
    Lg=[gtlen(a_) for a_ in norm["anchors"]]
    keep=[i for i,x in enumerate(Lg) if x is not None]
    Ls_gt=[Lg[i] for i in keep]
    anchors_actual=make([a["length_m"] for a in norm["anchors"]])
    anchors_gtL=make(Ls_gt) if keep else []
    # pair subsets consistently by taking kept anchors for actual too
    anchors_actual_keep=[anchors_actual[i] for i in keep]
    sol_a=fit_metric_scale(anchors_actual_keep,cfg)
    sol_g=fit_metric_scale(anchors_gtL,cfg)
    pred_a=apply_metric_scale(dB, sol_a.get("s_consistent", sol_a["s"]))
    pred_g=apply_metric_scale(dB, sol_g.get("s_consistent", sol_g["s"]))
    st_star=optimal_global_scale_absrel(dB,gt,mask)
    out.append({"sample_id":sid,"dataset":row["dataset"],"n_anchors":len(norm["anchors"]),
                "n_kept":len(keep),
                "B_actual_absrel":evaluate_depth(pred_a,gt,mask)["abs_rel"],
                "B_gtlen_absrel":evaluate_depth(pred_g,gt,mask)["abs_rel"],
                "B_oracle_absrel":evaluate_depth(dB*st_star,gt,mask)["abs_rel"],
                "status_actual":sol_a["status"],"status_gtlen":sol_g["status"]})
path=ROOT/"results"/"exploratory"/"step4_B_gtlen.json"
path.write_text(json.dumps(out,indent=1),encoding="utf-8")
import statistics as st
ok=[r for r in out if r["status_actual"]=="ok"]
def med(v): return float(np.median(v)) if v else float("nan")
print("B images",len(out),"ok",len(ok))
print("all: actual",round(med([r['B_actual_absrel'] for r in out]),4),"gtlen",round(med([r['B_gtlen_absrel'] for r in out]),4))
if ok: print("ok-subset: actual",round(med([r['B_actual_absrel'] for r in ok]),4),"gtlen",round(med([r['B_gtlen_absrel'] for r in ok]),4),"oracle",round(med([r['B_oracle_absrel'] for r in ok]),4))
