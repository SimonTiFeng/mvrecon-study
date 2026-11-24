"""Endpoint-correction diagnostic on the fixed targets (evaluator-only).

Fix the object and the agent's claimed length; replace only the endpoints with the
verifier reference endpoints; recompute the B metric scale and its AbsRel.
"""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.geometry.camera import Camera, sample_depth_bilinear
from src.evaluation.metrics import evaluate_depth

ROOT = pathlib.Path(__file__).resolve().parents[1]
cfg = load_config(ROOT/"configs"/"pilot.yaml")
t = json.loads((ROOT/"results/exploratory/fixed_targets.json").read_text(encoding="utf-8"))
refs = t["reference_endpoints_verifier_only"]
rows = {json.loads(l)["sample_id"]: json.loads(l) for l in
        (ROOT/"manifests/explor_20.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
cache = ROOT/"cache/hpc/depths"
out = {}
for run in ("explor_points_001", "explor_direct_001"):
    out[run] = {}
    for sid, ref in refs.items():
        nf = ROOT/"results"/run/"samples"/sid/"mcp_out"/"anchors"/f"{sid}.json"
        fb = cache/f"{sid}__B.npz"
        if not nf.exists() or not fb.exists():
            continue
        norm = json.loads(nf.read_text(encoding="utf-8"))
        if norm.get("status") != "ok" or not norm.get("anchors"):
            out[run][sid] = {"status": norm.get("status")}
            continue
        row = rows[sid]
        gt = np.load(pathlib.Path(row["dataset_root"])/row["gt"]).astype(np.float64)
        mask = np.load(pathlib.Path(row["dataset_root"])/row["mask"]).astype(bool)
        dB = np.load(fb)["depth"].astype(np.float64)
        meta = json.loads((cache/f"{sid}__B.json").read_text(encoding="utf-8"))
        K = np.asarray(row["K"], float); Hp, Wp = meta["processed_shape"]
        fB = ((K[0,0]*Wp/row["width"]) + (K[1,1]*Hp/row["height"]))/2/300.0
        dB = dB*fB
        cam = Camera(K[0,0],K[1,1],K[0,2],K[1,2],row["width"],row["height"],
                     model=row.get("camera_model","PINHOLE"), params=row.get("camera_params"))
        ref_np = [np.asarray(r,float) for r in ref]
        dref, okref = sample_depth_bilinear(dB, np.stack(ref_np))
        if not okref.all():
            out[run][sid] = {"status":"ref_depth_invalid"}; continue
        L_model_ref = float(np.linalg.norm(dref[0]*cam.q(ref_np[0])[0] - dref[1]*cam.q(ref_np[1])[0]))
        base_abs = evaluate_depth(dB, gt, mask)["abs_rel"]
        rows_out = []
        for a in norm["anchors"]:
            s = a["length_m"]/L_model_ref if L_model_ref>1e-9 else 1.0
            rows_out.append({"anchor_id":a["anchor_id"],"claimed_length_m":a["length_m"],
                             "scale_at_ref":s,
                             "absrel_after":evaluate_depth(dB*s, gt, mask)["abs_rel"]})
        out[run][sid] = {"status":"ok","baseline_absrel":base_abs,"anchors":rows_out}
path = ROOT/"results/exploratory/step4_endpoint_fix.json"
path.write_text(json.dumps(out,indent=1),encoding="utf-8")
print(json.dumps(out,indent=1)[:3000])
