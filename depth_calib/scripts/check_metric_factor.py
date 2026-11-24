"""Empirical check of the DA3-Metric canonical -> metre conversion (handoff 7, pitfall 13).

Compares raw prediction medians against GT medians on a few NYUv2 images and
reports whether the focal/300 factor is needed exactly once.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.backends.da3 import DA3Backend

cfg = load_config(ROOT / "configs" / "pilot.yaml")
meta = json.loads(pathlib.Path("E:/depth_calib_data/processed/nyuv2/meta.json").read_text(encoding="utf-8"))
be = DA3Backend(cfg["backends"]["B"], cfg["backend_inference"]).load()
out = []
for idx in [0, 137, 402, 733, 1101]:
    e = meta["entries"][idx]
    img = pathlib.Path("E:/depth_calib_data/processed/nyuv2") / e["image"]
    gt = np.load(pathlib.Path("E:/depth_calib_data/processed/nyuv2") / e["gt"])
    mask = np.load(pathlib.Path("E:/depth_calib_data/processed/nyuv2") / e["mask"]).astype(bool)
    res = be.predict(str(img), (e["width"], e["height"]))
    pred = res["depth"]
    m = mask & np.isfinite(pred) & (pred > 0)
    K = np.asarray(e["K"], float)
    Hp, Wp = res["processed_shape"]
    fx_proc = K[0, 0] * Wp / e["width"]
    factor = fx_proc / 300.0
    ratio_raw = float(np.median(gt[m]) / np.median(pred[m]))
    ratio_after = float(np.median(gt[m]) / np.median(pred[m] * factor))
    out.append({"index": idx, "scene": e["scene_id"], "processed_shape": [Hp, Wp],
                "fx_proc": fx_proc, "focal_over_300": factor,
                "gt_over_raw": ratio_raw, "gt_over_raw_times_factor": ratio_after})
    print(out[-1], flush=True)
summary = {
    "median_gt_over_raw": float(np.median([r["gt_over_raw"] for r in out])),
    "median_gt_over_raw_times_factor": float(np.median([r["gt_over_raw_times_factor"] for r in out])),
    "median_focal_over_300": float(np.median([r["focal_over_300"] for r in out])),
    "decision": None,
}
# decision rule: whichever variant is closer to 1.0 in median ratio wins
if abs(summary["median_gt_over_raw"] - 1) < abs(summary["median_gt_over_raw_times_factor"] - 1):
    summary["decision"] = "raw_already_metric_no_conversion"
else:
    summary["decision"] = "apply_focal_over_300_once"
dest = ROOT / "results" / "metric_factor_check.json"
dest.write_text(json.dumps({"rows": out, "summary": summary}, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))
