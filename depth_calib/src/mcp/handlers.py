"""Tool handlers: pure server-side validation, image tools, atomic submit.

No handler reads ground truth, depth predictions, evaluation scores or the
network. The submit handler is the only evidence entry point (handoff 5.5).
"""
from __future__ import annotations

import base64
import io
import json
import math
import pathlib

import cv2
import jsonschema
import numpy as np
from PIL import Image, ImageDraw

from .session import Session
from .store import canonical_hash

SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[2] / "schemas" / "tools.json"
_TOOLS = {t["name"]: t for t in json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))["tools"]}


# ---------------------------------------------------------------- helpers
def violation(code: str, path: str, message: str) -> dict:
    return {"code": code, "path": path, "message": message}


def receipt(accepted: bool, submission_id=None, anchor_count=0,
            session_status="open", violations=None, warnings=None) -> dict:
    return {"accepted": accepted, "submission_id": submission_id,
            "anchor_count": anchor_count, "session_status": session_status,
            "violations": violations or [], "warnings": warnings or []}


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _all_finite(node, path="$"):
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            out += _all_finite(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out += _all_finite(v, f"{path}[{i}]")
    elif isinstance(node, float) and not math.isfinite(node):
        out.append(path)
    return out


def _schema_errors(arguments: dict, tool: str) -> list[dict]:
    errs = []
    validator = jsonschema.Draft202012Validator(_TOOLS[tool]["inputSchema"])
    for e in sorted(validator.iter_errors(arguments), key=lambda e: list(e.absolute_path)):
        p = "$" + "".join(f"[{i}]" if isinstance(i, int) else f".{i}"
                          for i in e.absolute_path)
        errs.append(violation("SCHEMA_ERROR", p, e.message))
    for p in _all_finite(arguments):
        errs.append(violation("NONFINITE_NUMBER", p, "NaN/Infinity not allowed"))
    return errs


def _load_rgb(session: Session) -> Image.Image:
    img = Image.open(session.image_path)
    img = img.convert("RGB")
    if img.size != (session.width, session.height):
        raise RuntimeError(
            f"image size {img.size} != context ({session.width},{session.height})")
    return img


def _resolve_endpoint(ep: dict, session: Session, path: str,
                      violations: list[dict]) -> np.ndarray | None:
    xy, pid = ep.get("xy"), ep.get("point_id")
    if getattr(session, "require_point_ids", False):
        if pid is None:
            violations.append(violation(
                "POINT_ID_REQUIRED", path,
                "this condition only accepts candidate point_id endpoints"))
            return None
        if xy is not None:
            violations.append(violation(
                "ENDPOINT_AMBIGUOUS", path,
                "point_id conditions require xy=null"))
            return None
    if (xy is None) == (pid is None):
        violations.append(violation(
            "ENDPOINT_AMBIGUOUS", path,
            "exactly one of xy / point_id must be non-null"))
        return None
    if pid is not None:
        if pid not in session.point_registry:
            violations.append(violation(
                "POINT_ID_UNKNOWN", f"{path}.point_id",
                f"point_id {pid!r} is not registered for this sample"))
            return None
        return np.asarray(session.point_registry[pid], dtype=float)
    p = np.asarray(xy, dtype=float)
    if not (0.0 <= p[0] <= session.width - 1 and 0.0 <= p[1] <= session.height - 1):
        violations.append(violation(
            "POINT_OUT_OF_BOUNDS", f"{path}.xy",
            f"xy {p.tolist()} outside [0,{session.width - 1}]x[0,{session.height - 1}]"))
        return None
    return p


# ---------------------------------------------------------------- handlers
def crop_image(arguments: dict, session: Session) -> tuple[str, dict, list]:
    errs = _schema_errors(arguments, "crop_image")
    if arguments.get("image_id") != session.image_id:
        errs.append(violation("IMAGE_ID_MISMATCH", "$.image_id",
                              "image_id does not match the bound sample"))
    if errs:
        return "error", receipt(False, session_status="committed" if session.committed else "open",
                                violations=errs), []
    img = _load_rgb(session)
    x0, y0, x1, y1 = arguments["bbox_xyxy"]
    if not (0 <= x0 < x1 <= session.width and 0 <= y0 < y1 <= session.height):
        return "error", receipt(False, violations=[violation(
            "BBOX_INVALID", "$.bbox_xyxy",
            "require 0<=x0<x1<=W and 0<=y0<y1<=H (half-open)")]), []
    view_id = session.next_view_id()
    crop = img.crop((x0, y0, x1, y1))
    view_path = session.out_dir / "views" / f"{view_id}.png"
    view_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(view_path)
    sc = {
        "image_id": session.image_id,
        "view_id": view_id,
        "original_size": [session.width, session.height],
        "bbox_xyxy": [x0, y0, x1, y1],
        "crop_size": [x1 - x0, y1 - y0],
        "view_to_original": [[1, 0, x0], [0, 1, y0], [0, 0, 1]],
        "remaining_visual_calls": session.remaining_visual(),
    }
    return "ok", sc, [("image", _png_b64(crop))]


def propose_points(arguments: dict, session: Session) -> tuple[str, dict, list]:
    errs = _schema_errors(arguments, "propose_points")
    if arguments.get("image_id") != session.image_id:
        errs.append(violation("IMAGE_ID_MISMATCH", "$.image_id",
                              "image_id does not match the bound sample"))
    if errs:
        return "error", receipt(False, violations=errs), []
    roi = arguments["roi_xyxy"]
    if roi is None:
        x0, y0, x1, y1 = 0, 0, session.width, session.height
    else:
        x0, y0, x1, y1 = roi
        if not (0 <= x0 < x1 <= session.width and 0 <= y0 < y1 <= session.height):
            return "error", receipt(False, violations=[violation(
                "ROI_INVALID", "$.roi_xyxy", "invalid half-open ROI")]), []
    key = ((x0, y0, x1, y1), int(arguments["max_points"]))
    cached = session.cached_points(key)
    if cached:                      # identical config: reuse IDs, still counts as a call
        pts = [{"point_id": k, "xy_original": v}
               for k, v in session.point_registry.items() if k.startswith(cached + ":")]
        img = _load_rgb(session)
        marked = _draw_points(img, [(p["xy_original"], p["point_id"].split(":")[1]) for p in pts])
        return "ok", {"image_id": session.image_id, "view_id": cached,
                      "points": pts, "implementation": "cv2_goodFeaturesToTrack",
                      "note": "traditional-corner fallback; not a neural feature method",
                      "view_to_original": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      "remaining_visual_calls": session.remaining_visual()}, \
            [("image", _png_b64(marked))]

    img = _load_rgb(session)
    sub = np.asarray(img.crop((x0, y0, x1, y1)).convert("L"))
    max_pts = int(arguments["max_points"])
    corners = cv2.goodFeaturesToTrack(
        sub, maxCorners=max_pts, qualityLevel=0.01,
        minDistance=max(3.0, 0.03 * min(sub.shape[:2])), blockSize=7)
    pts_orig = [] if corners is None else [
        [float(c[0][0]) + x0, float(c[0][1]) + y0] for c in corners[:max_pts]]
    view_id = session.next_view_id()
    session.cache_points(key, view_id)
    reg = session.register_points(pts_orig, view_id)
    session.persist_usage()      # registry must survive into later CLI processes
    view_path = session.out_dir / "views" / f"{view_id}.png"
    view_path.parent.mkdir(parents=True, exist_ok=True)
    marked = _draw_points(img, [(p["xy_original"], p["point_id"].split(":")[1]) for p in reg])
    marked.save(view_path)
    sc = {"image_id": session.image_id, "view_id": view_id, "points": reg,
          "implementation": "cv2_goodFeaturesToTrack",
          "note": "traditional-corner fallback; not a neural feature method",
          "view_to_original": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
          "remaining_visual_calls": session.remaining_visual()}
    return "ok", sc, [("image", _png_b64(marked))]


def _draw_points(img: Image.Image, pts: list[tuple[list[float], str]]) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    for xy, label in pts:
        x, y = xy
        d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(255, 40, 40), width=2)
        d.text((x + 6, max(0, y - 12)), label, fill=(255, 255, 0))
    return out


def render_measurements(arguments: dict, session: Session) -> tuple[str, dict, list]:
    errs = _schema_errors(arguments, "render_measurements")
    if arguments.get("image_id") != session.image_id:
        errs.append(violation("IMAGE_ID_MISMATCH", "$.image_id",
                              "image_id does not match the bound sample"))
    if errs:
        return "error", receipt(False, violations=errs), []
    resolved = []
    for i, pair in enumerate(arguments["pairs"]):
        p1 = _resolve_endpoint(pair["p1"], session, f"$.pairs[{i}].p1", errs)
        p2 = _resolve_endpoint(pair["p2"], session, f"$.pairs[{i}].p2", errs)
        if p1 is not None and p2 is not None:
            if np.allclose(p1, p2):
                errs.append(violation("ENDPOINTS_COINCIDE", f"$.pairs[{i}]",
                                      "p1 and p2 resolve to the same point"))
            resolved.append({"anchor_id": pair["anchor_id"],
                             "p1_xy": p1.tolist(), "p2_xy": p2.tolist()})
    if errs:
        return "error", receipt(False, violations=errs), []
    img = _load_rgb(session)
    out = img.copy()
    d = ImageDraw.Draw(out)
    for r in resolved:
        d.line([tuple(r["p1_xy"]), tuple(r["p2_xy"])], fill=(0, 255, 0), width=3)
        for xy in (r["p1_xy"], r["p2_xy"]):
            d.ellipse([xy[0] - 4, xy[1] - 4, xy[0] + 4, xy[1] + 4],
                      outline=(255, 40, 40), width=2)
        d.text((r["p1_xy"][0] + 5, r["p1_xy"][1] - 12), r["anchor_id"], fill=(255, 255, 0))
    view_id = session.next_view_id()
    view_path = session.out_dir / "views" / f"{view_id}.png"
    view_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(view_path)
    sc = {"image_id": session.image_id, "view_id": view_id,
          "resolved_pairs": resolved,
          "remaining_visual_calls": session.remaining_visual()}
    return "ok", sc, [("image", _png_b64(out))]


def validate_submission(arguments: dict, session: Session) -> tuple[list[dict], dict | None, list[str]]:
    errs = _schema_errors(arguments, "submit_measurements")
    warnings: list[str] = []
    if arguments.get("image_id") != session.image_id:
        errs.append(violation("IMAGE_ID_MISMATCH", "$.image_id",
                              "image_id does not match the bound sample"))
    if errs:
        return errs, None, warnings

    status = arguments["status"]
    anchors = arguments["anchors"]
    reason = arguments["abstain_reason"]
    if status == "ok":
        if not (1 <= len(anchors) <= 6):
            errs.append(violation("ANCHOR_COUNT", "$.anchors",
                                  "status=ok requires 1..6 anchors"))
        if reason is not None:
            errs.append(violation("ABSTAIN_REASON_SET", "$.abstain_reason",
                                  "status=ok requires abstain_reason=null"))
    else:
        if anchors:
            errs.append(violation("ABSTAIN_HAS_ANCHORS", "$.anchors",
                                  "status=abstain requires an empty anchors array"))
        if not (isinstance(reason, str) and reason.strip()):
            errs.append(violation("ABSTAIN_REASON_MISSING", "$.abstain_reason",
                                  "status=abstain requires a non-empty reason"))
    if errs:
        return errs, None, warnings

    seen = set()
    norm_anchors = []
    for i, a in enumerate(anchors):
        aid = a["anchor_id"]
        if aid in seen:
            errs.append(violation("DUPLICATE_ANCHOR_ID", f"$.anchors[{i}].anchor_id",
                                  f"anchor_id {aid!r} repeated"))
        seen.add(aid)
        lo, hi = a.get("length_interval_m") or (None, None)
        if lo is not None:
            if not (0 < lo <= a["length_m"] <= hi):
                errs.append(violation("INTERVAL_INVALID", f"$.anchors[{i}].length_interval_m",
                                      "require 0 < low <= length_m <= high"))
        p1 = _resolve_endpoint(a["p1"], session, f"$.anchors[{i}].p1", errs)
        p2 = _resolve_endpoint(a["p2"], session, f"$.anchors[{i}].p2", errs)
        if p1 is not None and p2 is not None and np.allclose(p1, p2):
            errs.append(violation("ENDPOINTS_COINCIDE", f"$.anchors[{i}]",
                                  "p1 and p2 resolve to the same point"))
        if p1 is not None and p2 is not None:
            norm_anchors.append({
                "anchor_id": aid,
                "object_group_id": a["object_group_id"],
                "knowledge_group_id": a["knowledge_group_id"],
                "description": a["description"],
                "p1_xy": p1.tolist(), "p2_xy": p2.tolist(),
                "source_point_ids": [a["p1"]["point_id"], a["p2"]["point_id"]],
                "length_m": float(a["length_m"]),
                "length_interval_m": list(lo_hi) if (lo_hi := a.get("length_interval_m")) else None,
                "prior_kind": a["prior_kind"],
                "visible_evidence": a["visible_evidence"],
                "identity_uncertainty": a["identity_uncertainty"],
            })
    if errs:
        return errs, None, warnings

    normalized = {
        "image_id": session.image_id,
        "image_width": session.width,
        "image_height": session.height,
        "status": status,
        "anchors": norm_anchors,
        "abstain_reason": reason,
    }
    return [], normalized, warnings


def submit_measurements(arguments: dict, session: Session) -> tuple[str, dict]:
    req_hash = canonical_hash(arguments)
    if session.committed:
        if req_hash == session.request_hash:
            state = session.store.load_state() or {}
            return "ok", receipt(True, state.get("submission_id"),
                                 state.get("anchor_count", 0), "committed")
        return "error", receipt(False, session.submission_id, 0, "committed",
                                [violation("ALREADY_FINALIZED", "$",
                                           "sample already finalized; payload differs")])

    if session.submission_attempts >= session.attempts_max:
        return "error", receipt(False, violations=[violation(
            "SUBMISSION_BUDGET_EXHAUSTED", "$",
            f"max {session.attempts_max} submission attempts reached")])
    session.submission_attempts += 1
    session.persist_usage()

    errs, normalized, warnings = validate_submission(arguments, session)
    if errs:
        return "error", receipt(False, violations=errs, warnings=warnings)

    state = session.store.commit(
        request_hash=req_hash, raw_arguments=arguments, normalized=normalized,
        trusted_metadata={"image_id": session.image_id,
                          "image_path": session.image_path,
                          "image_width": session.width,
                          "image_height": session.height,
                          "run_id": session.context.get("run_id"),
                          "mode": session.mode})
    session.committed = True
    session.submission_id = state["submission_id"]
    session.request_hash = req_hash
    return "ok", receipt(True, state["submission_id"], state["anchor_count"],
                         "committed", warnings=warnings)
