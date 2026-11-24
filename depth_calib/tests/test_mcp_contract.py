"""MCP protocol-contract tests (handoff 5.9 acceptance list)."""
import base64
import io
import json
import sys
import pathlib

import jsonschema
import numpy as np
import pytest
import yaml
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.mcp import handlers
from src.mcp.session import Session

CFG = load_config(ROOT / "configs" / "pilot.yaml")
SCHEMAS = json.loads((ROOT / "schemas" / "tools.json").read_text(encoding="utf-8"))


@pytest.fixture()
def session(tmp_path):
    img = (np.random.rand(120, 160, 3) * 255).astype(np.uint8)
    Image.fromarray(img).save(tmp_path / "img.png")
    ctx = {"image_id": "s1", "image_path": str(tmp_path / "img.png"),
           "width": 160, "height": 120, "output_dir": str(tmp_path / "out"),
           "run_id": "t", "mode": "harness"}
    return Session(ctx, CFG)


def anchor(**kw):
    base = {"anchor_id": "a1", "object_group_id": "o1", "knowledge_group_id": "g1",
            "description": "d", "p1": {"xy": [10.0, 10.0], "point_id": None},
            "p2": {"xy": [50.0, 10.0], "point_id": None}, "length_m": 1.0,
            "length_interval_m": None, "prior_kind": "identified_standard",
            "visible_evidence": "v", "identity_uncertainty": "u"}
    base.update(kw)
    return base


def submit(session, anchors, status="ok", reason=None):
    return handlers.submit_measurements(
        {"image_id": "s1", "status": status, "anchors": anchors,
         "abstain_reason": reason}, session)


def test_schema_file_is_valid():
    for t in SCHEMAS["tools"]:
        jsonschema.Draft202012Validator.check_schema(t["inputSchema"])


def test_four_tools_and_f0_subset():
    names = {t["name"] for t in SCHEMAS["tools"]}
    assert names == {"submit_measurements", "crop_image", "propose_points",
                     "render_measurements"}


def test_direct_coordinates_accepted(session):
    st, rec = submit(session, [anchor()])
    assert st == "ok" and rec["accepted"] and rec["session_status"] == "committed"
    assert (session.out_dir / "anchors" / "s1.json").exists()


def test_point_id_and_mixed_endpoints(session):
    session.register_points([[20.0, 30.0], [80.0, 30.0]], "v0")
    st, rec = submit(session, [anchor(p1={"xy": None, "point_id": "v0:p0"},
                                      p2={"xy": [80.0, 30.0], "point_id": None})])
    assert st == "ok", rec


def test_unknown_point_id_rejected(session):
    st, rec = submit(session, [anchor(p1={"xy": None, "point_id": "v9:p9"})])
    assert st == "error"
    assert rec["violations"][0]["code"] == "POINT_ID_UNKNOWN"


def test_out_of_bounds_rejected(session):
    st, rec = submit(session, [anchor(p2={"xy": [999.0, 10.0], "point_id": None})])
    assert st == "error" and rec["violations"][0]["code"] == "POINT_OUT_OF_BOUNDS"


def test_both_endpoint_forms_rejected(session):
    st, rec = submit(session, [anchor(p1={"xy": [1.0, 1.0], "point_id": "v0:p0"})])
    assert st == "error" and rec["violations"][0]["code"] == "ENDPOINT_AMBIGUOUS"


def test_neither_endpoint_form_rejected(session):
    st, rec = submit(session, [anchor(p1={"xy": None, "point_id": None})])
    assert st == "error" and rec["violations"][0]["code"] == "ENDPOINT_AMBIGUOUS"


def test_duplicate_anchor_id_rejected(session):
    st, rec = submit(session, [anchor(), anchor(p1={"xy": [11.0, 11.0], "point_id": None})])
    assert st == "error" and any(v["code"] == "DUPLICATE_ANCHOR_ID"
                                 for v in rec["violations"])


def test_nonpositive_length_and_bad_interval_rejected(session):
    st, rec = submit(session, [anchor(length_m=0.0)])
    assert st == "error"
    st, rec = submit(session, [anchor(length_m=1.0, length_interval_m=[2.0, 3.0])])
    assert st == "error" and rec["violations"][0]["code"] == "INTERVAL_INVALID"


def test_nan_rejected(session):
    st, rec = submit(session, [anchor(length_m=float("nan"))])
    assert st == "error"


def test_abstain_legal_and_stored(session):
    st, rec = submit(session, [], status="abstain", reason="no reliable evidence")
    assert st == "ok" and rec["accepted"]
    rec2 = json.loads((session.out_dir / "anchors" / "s1.json").read_text(encoding="utf-8"))
    assert rec2["status"] == "abstain" and rec2["anchors"] == []


def test_abstain_with_anchors_rejected(session):
    st, rec = submit(session, [anchor()], status="abstain", reason="x")
    assert st == "error" and rec["violations"][0]["code"] == "ABSTAIN_HAS_ANCHORS"


def test_idempotent_replay_and_conflict(session):
    args = {"image_id": "s1", "status": "ok", "anchors": [anchor()],
            "abstain_reason": None}
    st1, rec1 = handlers.submit_measurements(args, session)
    st2, rec2 = handlers.submit_measurements(args, session)
    assert st1 == st2 == "ok" and rec1["submission_id"] == rec2["submission_id"]
    st3, rec3 = handlers.submit_measurements(
        {**args, "anchors": [anchor(p1={"xy": [12.0, 12.0], "point_id": None})]},
        session)
    assert st3 == "error" and rec3["violations"][0]["code"] == "ALREADY_FINALIZED"


def test_submission_attempt_budget(session):
    bad = [anchor(p1={"xy": None, "point_id": "nope"})]
    st1, _ = submit(session, bad)
    st2, _ = submit(session, bad)
    st3, rec3 = submit(session, [anchor()])
    assert (st1, st2) == ("error", "error")
    assert st3 == "error"
    assert rec3["violations"][0]["code"] == "SUBMISSION_BUDGET_EXHAUSTED"


def test_visual_budget_enforced(session):
    for _ in range(CFG["budget"]["visual_calls_max"]):
        ok, code = session.consume_visual(is_crop=False)
        assert ok and code is None
    ok, code = session.consume_visual(is_crop=False)
    assert not ok and code == "VISUAL_BUDGET_EXHAUSTED"


def test_crop_budget_enforced(session):
    for _ in range(CFG["budget"]["crops_max"]):
        assert session.consume_visual(is_crop=True)[0]
    ok, code = session.consume_visual(is_crop=True)
    assert not ok and code == "CROP_BUDGET_EXHAUSTED"


def test_crop_returns_real_image(session):
    st, sc, imgs = handlers.crop_image(
        {"image_id": "s1", "bbox_xyxy": [10, 10, 60, 50]}, session)
    assert st == "ok"
    raw = base64.b64decode(imgs[0][1])
    im = Image.open(io.BytesIO(raw))
    assert im.size == (50, 40)
    assert sc["view_to_original"] == [[1, 0, 10], [0, 1, 10], [0, 0, 1]]


def test_render_does_not_submit(session):
    session.register_points([[20.0, 30.0], [80.0, 30.0]], "v0")
    st, sc, imgs = handlers.render_measurements(
        {"image_id": "s1", "pairs": [{"anchor_id": "a1",
                                     "p1": {"xy": None, "point_id": "v0:p0"},
                                     "p2": {"xy": None, "point_id": "v0:p1"}}]},
        session)
    assert st == "ok" and len(sc["resolved_pairs"]) == 1
    assert not (session.out_dir / "anchors" / "s1.json").exists()


def test_points_condition_requires_point_ids(tmp_path):
    from src.mcp.session import Session as S
    img = (np.random.rand(60, 80, 3) * 255).astype(np.uint8)
    Image.fromarray(img).save(tmp_path / "img.png")
    ctx = {"image_id": "p1", "image_path": str(tmp_path / "img.png"),
           "width": 80, "height": 60, "output_dir": str(tmp_path / "out"),
           "run_id": "t", "mode": "harness", "toolset": "points",
           "require_point_ids": True}
    s = S(ctx, CFG)
    payload = {"image_id": "p1", "status": "ok",
               "anchors": [anchor(p1={"xy": [10.0, 10.0], "point_id": None},
                                  p2={"xy": [50.0, 10.0], "point_id": None})],
               "abstain_reason": None}
    st, rec = handlers.submit_measurements(payload, s)
    assert st == "error" and rec["violations"][0]["code"] == "POINT_ID_REQUIRED"


def test_direct_coords_condition_hides_points():
    from src.mcp.server import _tool_list
    names = {t.name for t in _tool_list("harness", "direct_coords")}
    assert names == {"submit_measurements", "crop_image", "render_measurements"}
    names2 = {t.name for t in _tool_list("harness", "points")}
    assert "propose_points" in names2


def test_points_survive_reload(tmp_path):
    """Regression: registered point IDs must persist across CLI processes."""
    from src.mcp.session import Session as S, load_context
    import json as _json
    img = (np.random.rand(90, 120, 3) * 255).astype(np.uint8)
    Image.fromarray(img).save(tmp_path / "img.png")
    ctx = {"image_id": "persist1", "image_path": str(tmp_path / "img.png"),
           "width": 120, "height": 90, "output_dir": str(tmp_path / "out"),
           "run_id": "t", "mode": "harness", "toolset": "points",
           "require_point_ids": True}
    s1 = S(ctx, CFG)
    st, sc, _ = handlers.propose_points(
        {"image_id": "persist1", "roi_xyxy": None, "max_points": 8}, s1)
    assert st == "ok" and sc["points"], "expected candidate points"
    first_ids = [p["point_id"] for p in sc["points"]]

    ss = S(ctx, CFG)                     # simulates the next CLI invocation
    assert set(first_ids) <= set(ss.point_registry), "registry lost across processes"
    p1 = {"xy": None, "point_id": first_ids[0]}
    p2 = first_ids[1] if len(first_ids) > 1 else first_ids[0]
    if p2 == first_ids[0]:
        pytest.skip("only one candidate point found")
    st2, rec = handlers.submit_measurements(
        {"image_id": "persist1", "status": "ok",
         "anchors": [anchor(p1=p1, p2={"xy": None, "point_id": p2})],
         "abstain_reason": None}, ss)
    assert st2 == "ok", rec
