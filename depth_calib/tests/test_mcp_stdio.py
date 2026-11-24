"""End-to-end MCP stdio test: spawn the real server, list tools, submit.

This is the transport contract Codex will use; it must pass before any VLM run.
"""
import asyncio
import json
import pathlib
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PY = sys.executable
SERVER = ROOT / "src" / "mcp" / "server.py"


def _run(tmp_path, tool_calls):
    img = (np.random.rand(90, 120, 3) * 255).astype(np.uint8)
    Image.fromarray(img).save(tmp_path / "img.png")
    ctx = {"image_id": "s1", "image_path": str(tmp_path / "img.png"),
           "width": 120, "height": 90, "output_dir": str(tmp_path / "out"),
           "run_id": "stdio", "mode": "harness"}
    ctx_path = tmp_path / "ctx.json"
    ctx_path.write_text(json.dumps(ctx))

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=PY, args=[str(SERVER), "--context", str(ctx_path),
                          "--config", str(ROOT / "configs" / "pilot.yaml")])

    async def go():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as sess:
                await sess.initialize()
                tools = await sess.list_tools()
                names = sorted(t.name for t in tools.tools)
                out = {"tools": names, "calls": []}
                for name, args in tool_calls:
                    res = await sess.call_tool(name, args)
                    out["calls"].append({"name": name, "isError": res.isError,
                                         "structured": res.structuredContent,
                                         "content_types": [c.type for c in res.content]})
                return out

    return asyncio.run(go())


def test_stdio_full_contract(tmp_path):
    anchor = {"anchor_id": "a1", "object_group_id": "o1",
              "knowledge_group_id": "g1", "description": "d",
              "p1": {"xy": [10.0, 12.0], "point_id": None},
              "p2": {"xy": [60.0, 12.0], "point_id": None},
              "length_m": 0.9, "length_interval_m": None,
              "prior_kind": "identified_standard",
              "visible_evidence": "v", "identity_uncertainty": "u"}
    calls = [
        ("crop_image", {"image_id": "s1", "bbox_xyxy": [5, 5, 65, 45]}),
        ("render_measurements", {"image_id": "s1", "pairs": [
            {"anchor_id": "a1", "p1": {"xy": [10.0, 12.0], "point_id": None},
             "p2": {"xy": [60.0, 12.0], "point_id": None}}]}),
        ("submit_measurements", {"image_id": "s1", "status": "ok",
                                 "anchors": [anchor], "abstain_reason": None}),
    ]
    out = _run(tmp_path, calls)
    assert out["tools"] == ["crop_image", "propose_points",
                            "render_measurements", "submit_measurements"]
    crop, render, sub = out["calls"]
    assert crop["isError"] is False and "image" in crop["content_types"]
    assert render["isError"] is False and "image" in render["content_types"]
    assert sub["isError"] is False and sub["structured"]["accepted"] is True
    assert (tmp_path / "out" / "anchors" / "s1.json").exists()


def test_stdio_direct_mode_hides_observation_tools(tmp_path):
    img = (np.random.rand(90, 120, 3) * 255).astype(np.uint8)
    Image.fromarray(img).save(tmp_path / "img.png")
    ctx = {"image_id": "s1", "image_path": str(tmp_path / "img.png"),
           "width": 120, "height": 90, "output_dir": str(tmp_path / "out"),
           "run_id": "stdio", "mode": "direct"}
    ctx_path = tmp_path / "ctx.json"
    ctx_path.write_text(json.dumps(ctx))

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    params = StdioServerParameters(
        command=PY, args=[str(SERVER), "--context", str(ctx_path),
                          "--config", str(ROOT / "configs" / "pilot.yaml")])

    async def go():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as sess:
                await sess.initialize()
                tools = await sess.list_tools()
                return sorted(t.name for t in tools.tools)

    assert asyncio.run(go()) == ["submit_measurements"]
