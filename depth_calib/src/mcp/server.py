"""stdio MCP server bound to one sample (handoff 3.3 / 5).

Usage (paths are absolute; the scheduler writes sample_context.json):
  python src/mcp/server.py --context /abs/run/sample_context.json \
      --config /abs/project/configs/pilot.yaml --transport stdio

The process never receives ground truth, evaluation data or network access.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

_ROOTS = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOTS) not in sys.path:
    sys.path.insert(0, str(_ROOTS))

from mcp import types                                   # noqa: E402
from mcp.server import Server                           # noqa: E402
from mcp.server.stdio import stdio_server               # noqa: E402

from src.config import load_config                      # noqa: E402
from src.mcp import handlers                            # noqa: E402
from src.mcp.handlers import _TOOLS, receipt, violation  # noqa: E402
from src.mcp.session import Session, load_context       # noqa: E402


def _tool_list(mode: str, toolset: str = "all") -> list[types.Tool]:
    if toolset == "none" or mode == "direct":
        names = ["submit_measurements"]
    elif toolset == "points":
        names = ["submit_measurements", "crop_image", "propose_points", "render_measurements"]
    elif toolset == "direct_coords":
        names = ["submit_measurements", "crop_image", "render_measurements"]
    else:
        names = list(_TOOLS.keys())
    return [types.Tool(**{k: v for k, v in _TOOLS[n].items()
                          if k in types.Tool.model_fields}) for n in names]


def build_server(session: Session) -> Server:
    server = Server("depth-measurement")
    allowed = {t.name for t in _tool_list(session.mode, session.toolset)}

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return _tool_list(session.mode, session.toolset)

    @server.call_tool()
    async def call_tool(name: str, arguments: dict | None) -> types.CallToolResult:
        arguments = arguments or {}
        async with session.lock:
            if name not in allowed:
                rec = receipt(False, violations=[violation(
                    "TOOL_NOT_AVAILABLE", "$", f"tool {name!r} is not exposed in mode "
                    f"{session.mode!r}")])
                return _result(rec, is_error=True)
            if name == "submit_measurements":
                status, rec = handlers.submit_measurements(arguments, session)
                return _result(rec, is_error=(status == "error"))
            ok, code = session.consume_visual(is_crop=(name == "crop_image"))
            if not ok:
                rec = receipt(False, violations=[violation(code, "$", "server-side "
                                                           "budget exhausted")])
                return _result(rec, is_error=True)
            fn = {"crop_image": handlers.crop_image,
                  "propose_points": handlers.propose_points,
                  "render_measurements": handlers.render_measurements}[name]
            status, sc, imgs = fn(arguments, session)
            content: list[types.ContentBlock] = [
                types.ImageContent(type="image", data=data, mimeType="image/png")
                for _, data in imgs]
            content.append(types.TextContent(
                type="text", text=json.dumps(sc, ensure_ascii=False)))
            return types.CallToolResult(content=content, structuredContent=sc,
                                        isError=(status == "error"))
    return server


def _result(rec: dict, is_error: bool) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(rec, ensure_ascii=False))],
        structuredContent=rec, isError=is_error)


async def _serve(session: Session) -> None:
    server = build_server(session)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", required=True)
    ap.add_argument("--config", default=str(_ROOTS / "configs" / "pilot.yaml"))
    ap.add_argument("--transport", default="stdio", choices=["stdio"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    ctx = load_context(args.context)
    session = Session(ctx, cfg)
    print(f"[mcp] image_id={session.image_id} mode={session.mode} "
          f"committed={session.committed}", file=sys.stderr, flush=True)
    asyncio.run(_serve(session))


if __name__ == "__main__":
    main()
