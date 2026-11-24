"""Vision-agent tool client: same server-side contract, invoked from the CLI.

The measurement agent (Claude vision) calls these subcommands; every call goes
through the same handlers the MCP server exposes, so budgets, validation and
atomic evidence storage are identical. No GT, depth or scores are reachable here.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

_ROOTS = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOTS) not in sys.path:
    sys.path.insert(0, str(_ROOTS))

from src.config import load_config                 # noqa: E402
from src.mcp import handlers                       # noqa: E402
from src.mcp.session import Session, load_context  # noqa: E402


def _bbox(s: str) -> list[int]:
    v = [int(x) for x in s.replace(",", " ").split()]
    if len(v) != 4:
        raise SystemExit("bbox must be x0 y0 x1 y1")
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["status", "crop", "points", "render", "submit"])
    ap.add_argument("--context", required=True)
    ap.add_argument("--config", default=str(_ROOTS / "configs" / "pilot.yaml"))
    ap.add_argument("--bbox")
    ap.add_argument("--roi", default="none")
    ap.add_argument("--max", type=int, default=16)
    ap.add_argument("--payload")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sess = Session(load_context(args.context), cfg)

    if args.command == "status":
        print(json.dumps({
            "image_id": sess.image_id, "mode": sess.mode,
            "committed": sess.committed, "submission_id": sess.submission_id,
            "visual_calls": sess.visual_calls, "crops": sess.crop_calls,
            "submission_attempts": sess.submission_attempts,
            "remaining_visual_calls": sess.remaining_visual(),
            "points": list(sess.point_registry.keys())}, indent=1))
        return

    if args.command == "submit":
        payload = json.loads(pathlib.Path(args.payload).read_text(encoding="utf-8"))
        status, rec = handlers.submit_measurements(payload, sess)
        print(json.dumps({"status": status, "receipt": rec}, indent=1))
        raise SystemExit(0 if status == "ok" else 1)

    if not sess.allow_observe:
        raise SystemExit("observation tools are not exposed in this mode")
    name = {"crop": "crop_image", "points": "propose_points",
            "render": "render_measurements"}[args.command]
    if name == "propose_points" and not sess.allow_points:
        raise SystemExit("propose_points is not exposed in this condition")
    ok, code = sess.consume_visual(is_crop=(name == "crop_image"))
    if not ok:
        print(json.dumps({"error": code,
                          "message": "server-side budget exhausted"}))
        raise SystemExit(2)

    if name == "crop_image":
        if not args.bbox:
            raise SystemExit("--bbox required")
        status, sc, imgs = handlers.crop_image(
            {"image_id": sess.image_id, "bbox_xyxy": _bbox(args.bbox)}, sess)
    elif name == "propose_points":
        roi = None if args.roi in ("none", "None", "") else _bbox(args.roi)
        status, sc, imgs = handlers.propose_points(
            {"image_id": sess.image_id, "roi_xyxy": roi,
             "max_points": max(1, min(32, args.max))}, sess)
    else:
        pairs = json.loads(pathlib.Path(args.payload).read_text(encoding="utf-8"))
        status, sc, imgs = handlers.render_measurements(
            {"image_id": sess.image_id, "pairs": pairs}, sess)

    if status == "ok" and "view_id" in sc:
        sc["view_path"] = (sess.out_dir / "views" / f"{sc['view_id']}.png").as_posix()
    print(json.dumps({"status": status, "result": sc}, indent=1))
    raise SystemExit(0 if status == "ok" else 1)


if __name__ == "__main__":
    main()
