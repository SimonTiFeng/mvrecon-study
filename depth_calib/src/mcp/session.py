"""Per-sample MCP session state and server-side budget enforcement."""
from __future__ import annotations

import asyncio
import json
import pathlib

from .store import Store


class Session:
    def __init__(self, context: dict, config: dict):
        self.context = context
        self.config = config
        self.image_id: str = context["image_id"]
        self.image_path: str = context["image_path"]
        self.width: int = int(context["width"])
        self.height: int = int(context["height"])
        self.mode: str = context.get("mode", "harness")   # harness | direct
        ts = context.get("toolset")
        if ts is None:
            ts = "none" if self.mode == "direct" else "all"
        self.toolset: str = ts            # none | points | direct_coords | all
        self.require_point_ids: bool = bool(context.get("require_point_ids", False))
        self.allow_points = self.toolset in ("all", "points")
        self.allow_observe = self.toolset != "none"
        self.out_dir = pathlib.Path(context["output_dir"])
        self.store = Store(self.out_dir, self.image_id)

        b = config["budget"]
        self.visual_calls_max = int(b["visual_calls_max"])
        self.crops_max = int(b["crops_max"])
        self.anchors_max = int(b["anchors_max"])
        self.attempts_max = int(b["submission_attempts_max"])

        self.visual_calls = 0
        self.crop_calls = 0
        self.submission_attempts = 0
        self.point_registry: dict[str, list[float]] = {}
        self._point_cache: dict[tuple, str] = {}
        self.view_counter = 0
        self.lock = asyncio.Lock()

        prior = self.store.load_state()
        self.committed = bool(prior and prior.get("session_status") == "committed")
        self.submission_id = prior.get("submission_id") if prior else None
        self.request_hash = prior.get("request_hash") if prior else None
        self._restore_usage()

    def _usage_path(self) -> pathlib.Path:
        return self.out_dir / "state" / f"{self.image_id}.usage.json"

    def _restore_usage(self) -> None:
        p = self._usage_path()
        if not p.exists():
            return
        u = json.loads(p.read_text(encoding="utf-8"))
        self.visual_calls = int(u.get("visual_calls", 0))
        self.crop_calls = int(u.get("crop_calls", 0))
        self.submission_attempts = int(u.get("submission_attempts", 0))
        self.view_counter = int(u.get("view_counter", 0))
        self.point_registry = {k: list(v) for k, v in u.get("point_registry", {}).items()}
        self._point_cache = {}
        for k, v in u.get("point_cache", []):
            roi, maxp = k
            self._point_cache[(tuple(roi), int(maxp))] = v

    # -- budgets ----------------------------------------------------------
    def remaining_visual(self) -> int:
        return max(0, self.visual_calls_max - self.visual_calls)

    def persist_usage(self) -> None:
        """Usage/registry evidence survives across CLI-style tool invocations."""
        p = self._usage_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "visual_calls": self.visual_calls,
            "crop_calls": self.crop_calls,
            "submission_attempts": self.submission_attempts,
            "view_counter": self.view_counter,
            "point_registry": self.point_registry,
            "point_cache": [[list(k), v] for k, v in self._point_cache.items()],
            "visual_calls_max": self.visual_calls_max,
            "crops_max": self.crops_max,
            "attempts_max": self.attempts_max,
        }), encoding="utf-8")

    def consume_visual(self, is_crop: bool) -> tuple[bool, str | None]:
        if self.committed:
            return False, "ALREADY_FINALIZED"
        if self.visual_calls >= self.visual_calls_max:
            return False, "VISUAL_BUDGET_EXHAUSTED"
        if is_crop and self.crop_calls >= self.crops_max:
            return False, "CROP_BUDGET_EXHAUSTED"
        self.visual_calls += 1
        if is_crop:
            self.crop_calls += 1
        self.persist_usage()
        return True, None

    def next_view_id(self) -> str:
        self.view_counter += 1
        self.persist_usage()
        return f"v{self.view_counter}"
    def register_points(self, points: list[list[float]], view_id: str) -> list[dict]:
        out = []
        for i, xy in enumerate(points):
            pid = f"{view_id}:p{i}"
            self.point_registry[pid] = [float(xy[0]), float(xy[1])]
            out.append({"point_id": pid, "xy_original": list(self.point_registry[pid])})
        return out

    def cached_points(self, key: tuple) -> str | None:
        return self._point_cache.get(key)

    def cache_points(self, key: tuple, view_id: str) -> None:
        self._point_cache[key] = view_id


def load_context(path: str) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
