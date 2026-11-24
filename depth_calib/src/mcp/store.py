"""Atomic, server-side storage for one sample's submission evidence.

Only this module writes anchors/metadata/raw payloads; the model can never pass
a file path. Writes are temp-file + os.replace so a crash cannot leave a
half-written record that the evaluator would consume.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
from datetime import datetime, timezone


def canonical_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode()).hexdigest()


def _atomic_write(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


class Store:
    def __init__(self, out_dir: str | pathlib.Path, image_id: str):
        self.root = pathlib.Path(out_dir)
        self.image_id = image_id
        self.anchors_dir = self.root / "anchors"
        self.state_path = self.root / "state" / f"{image_id}.state.json"

    def load_state(self) -> dict | None:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return None

    def save_state(self, state: dict) -> None:
        _atomic_write(self.state_path, state)

    def commit(self, *, request_hash: str, raw_arguments: dict,
               normalized: dict, trusted_metadata: dict) -> dict:
        """Write all artifacts, then the state. Returns the state record."""
        submission_id = "sub_" + request_hash[:16]
        stamp = datetime.now(timezone.utc).isoformat()
        normalized = dict(normalized)
        normalized["submission_id"] = submission_id
        normalized["schema_version"] = "0.2"
        meta = dict(trusted_metadata)
        meta.update({"submission_id": submission_id, "committed_at": stamp,
                     "request_hash": request_hash})
        _atomic_write(self.anchors_dir / f"{self.image_id}.raw.json",
                      {"arguments": raw_arguments, "request_hash": request_hash})
        _atomic_write(self.anchors_dir / f"{self.image_id}.metadata.json", meta)
        _atomic_write(self.anchors_dir / f"{self.image_id}.json", normalized)
        state = {"image_id": self.image_id, "session_status": "committed",
                 "submission_id": submission_id, "request_hash": request_hash,
                 "committed_at": stamp, "anchor_count": len(normalized.get("anchors", [])),
                 "status": normalized.get("status")}
        self.save_state(state)
        return state
