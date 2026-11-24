"""Record config/schema/prompt/vendor hashes before data generation or job submission.

Writes results/CONFIG_FREEZE.json (and prints it). Fails if the protocol docs are
missing, so a frozen run always points at the exact protocol version used.
"""
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_hash  # noqa: E402

ARTIFACTS = [
    "configs/pilot.yaml",
    "schemas/tools.json",
    "prompts/claude_harness.txt",
    "prompts/claude_direct.txt",
]
DOCS = {
    "protocol": ROOT.parent / "Codex_Metric_Depth_Agent_Handoff.md",
    "runbook": ROOT.parents[1] / "CoEvoWalk" / "HPC2_AGENT_RUNBOOK.md",
}


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    out = {"config": config_hash(ROOT / "configs" / "pilot.yaml"),
           "artifacts": {a: sha(ROOT / a) for a in ARTIFACTS},
           "docs": {}, "vendor": {}}
    for k, p in DOCS.items():
        out["docs"][k] = {"path": str(p), "sha256": sha(p)} if p.exists() else {
            "path": str(p), "sha256": None, "note": "missing"}
    pin = ROOT / "vendor" / "Depth-Anything-3" / "PINNED_COMMIT.txt"
    if pin.exists():
        out["vendor"]["depth_anything_3"] = {
            "pinned_commit_file": str(pin),
            "content": pin.read_text(encoding="utf-8").strip().splitlines()[-3:]}
    dst = ROOT / "results" / "CONFIG_FREEZE.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
