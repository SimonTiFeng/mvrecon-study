#!/usr/bin/env bash
#SBATCH --job-name=depth-gate
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Free debug-gate: run-qualification check before any paid partition.
# NOT a formal result: verifies code path, real inference steps, cache write/read,
# checkpoint provenance and GPU memory on the cluster.
set -uo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

ROOT=/hpc2hdd/home/hourujiang/jhspoolers/depthcalib
PY=$ROOT/venv/bin/python
cd "$ROOT/project" || exit 42

echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is) partition=$SLURM_JOB_PARTITION"
timeout 20s nvidia-smi --query-gpu=name,memory.total,memory.used \
  --format=csv,noheader || { echo "BADNODE: nvidia-smi failed"; exit 42; }
"$PY" - <<'PYCHECK' || exit 43
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
assert torch.cuda.is_available()
print("gpu", torch.cuda.get_device_name(0),
      "total_MB", int(torch.cuda.get_device_properties(0).total_memory/1e6))
PYCHECK

GATE_OUT=$ROOT/gate/$SLURM_JOB_ID
export GATE_OUT
mkdir -p "$GATE_OUT"
# Two real inference steps (A and B), one image each, then reload the cache.
"$PY" -u scripts/infer_depths.py \
  --manifest "$ROOT/manifests/gate.jsonl" \
  --backends A B --limit 1 --out "$GATE_OUT/depths" || exit 44
"$PY" - <<'PYVERIFY' || exit 45
import glob, os, numpy as np, pathlib
files = sorted(glob.glob(os.path.join(os.environ["GATE_OUT"], "depths", "*.npz")))
assert len(files) == 2, f"expected 2 cache files, got {files}"
for f in files:
    with np.load(f) as z:
        d = z["depth"]
        assert d.ndim == 2 and np.isfinite(d).any(), f
        print(pathlib.Path(f).name, d.shape, float(np.nanmin(d)), float(np.nanmax(d)))
PYVERIFY
GATE_OUT="$GATE_OUT" "$PY" - <<'PYMEM' || true
import json, os, subprocess
nvsmi = subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.total","--format=csv,noheader"],
                       capture_output=True, text=True).stdout.strip()
open(os.path.join(os.environ["GATE_OUT"],"gpu_after.txt"),"w").write(nvsmi+"\n")
print("gpu after:", nvsmi)
PYMEM
echo "GATE_OK job=$SLURM_JOB_ID out=$GATE_OUT"
