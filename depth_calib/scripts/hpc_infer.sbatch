#!/usr/bin/env bash
#SBATCH --job-name=depth-infer
#SBATCH --partition=i64m1tga40u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Formal frozen-backend depth inference on the full pilot/dev/smoke manifests.
set -uo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

ROOT=/hpc2hdd/home/hourujiang/jhspoolers/depthcalib
PY=$ROOT/venv/bin/python
cd "$ROOT/project" || exit 42

echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is) partition=$SLURM_JOB_PARTITION"
timeout 20s nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
  || { echo "BADNODE: nvidia-smi failed"; exit 42; }

MANIFEST=${MANIFEST:-$ROOT/manifests/all.jsonl}
"$PY" -u scripts/infer_depths.py \
  --manifest "$MANIFEST" \
  --backends A B --out "$ROOT/cache/depths" || exit 44

echo "INFER_OK job=$SLURM_JOB_ID end=$(date -Is)"
