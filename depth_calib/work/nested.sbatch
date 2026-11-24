#!/usr/bin/env bash
#SBATCH --job-name=dc-nested
#SBATCH --partition=i64m1tga40u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date)"
timeout 15s nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
  echo "BADNODE"; exit 42; }

cd "$HOME/agent/depth_calib" || exit 2
PY="$HOME/envs/dc/bin/python"

"$PY" -u work/nested_multi.py
echo DONE
