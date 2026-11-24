#!/usr/bin/env bash
#SBATCH --job-name=dc-baseline-probe
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date)"
timeout 15s nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
  echo "BADNODE: nvidia-smi failed"; exit 42; }

cd "$HOME/agent/depth_calib" || exit 2
PY="$HOME/envs/dc/bin/python"

"$PY" scripts/baseline_scale_probe.py \
  --scene-root "$HOME/jhspoolers/dc_data/eth3d_extracted/courtyard" \
  --scene courtyard \
  --images DSC_0286.JPG DSC_0287.JPG DSC_0288.JPG DSC_0289.JPG DSC_0290.JPG \
  --out "$HOME/agent/depth_calib/results/bl_courtyard.json" \
  --device cuda \
  --local-weights "$HOME/agent/depth_calib/models/DA3METRIC-LARGE"

echo DONE
