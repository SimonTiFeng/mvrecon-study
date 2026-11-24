#!/usr/bin/env bash
#SBATCH --job-name=dc-depth-pose
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:28:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date)"
timeout 15s nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
  echo "BADNODE"; exit 42; }

cd "$HOME/agent/depth_calib" || exit 2
PY="$HOME/envs/dc/bin/python"
W="$HOME/agent/depth_calib/models/DA3METRIC-LARGE"
S="$HOME/jhspoolers/dc_data/eth3d_extracted/courtyard"

"$PY" -u scripts/depth_pose_scale.py \
  --scene-root "$S" --scene courtyard \
  --pairs DSC_0286.JPG DSC_0287.JPG DSC_0288.JPG DSC_0289.JPG \
  --local-weights "$W" --device cuda --stride 24 \
  --out results/dps_courtyard.json

echo DONE
