#!/usr/bin/env bash
#SBATCH --job-name=dc-ms
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
echo "job=$SLURM_JOB_ID node=$(hostname)"
cd "$HOME/agent/depth_calib" || exit 2
PY="$HOME/envs/dc/bin/python"; W="$HOME/agent/depth_calib/models/DA3METRIC-LARGE"
D="$HOME/jhspoolers/dc_data/eth3d_extracted"

run () {
  SC="$1"; shift
  echo "########## $SC ##########"
  "$PY" -u scripts/depth_pose_scale.py \
    --scene-root "$D/$SC" --scene "$SC" --pairs "$@" \
    --local-weights "$W" --device cuda --stride 24 \
    --out "results/dps_$SC.json" 2>&1 | grep -E "clouds|s=|GT-FREE|Error|Traceback"
}

run facade    DSC_0324.JPG DSC_0325.JPG DSC_0326.JPG DSC_0328.JPG
run delivery_area DSC_0675.JPG DSC_0676.JPG DSC_0677.JPG DSC_0678.JPG
run electro   DSC_9257.JPG DSC_9258.JPG DSC_9259.JPG DSC_9260.JPG
echo DONE
