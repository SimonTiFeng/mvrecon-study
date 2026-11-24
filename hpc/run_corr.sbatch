#!/bin/bash
#SBATCH --job-name=mvcorr
#SBATCH --partition=i64m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=mvcorr_%j.log
#SBATCH --error=mvcorr_%j.err

# Geometry work, no GPU needed: CPU partition keeps the GPU quota free.
source /opt/slurm/profile.slurm 2>/dev/null
set -e
cd "$HOME/mvrecon_hpc"
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"
"$HOME/mvrenv/bin/python" -c "import cv2, numpy, scipy; print('cv2', cv2.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)"
"$HOME/mvrenv/bin/python" exp_corr.py --root . --out "results_${SLURM_JOB_ID}.jsonl"
echo "end=$(date -Is) lines=$(wc -l < results_${SLURM_JOB_ID}.jsonl)"
