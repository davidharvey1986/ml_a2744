#!/bin/bash
# Incremental-learning training on a SLURM GPU cluster: one seed per array task, one GPU each.
#
# Submit from the repo root on the cluster:
#   sbatch slurm/incremental_learning.sbatch                 # seeds 1-30
#   sbatch --array=5,12 slurm/incremental_learning.sbatch    # only some seeds
# Before the first run, set the cluster-specific lines marked EDIT below.
#
#SBATCH --job-name=incr_learn
#SBATCH --array=1-30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err
##SBATCH --partition=gpu          # EDIT: GPU partition name on bonsai
##SBATCH --account=<account>      # EDIT: if the cluster needs one

set -e
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# EDIT: activate the Python environment with torch (CUDA build), lenspack, astropy, ...
# module load cuda
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate ml_check

# Base models live in the repo on the cluster (copy existing ones here to skip stage 1)
export BASE_DIR=${BASE_DIR:-$PWD/models/base_models}
export PYTHON=${PYTHON:-python}
# Load data in parallel with the GPU; keep wandb runs local unless logged in on the nodes
export EXTRA_ARGS=${EXTRA_ARGS:-"--num_workers ${SLURM_CPUS_PER_TASK:-4}"}
export WANDB_MODE=${WANDB_MODE:-offline}

echo "seed ${SLURM_ARRAY_TASK_ID} on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-none}"
$PYTHON -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

bash train_seed.sh "${SLURM_ARRAY_TASK_ID}" concat
