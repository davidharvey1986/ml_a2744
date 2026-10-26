#!/bin/bash
# Copy what training needs to bonsai10 through the UNIGE login node.
# You are asked for your password once: the first ssh opens a shared connection that
# the rsync calls reuse (ControlPersist keeps it for 2 hours).
#
# Usage (from anywhere): bash slurm/send_to_bonsai.sh
set -e
cd "$(dirname "$0")/.."

JUMP=${unige:-dharvey@login22.astro.unige.ch}
DEST_HOST=dharvey@bonsai10
DEST=/srv/scratch/dharvey/ml_nature
BASE_MODELS=/Users/davidharvey/Work/Dark_Matter_DA/code/models/base_models

mkdir -p ~/.ssh/sockets
SSH="ssh -J $JUMP -o ControlMaster=auto -o ControlPath=$HOME/.ssh/sockets/%C -o ControlPersist=2h"

echo "== connecting (enter your password when asked)"
$SSH $DEST_HOST "mkdir -p $DEST/models/base_models $DEST/data/100/obs/concat $DEST/data/100/a2744"

echo "== code"
rsync -av -e "$SSH" --exclude .git --exclude data --exclude models --exclude astro-env --exclude wandb --exclude logs \
    --exclude 'sbi/summaries*' --exclude sbi/results --exclude notebooks/pickles \
    --exclude __pycache__ --exclude .ipynb_checkpoints ./ $DEST_HOST:$DEST/

echo "== data: obs/concat without flamingo, and the noise model"
rsync -avR -e "$SSH" --exclude 'flamingo*' data/100/obs/concat data/100/a2744/ngal_concat.pkl $DEST_HOST:$DEST/

echo "== base models"
rsync -av -e "$SSH" --include '*_nob1_final.pth' --exclude '*' "$BASE_MODELS"/ $DEST_HOST:$DEST/models/base_models/

echo "== done; on bonsai: cd $DEST && sbatch slurm/incremental_learning.sbatch"
