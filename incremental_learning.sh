#!/bin/bash
# Train all seeds locally, one after another. The per-seed steps are in train_seed.sh;
# on the cluster use slurm/incremental_learning.sbatch, which runs one seed per GPU.
# Environment: BASE_DIR, PYTHON, EXTRA_ARGS (see train_seed.sh).
cd "$(dirname "$0")"

for FILTER in concat
do
    for SEED in {30..1}
    do
        bash train_seed.sh "$SEED" "$FILTER"
    done
done
