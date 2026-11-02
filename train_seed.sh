#!/bin/bash
# Train one seed: base model (if missing) -> incremental fine-tuning over intrinsic
# ellipticity -> final "_ba" stage. Called by incremental_learning.sh (local loop over
# seeds) and slurm/incremental_learning.sbatch (one seed per array task).
#
# Usage: train_seed.sh SEED [FILTER]
# Environment:
#   BASE_DIR    where base models are read/written
#               (default: /Users/davidharvey/Work/Dark_Matter_DA/code/models/base_models)
#   PYTHON      python executable (default: python)
#   EXTRA_ARGS  extra arguments passed to every main.py call (e.g. "--num_workers 8")
#   CENTRE_JITTER_KPC  if > 0, shift the training maps by a Gaussian offset with this sigma
#               per axis [kpc] in the fine-tuning and _ba stages (35 = the 50 kpc rms A2744
#               centring error); the run name gets _ftjit<kpc>
#   JITTER_BASE if 1 (with CENTRE_JITTER_KPC), retrain the base model with the jitter as well
#   NOISE_TAG   added to the fine-tuned run name (default _cx: shape noise composed as complex
#               numbers, dataset.py ea47c77; '' for the old names)
#               instead of starting from the un-jittered base model; the run name gets _jit<kpc>
#   PER_GALAXY_CHI  1 (default): shape noise combined with the shear per galaxy, then averaged per pixel,
#               as the data map (dataset.apply_intrinsic_ell --per_galaxy_chi); the run name gets _pgc.
#               0 for the old pixel-mean combination and names
set -e
SEED=${1:?usage: train_seed.sh SEED [FILTER]}
FILTER=${2:-concat}
cd "$(dirname "$0")"
mkdir -p logs

PYTHON=${PYTHON:-python}
BASE_DIR=${BASE_DIR:-/Users/davidharvey/Work/Dark_Matter_DA/code/models/base_models}
EXTRA_ARGS=${EXTRA_ARGS:-}
CENTRE_JITTER_KPC=${CENTRE_JITTER_KPC:-0}
JITTER_BASE=${JITTER_BASE:-0}
NOISE_TAG=${NOISE_TAG-_cx}
PER_GALAXY_CHI=${PER_GALAXY_CHI:-1}
PGC_ARGS="--no-per_galaxy_chi"       # utils.py defaults to per-galaxy chi
PGC_TAG=""
if [ "${PER_GALAXY_CHI}" == "1" ]; then
    PGC_ARGS="--per_galaxy_chi"
    PGC_TAG="_pgc"
fi
JITTER_ARGS=""
JITTER_TAG=""
if [ "${CENTRE_JITTER_KPC}" != "0" ]; then
    JITTER_ARGS="--aug_centre_jitter_kpc ${CENTRE_JITTER_KPC}"
    JITTER_TAG="_jit${CENTRE_JITTER_KPC}"
fi
BASE_JITTER_ARGS=""
BASE_JITTER_TAG=""
FT_JITTER_TAG=""
if [ -n "${JITTER_TAG}" ] && [ "${JITTER_BASE}" == "1" ]; then
    BASE_JITTER_ARGS=${JITTER_ARGS}
    BASE_JITTER_TAG=${JITTER_TAG}
elif [ -n "${JITTER_TAG}" ]; then
    FT_JITTER_TAG="_ftjit${CENTRE_JITTER_KPC}"
fi
mkdir -p "$BASE_DIR"

fine_tune=0 #the weight to force the fine-tuning to align domains
adaptation_weight=1

function get_zs () {
    if [[ $1 == "concat" ]]; then
        zs=2.28
    elif [[ $1 == "f115w" ]]; then
        zs=2.28
    elif [[ $1 == "f150w" ]]; then
        zs=2.28
    else
        echo "NOT FOUND"
        exit 1
    fi
    echo $zs
}
ZS=$(get_zs "${FILTER}")

SRC="bahamas"
TGT="darkskies"
TGT_LAB=`echo $TGT | cut -c1-4`
SRC_LAB=`echo $SRC | cut -c1-4`

COMMON_ARGS="--batch_size 32 --epochs 100 --lr 0.0001 --eval_interval 10 --project_name shear_input --image_size 100 --use_wandb"
COMMON_ARGS=$COMMON_ARGS" --model squeezenet1_1 --pretrained --mass_index 0 --in_channels 2 --verbose --weighting_scheme inverse_frequency"

# _spin2: trained with the spin-2 aware flips/rotations (dataset.SpinFlip/SpinRotation), so
# the base models are retrained rather than reused from the E/B-blind runs
RUN_NAME=cdan_${SRC_LAB}2${TGT_LAB}_pre_squeezenet1_aw_${adaptation_weight}_pad_shear_avgpool_gauss_seed_${SEED}_nob1_spin2${BASE_JITTER_TAG}

if [ ! -f ${BASE_DIR}/${RUN_NAME}_final.pth ]
then
    $PYTHON main.py $COMMON_ARGS $EXTRA_ARGS $BASE_JITTER_ARGS \
        --source_domain $SRC --target_domain $TGT \
        --run_name ${RUN_NAME} \
        --adaptation_weight ${adaptation_weight} \
        --adaptation cdan \
        --aug_rotation_prob 1.0 \
        --num_avgpool_head 1  \
        --seed ${SEED} \
        --save_dir ${BASE_DIR} \
        --ignore_dataset bahamas_1.pkl \
        --apply_intrinsic_ell 0. \
        --zl 0.305
else
    echo "Done base model"
fi

SRC="bahamas_obs"
TGT="darkskies_obs"

PREVIOUS_NAME=${BASE_DIR}/${RUN_NAME}_final.pth

COMMON_ARGS="--batch_size 32 --epochs 10 --lr 0.0001 --eval_interval 2 --project_name incremental_learning --image_size 100 --use_wandb"
COMMON_ARGS=$COMMON_ARGS" --model squeezenet1_1 --pretrained --mass_index 0 --in_channels 2 --verbose --weighting_scheme inverse_frequency"

TUNE_DIR=models/${FILTER}/
OUTPUT=${RUN_NAME}${FT_JITTER_TAG}${NOISE_TAG}${PGC_TAG}_ft_tweakedalign_k0p9_ctr   # stamp centred on the A2744 main peak

if [ -f ${TUNE_DIR}/${OUTPUT}_best.pth ]
then
        echo "Already fine-tiuned"
        exit
fi
for intrinsic_ell in 0.1 0.3 0.5 0.7 0.9
do
    

    $PYTHON main.py $COMMON_ARGS $EXTRA_ARGS $JITTER_ARGS $PGC_ARGS --source_domain $SRC --target_domain $TGT \
        --model squeezenet1_1 --pretrained \
        --weighting_scheme inverse_frequency \
        --run_name ${OUTPUT} --checkpoint ${PREVIOUS_NAME} \
        --adaptation_weight ${adaptation_weight} --adaptation cdan \
        --aug_rotation_prob 1.0 \
        --apply_intrinsic_ell $intrinsic_ell \
        --num_avgpool_head 1  \
        --seed $SEED \
        --zl 0.305 \
        --jwst_filter ${FILTER} \
        --save_dir ${TUNE_DIR} \
        --ignore_dataset bahamas_1.pkl

    PREVIOUS_NAME=${TUNE_DIR}/${OUTPUT}_best.pth
done

PREVIOUS_NAME=${TUNE_DIR}/${OUTPUT}_best.pth
COMMON_ARGS="--batch_size 32 --epochs 20 --lr 0.0001 --eval_interval 2 --project_name incremental_learning --image_size 100 --use_wandb"
COMMON_ARGS=$COMMON_ARGS" --model squeezenet1_1 --pretrained --mass_index 0 --in_channels 2 --verbose --weighting_scheme inverse_frequency"

# the final iteration is to save the most aligned version
$PYTHON main.py $COMMON_ARGS $EXTRA_ARGS $JITTER_ARGS $PGC_ARGS --source_domain $SRC --target_domain $TGT \
    --model squeezenet1_1 --pretrained \
    --weighting_scheme inverse_frequency \
    --run_name ${OUTPUT}_ba --checkpoint ${PREVIOUS_NAME} \
    --adaptation_weight ${adaptation_weight} --adaptation cdan \
    --aug_rotation_prob 1.0 \
    --apply_intrinsic_ell 1.0 \
    --num_avgpool_head 1  \
    --seed $SEED \
    --zl 0.305 \
    --zs ${ZS} \
    --jwst_filter ${FILTER} \
    --save_dir ${TUNE_DIR}
