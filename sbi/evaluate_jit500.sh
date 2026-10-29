#!/bin/bash
# Evaluate the models trained with 500 kpc centre jitter in both the base and fine-tuning stages
# (train_seed.sh JITTER_BASE=1 CENTRE_JITTER_KPC=500, run name *_nob1_spin2_jit500_ft_tweakedalign_k0p9_ctr*)
# against the plain spin2 models. Waits on bonsai for all 10 _ba_best checkpoints, copies them, then
# runs every step one after the other (the A2744 catalogue steps must not overlap: get_obs_data
# rewrites the filtered fits). Logs to sbi/logs_eval_jit500/<step>.txt. The spin2 baselines are the
# summaries with the corrected noise composition (evaluate_chifix.sh, which runs this script last).
cd "$(dirname "$0")/.."
PY=astro-env/bin/python
SBI_PY=../Dark_Matter_DA/sbi/bin/python
LOG=sbi/logs_eval_jit500
INC=spin2_jit500_ft_
MODEL=tweakedalign_k0p9_ctr_ba_best
REMOTE=/srv/scratch/dharvey/ml_nature/models/concat
mkdir -p $LOG
step() { name=$1; shift; echo "== $(date +%H:%M) $name"; "$@" > $LOG/$name.txt 2>&1 || echo "   FAILED (see $LOG/$name.txt)"; }

until [ "$(ssh -o BatchMode=yes bonsai "ls $REMOTE/*${INC}*_ba_best.pth 2>/dev/null | wc -l" 2>/dev/null)" -ge 10 ]; do
    sleep 120
done
sleep 30                                   # let the last checkpoint finish writing
echo "== $(date +%H:%M) copying checkpoints"
scp -q "bonsai:$REMOTE/*${INC}*_ba_best.pth" models/concat/ || exit 1
ls models/concat/*${INC}*_ba_best.pth | wc -l

S=$PWD/sbi/summaries_jit500_best_n5
step summaries $PY sbi/make_summaries.py --model $MODEL --include-str $INC \
    --n-noise 5 --domains bahamas darkskies --no-latents --out-dir $S
step summaries_shift500 $PY sbi/make_summaries.py --model $MODEL --include-str $INC --centre-jitter-kpc 500 \
    --n-noise 1 --domains bahamas darkskies --no-latents --out-dir $PWD/sbi/summaries_jit500_best_shift500
step eb_swap $PY sbi/eb_swap_test.py --model $MODEL --include-str $INC \
    --summaries $S --out-dir $PWD/sbi/results/eb_swap_test_jit500
step jitter_response $PY sbi/jitter_response.py --title "spin2 vs spin2 jit500 models" \
    --summaries sbi/summaries_spin2_best_n5_chifix sbi/summaries_spin2_best_shift500_chifix $S sbi/summaries_jit500_best_shift500 \
    --labels "spin2, centred" "spin2, shifted 500 kpc" "jit500, centred" "jit500, shifted 500 kpc" \
    --out sbi/results/jitter_response/jitter_response_jit500_models.png
step centre_uncertainty $PY sbi/centre_uncertainty.py --model $MODEL --include-str $INC
step response $PY sbi/response_plot.py --summaries $S --label "spin2 jit500 best" \
    --centre-json sbi/results/centre_uncertainty/${MODEL}_${INC}.json
for c in "" O1 O2 O3; do
    out=sbi/results/sensitivity_jit500_best_stride1${c:+_centre_$c}
    step sensitivity${c:+_$c} $PY sbi/sensitivity.py --model $MODEL --include-str $INC --nmodels 5 --stride 1 \
        --domains --summaries $S --sigma-range 2 5 --no-excess-contours ${c:+--recentre $c} --out-dir $out
    grep -E "P\(SIDM\)|windows >=" $LOG/sensitivity${c:+_$c}.txt
done
echo "== $(date +%H:%M) done"
