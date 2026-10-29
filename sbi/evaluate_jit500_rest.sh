#!/bin/bash
# The steps of evaluate_jit500.sh after the centred summaries, with the A2744 sensitivity map at the
# adopted centre first. Same logs (sbi/logs_eval_jit500), one step at a time.
cd "$(dirname "$0")/.."
PY=astro-env/bin/python
LOG=sbi/logs_eval_jit500
INC=spin2_jit500_ft_
MODEL=tweakedalign_k0p9_ctr_ba_best
S=$PWD/sbi/summaries_jit500_best_n5
step() { name=$1; shift; echo "== $(date +%H:%M) $name"; "$@" > $LOG/$name.txt 2>&1 || echo "   FAILED (see $LOG/$name.txt)"; }
sens() {
    c=$1
    step sensitivity${c:+_$c} $PY sbi/sensitivity.py --model $MODEL --include-str $INC --nmodels 5 --stride 1 \
        --domains --summaries $S --sigma-range 2 5 --no-excess-contours ${c:+--recentre $c} \
        --out-dir sbi/results/sensitivity_jit500_best_stride1${c:+_centre_$c}
    grep -E "P\(SIDM\)|windows >=" $LOG/sensitivity${c:+_$c}.txt
}
sens ""
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
for c in O1 O2 O3; do sens $c; done
echo "== $(date +%H:%M) done"
