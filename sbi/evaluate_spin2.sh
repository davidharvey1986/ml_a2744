#!/bin/bash
# Evaluate the spin-2 models (train_seed.sh run name *_nob1_spin2_ft_tweakedalign_k0p9_ctr*) on
# BAHAMAS and DARKSKIES: summaries, group comparison, conditioned posteriors, E/B test,
# obs-map checks, centring systematic, response plot and the systematics suite.
# Run from anywhere; each step logs to sbi/logs_eval_spin2/<step>.txt and a failed step
# does not stop the others.
cd "$(dirname "$0")/.."
PY=astro-env/bin/python
SBI_PY=../Dark_Matter_DA/sbi/bin/python
LOG=sbi/logs_eval_spin2
mkdir -p $LOG
step() { name=$1; shift; echo "== $(date +%H:%M) $name"; "$@" > $LOG/$name.txt 2>&1 || echo "   FAILED (see $LOG/$name.txt)"; }

for g in best final; do
    step summaries_$g $PY sbi/make_summaries.py --model tweakedalign_k0p9_ctr_ba_$g --include-str spin2 \
        --n-noise 5 --domains bahamas darkskies --no-latents --out-dir sbi/summaries_spin2_${g}_n5
done
step compare_groups $PY sbi/compare_model_groups.py --match-ncomp
for g in best final; do
    step posteriors_$g $SBI_PY sbi/run_sbi.py --summaries sbi/summaries_spin2_${g}_n5 --noise all --a2744-ncomp 2+ \
        --out-dir sbi/results/summaries_spin2_${g}_n5_alldraws_nc2plus
done
step centre_uncertainty $PY sbi/centre_uncertainty.py --model tweakedalign_k0p9_ctr_ba_best --include-str spin2
step response_best $PY sbi/response_plot.py --summaries sbi/summaries_spin2_best_n5 --label "spin2 best" \
    --centre-json sbi/results/centre_uncertainty/tweakedalign_k0p9_ctr_ba_best_spin2.json
step response_final $PY sbi/response_plot.py --summaries sbi/summaries_spin2_final_n5 --label "spin2 final"
step eb_swap $PY sbi/eb_swap_test.py --model tweakedalign_k0p9_ctr_ba_best --include-str spin2 \
    --summaries $PWD/sbi/summaries_spin2_best_n5 --out-dir $PWD/sbi/results/eb_swap_test_spin2
step obs_map_checks $PY sbi/obs_map_checks.py --model tweakedalign_k0p9_ctr_ba_best --include-str spin2 \
    --summaries $PWD/sbi/summaries_spin2_best_n5 --n 100 --out-dir $PWD/sbi/results/obs_map_checks_spin2
step systematics $PY sbi/systematics.py --model tweakedalign_k0p9_ctr_ba_best --include-str spin2 --min-ncomp 2 \
    --summaries sbi/summaries_spin2_best_n5 --out-dir sbi/results/systematics_spin2_best
echo "== $(date +%H:%M) done"
