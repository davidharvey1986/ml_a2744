#!/bin/bash
# Re-evaluate the plain spin2 models with the shape noise composed as complex numbers
# (dataset.apply_intrinsic_ell, fixed in ea47c77): summaries centred and shifted 500 kpc,
# the response plot, and the old against the new response; then the jit500 evaluation.
# Steps run one after the other (get_obs_data rewrites the A2744 filtered fits).
cd "$(dirname "$0")/.."
PY=astro-env/bin/python
LOG=sbi/logs_eval_chifix
MODEL=tweakedalign_k0p9_ctr_ba_best
mkdir -p $LOG
step() { name=$1; shift; echo "== $(date +%H:%M) $name"; "$@" > $LOG/$name.txt 2>&1 || echo "   FAILED (see $LOG/$name.txt)"; }

S=$PWD/sbi/summaries_spin2_best_n5_chifix
step summaries $PY sbi/make_summaries.py --model $MODEL --include-str spin2_ft_ \
    --n-noise 5 --domains bahamas darkskies --no-latents --out-dir $S
step response $PY sbi/response_plot.py --summaries $S --label "spin2 best, complex noise" \
    --centre-json sbi/results/centre_uncertainty/${MODEL}_spin2.json
step response_old_vs_new $PY sbi/jitter_response.py --title "spin2 best: shape noise as trained vs composed correctly" \
    --summaries sbi/summaries_spin2_best_n5 $S --labels "noise as trained" "complex noise" \
    --out sbi/results/chifix/response_old_vs_new.png
grep -E "bahamas|darkskies" $LOG/response_old_vs_new.txt
step summaries_shift500 $PY sbi/make_summaries.py --model $MODEL --include-str spin2_ft_ --centre-jitter-kpc 500 \
    --n-noise 1 --domains bahamas darkskies --no-latents --out-dir $PWD/sbi/summaries_spin2_best_shift500_chifix
echo "== $(date +%H:%M) chifix done, starting the jit500 evaluation"
exec ./sbi/evaluate_jit500.sh
