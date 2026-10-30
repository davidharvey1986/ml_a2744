#!/bin/bash
# Evaluate the spin2 models re-fine-tuned with the shape noise composed correctly (NOISE_TAG=_cx,
# run name *_nob1_spin2_cx_ft_tweakedalign_k0p9_ctr*). Waits on bonsai for NEED (default 30) _ba_best
# checkpoints, copies them, then runs one step at a time (get_obs_data rewrites the A2744 filtered
# fits). Logs to sbi/logs_eval_cx/<step>.txt.
cd "$(dirname "$0")/.."
PY=astro-env/bin/python
LOG=sbi/logs_eval_cx
INC=spin2_cx_ft_
MODEL=tweakedalign_k0p9_ctr_ba_best
REMOTE=/srv/scratch/dharvey/ml_nature/models/concat
mkdir -p $LOG
step() { name=$1; shift; echo "== $(date +%H:%M) $name"; "$@" > $LOG/$name.txt 2>&1 || echo "   FAILED (see $LOG/$name.txt)"; }

until [ "$(ssh -o BatchMode=yes bonsai "ls $REMOTE/*${INC}*_ba_best.pth 2>/dev/null | wc -l" 2>/dev/null)" -ge ${NEED:-30} ]; do
    sleep 300
done
sleep 30
echo "== $(date +%H:%M) copying checkpoints"
scp -q "bonsai:$REMOTE/*${INC}*_ba_best.pth" models/concat/ || exit 1

S=$PWD/sbi/summaries_spin2cx_best_n5
step summaries $PY sbi/make_summaries.py --model $MODEL --include-str $INC \
    --n-noise 5 --domains bahamas darkskies --no-latents --out-dir $S
step centre_uncertainty $PY sbi/centre_uncertainty.py --model $MODEL --include-str $INC
step response $PY sbi/response_plot.py --summaries $S --label "spin2 re-fine-tuned, complex noise" \
    --centre-json sbi/results/centre_uncertainty/${MODEL}_${INC}.json
grep -E "bahamas|darkskies" $LOG/response.txt
step response_ratio $PY sbi/response_ratio.py --summaries $S --label "spin2 re-fine-tuned, complex noise"
grep -E "^bahamas|^darkskies" $LOG/response_ratio.txt
step response_compare $PY sbi/jitter_response.py --title "spin2: old models vs re-fine-tuned with the corrected noise" \
    --summaries sbi/summaries_spin2_best_n5 sbi/summaries_spin2_best_n5_chifix $S \
    --labels "old models, old noise" "old models, corrected noise" "re-fine-tuned, corrected noise" \
    --out sbi/results/chifix/response_refinetuned.png
grep -E "bahamas|darkskies" $LOG/response_compare.txt
step eb_swap $PY sbi/eb_swap_test.py --model $MODEL --include-str $INC \
    --summaries $S --out-dir $PWD/sbi/results/eb_swap_test_spin2cx
out=sbi/results/sensitivity_spin2cx_best_stride1
step sensitivity $PY sbi/sensitivity.py --model $MODEL --include-str $INC --nmodels 5 --stride 1 \
    --domains --summaries $S --out-dir $out
step sensitivity_plot $PY sbi/sensitivity.py --a2744-only --model $MODEL --include-str $INC \
    --sigma-range 2 5 --no-excess-contours --out-dir $out
grep -E "P\(SIDM\)|pushes" $LOG/sensitivity*.txt
echo "== $(date +%H:%M) done"
