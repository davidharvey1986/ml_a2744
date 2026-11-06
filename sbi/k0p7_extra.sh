#!/bin/bash
# After k0p7_cx.sh (kappa < 0.7): the response curves, the E/B-swap test and the sensitivity maps for
# ba_best and ba_final, as evaluate_cx.sh, rerun_cx.sh, rerun_cx_final.sh and after_final.sh did at
# kappa < 0.9 (those outputs are in archive_k0p9/). One catalogue reader at a time. Logs to sbi/logs_k0p7/.
#
#   bash sbi/k0p7_extra.sh
cd "$(dirname "$0")/.."
caffeinate -i -w $$ &
PY=astro-env/bin/python
LOG=sbi/logs_k0p7
INC=spin2_cx_ft_
R=sbi/results
mkdir -p $LOG
step() { name=$1; shift; echo "== $(date +%H:%M) $name"; "$@" > $LOG/$name.txt 2>&1 || echo "   FAILED (see $LOG/$name.txt)"; }
while pgrep -f "sbi/k0p7_cx.sh" > /dev/null; do sleep 120; done

# response curves
for t in best final; do
    S=$PWD/sbi/summaries_spin2cx_${t}_n5
    C=$R/centre_uncertainty/tweakedalign_k0p9_ctr_ba_${t}_${INC}.json
    step response_$t $PY sbi/response_plot.py --summaries $S --label "spin2 re-fine-tuned, ba_$t, κ < 0.7" --centre-json $C
    grep -E "bahamas|darkskies" $LOG/response_$t.txt
    out=$R/chifix/response_combined; [ $t = final ] && out=${out}_final
    for nc in 2 3; do
        step response_combined_${t}_nc$nc $PY sbi/response_combined.py --summaries $S --centre-json $C --min-ncomp $nc \
            --label "spin2 re-fine-tuned, ba_$t (30 models), κ < 0.7" --out $out
    done
done
step response_compare $PY sbi/jitter_response.py --title "spin2 re-fine-tuned: best vs final checkpoints, κ < 0.7" \
    --summaries sbi/summaries_spin2cx_best_n5 sbi/summaries_spin2cx_final_n5 --labels "ba_best" "ba_final" \
    --out $R/chifix/response_best_vs_final.png

# E/B swap (ba_best, as before)
step eb_swap $PY sbi/eb_swap_test.py --model tweakedalign_k0p9_ctr_ba_best --include-str $INC \
    --summaries $PWD/sbi/summaries_spin2cx_best_n5 --out-dir $PWD/$R/eb_swap_test_spin2cx

# sensitivity maps
MODEL=tweakedalign_k0p9_ctr_ba_best
out=$R/sensitivity_spin2cx_best_stride1
step sensitivity_best_stride1 $PY sbi/sensitivity.py --model $MODEL --include-str $INC --nmodels 5 --stride 1 \
    --domains --summaries $PWD/sbi/summaries_spin2cx_best_n5 --out-dir $out
step sensitivity_best_stride1_plot $PY sbi/sensitivity.py --a2744-only --model $MODEL --include-str $INC \
    --sigma-range 2 5 --no-excess-contours --out-dir $out
for t in best final; do
    MODEL=tweakedalign_k0p9_ctr_ba_$t
    S=$PWD/sbi/summaries_spin2cx_${t}_n5
    step sensitivity_all_bahamas_$t $PY sbi/sensitivity.py --model $MODEL --include-str $INC --nmodels 5 --min-ncomp 2 \
        --domains bahamas --all-clusters --summaries $S --out-dir $PWD/$R/sensitivity_spin2cx_${t}_all_bahamas
    step sensitivity_w4s1_$t $PY sbi/sensitivity.py --model $MODEL --include-str $INC --nmodels 5 \
        --window 4 --stride 1 --min-ncomp 2 --domains bahamas darkskies --summaries $S \
        --out-dir $PWD/$R/sensitivity_spin2cx_${t}_w4s1
    grep -E "P\(SIDM\)|pushes" $LOG/sensitivity_w4s1_$t.txt | tail -4
done
echo "== $(date +%H:%M) done"
