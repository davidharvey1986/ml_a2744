#!/bin/bash
# As evaluate_cx.sh + rerun_cx.sh for the final checkpoints (*spin2_cx_ft_*_ba_final, 30 seeds):
# summaries, centring, response curves, SBI, systematics (with the kappa cut and centring) and the
# all-BAHAMAS sensitivity maps. Starts with the ba_best systematics on the doubled grids. Waits for rerun_cx.sh to finish (get_obs_data rewrites the A2744
# filtered fits). Logs to sbi/logs_rerun_cx_final/<step>.txt.
#
#   bash sbi/rerun_cx_final.sh
cd "$(dirname "$0")/.."
PY=astro-env/bin/python
SBI_PY=../Dark_Matter_DA/sbi/bin/python
LOG=sbi/logs_rerun_cx_final
INC=spin2_cx_ft_
MODEL=tweakedalign_k0p9_ctr_ba_final
S=$PWD/sbi/summaries_spin2cx_final_n5
R=sbi/results
mkdir -p $LOG
step() { name=$1; shift; echo "== $(date +%H:%M) $name"; "$@" > $LOG/$name.txt 2>&1 || echo "   FAILED (see $LOG/$name.txt)"; }

while pgrep -f "sbi/rerun_cx.sh" > /dev/null; do sleep 120; done

# ba_best systematics on the doubled grids (cached points reused)
step systematics_best_fine $PY -u sbi/systematics.py --model tweakedalign_k0p9_ctr_ba_best --include-str $INC \
    --min-ncomp 2 --summaries $PWD/sbi/summaries_spin2cx_best_n5 --out-dir $R/systematics_spin2cx_best

step summaries $PY sbi/make_summaries.py --model $MODEL --include-str $INC \
    --n-noise 5 --domains bahamas darkskies --no-latents --out-dir $S
step centre_uncertainty $PY sbi/centre_uncertainty.py --model $MODEL --include-str $INC
C=sbi/results/centre_uncertainty/${MODEL}_${INC}.json
step response $PY sbi/response_plot.py --summaries $S --label "spin2 re-fine-tuned, final" --centre-json $C
grep -E "bahamas|darkskies" $LOG/response.txt
for nc in 2 3; do
    step response_combined_nc$nc $PY sbi/response_combined.py --summaries $S --centre-json $C --min-ncomp $nc \
        --label "spin2 re-fine-tuned, final (30 models)" --out $R/chifix/response_combined_final
done
step response_compare $PY sbi/jitter_response.py --title "spin2 re-fine-tuned: best vs final checkpoints" \
    --summaries sbi/summaries_spin2cx_best_n5 $S --labels "ba_best" "ba_final" \
    --out $R/chifix/response_best_vs_final.png

B=$R/summaries_spin2cx_final_n5_alldraws
step sbi_nc2plus $SBI_PY -u sbi/run_sbi.py --summaries $S --noise all --a2744-ncomp 2+ --retrain --out-dir ${B}_nc2plus
for nc in 3 3+ 4+; do
    d=${B}_nc${nc/+/plus}; mkdir -p "$d"; cp ${B}_nc2plus/nle_*.pt ${B}_nc2plus/npe_conditional_*.pt "$d"/
    step sbi_nc${nc/+/plus} $SBI_PY -u sbi/run_sbi.py --summaries $S --noise all --a2744-ncomp "$nc" --out-dir "$d"
done
grep -h -E "NLE:|conditional|A2744" $LOG/sbi_*.txt | grep -v -i warn

step systematics $PY -u sbi/systematics.py --model $MODEL --include-str $INC --min-ncomp 2 --summaries $S \
    --out-dir $R/systematics_spin2cx_final --redo
grep -E "^  (fiducial|kappa_cut|centring)" $LOG/systematics.txt

out=$R/sensitivity_spin2cx_final_all_bahamas
step sensitivity_all_bahamas $PY sbi/sensitivity.py --model $MODEL --include-str $INC --nmodels 5 --min-ncomp 2 \
    --domains bahamas --all-clusters --summaries $S --out-dir $PWD/$out
grep -E "P\(SIDM\)|pushes|clusters" $LOG/sensitivity_all_bahamas.txt | tail -12
echo "== $(date +%H:%M) done"
