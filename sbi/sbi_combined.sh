#!/bin/bash
# Add the combined BAHAMAS+DARKSKIES SBI to the results for TAG (best or final: bash sbi/sbi_combined.sh final) (per-simulation estimators reused,
# combined trained once on n_comp 2+ and copied to the other conditionings).
cd "$(dirname "$0")/.."; TAG=${1:-best}
SBI_PY=../Dark_Matter_DA/sbi/bin/python; LOG=sbi/logs_rerun_cx; S=sbi/summaries_spin2cx_${TAG}_n5
B=sbi/results/summaries_spin2cx_${TAG}_n5_alldraws
$SBI_PY -u sbi/run_sbi.py --summaries $S --noise all --a2744-ncomp 2+ --out-dir ${B}_nc2plus > $LOG/sbi_comb_${TAG}_nc2plus.txt 2>&1
for nc in 3 3+ 4+; do d=${B}_nc${nc/+/plus}; cp ${B}_nc2plus/nle_combined.pt ${B}_nc2plus/npe_conditional_combined.pt "$d"/
    $SBI_PY -u sbi/run_sbi.py --summaries $S --noise all --a2744-ncomp "$nc" --out-dir "$d" > $LOG/sbi_comb_${TAG}_nc${nc/+/plus}.txt 2>&1; done
grep -h -E "===|empirical:|NLE:|conditional on|within sim" $LOG/sbi_comb_${TAG}_*.txt | grep -v -i warn
