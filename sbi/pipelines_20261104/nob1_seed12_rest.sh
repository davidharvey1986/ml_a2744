#!/bin/bash
# Rest of the seed-12 rerun (interrupted 17:25 in summaries_dz_final; summaries_final and _flamingo_final are done):
# dz_final, all ba_best summaries, then centre uncertainty / SBI / response for both. Logs to sbi/logs_k0p7_pgc_rerun/.
source $(dirname $0)/common.sh
LOG=$PWD/sbi/logs_k0p7_pgc_rerun
INC=spin2_cx_pgc_ft_
DZ="flamingo_dz25 flamingo_dz50 flamingo_dz100 flamingo_dz200 flamingo_dz250 flamingo_dz500"
summaries summaries_dz_final tweakedalign_k0p7_ctr_ba_final $INC $PWD/sbi/summaries_spin2cx_final_n5_dz $DZ
S=$PWD/sbi/summaries_spin2cx_best_n5; M=tweakedalign_k0p7_ctr_ba_best
summaries summaries_best $M $INC $S bahamas darkskies
grep -h "A2744: ensemble" $LOG/summaries_best.txt | tr '\r' '\n' | grep A2744
summaries summaries_flamingo_best $M $INC ${S}_flamingo flamingo flamingo_lo flamingo_hi tng vd_bahamas
summaries summaries_dz_best $M $INC ${S}_dz $DZ
A=$R/archive_pre_seed12_20261104; mkdir -p $A
cp -a $R/summaries_spin2cx_*_n5_alldraws_nc* $R/chifix $R/centre_uncertainty $A/
for t in final best; do
    out=$R/chifix/response_combined; [ $t = final ] && out=${out}_final
    sbi_response $t tweakedalign_k0p7_ctr_ba_$t $INC $PWD/sbi/summaries_spin2cx_${t}_n5 $R/summaries_spin2cx_${t}_n5_alldraws $out \
        "spin2 κ<0.7 per-galaxy χ"
done
step response_compare $PY sbi/jitter_response.py --title "spin2 κ<0.7 per-galaxy χ: best vs final checkpoints" \
    --summaries sbi/summaries_spin2cx_best_n5 sbi/summaries_spin2cx_final_n5 --labels "ba_best" "ba_final" \
    --out $R/chifix/response_best_vs_final.png
echo "== $(date '+%d %H:%M') nob1 done"
