#!/bin/bash
# wb1 ensembles (bahamas_1 kept in fine-tuning, *_spin2_cx_pgc_ft_tweakedalign_k0p7_ctr_wb1_ba_*): wait for bonsai,
# copy the checkpoints, summaries -> centre uncertainty -> SBI -> response. Logs to sbi/logs_wb1/.
source $(dirname $0)/common.sh
LOG=$PWD/sbi/logs_wb1; mkdir -p $LOG
REMOTE=/srv/scratch/dharvey/ml_nature/models/concat
PAT="*_spin2_cx_pgc_ft_tweakedalign_k0p7_ctr_wb1_ba"
INC=spin2_cx_pgc_ft_
while true; do
    n=$(ssh -o BatchMode=yes bonsai "ls $REMOTE/${PAT}_final.pth 2>/dev/null | wc -l" 2>/dev/null)
    q=$(ssh -o BatchMode=yes bonsai "squeue -u dharvey -h -n incr_learn | wc -l" 2>/dev/null)
    [ -n "$n" ] && [ -n "$q" ] && [ "$q" -eq 0 ] && break
    sleep 300
done
echo "== $(date '+%d %H:%M') bonsai: $n wb1 ba_final checkpoints, no incr_learn jobs left"
[ "$n" -eq 30 ] || { echo "   expected 30: check the bonsai logs; stopping"; exit 1; }
rsync -a --include="${PAT}_best.pth" --include="${PAT}_final.pth" --exclude='*' bonsai:$REMOTE/ models/concat/ 2>/dev/null
echo "   copied: $(ls models/concat/${PAT}_*.pth | wc -l) checkpoints"
mkdir -p $R/wb1
for t in final best; do
    MODEL=tweakedalign_k0p7_ctr_wb1_ba_$t
    S=$PWD/sbi/summaries_wb1_${t}_n5
    summaries summaries_$t $MODEL $INC $S bahamas darkskies
    grep -h "A2744: ensemble" $LOG/summaries_$t.txt | tr '\r' '\n' | grep A2744
    out=$R/wb1/response_combined; [ $t = final ] && out=${out}_final
    sbi_response $t $MODEL $INC $S $R/summaries_wb1_${t}_n5_alldraws $out "wb1 κ<0.7 per-galaxy χ"
done
step response_compare $PY sbi/jitter_response.py --title "wb1 κ<0.7 per-galaxy χ: best vs final checkpoints" \
    --summaries sbi/summaries_wb1_best_n5 sbi/summaries_wb1_final_n5 --labels "ba_best" "ba_final" \
    --out $R/wb1/response_best_vs_final.png
echo "== $(date '+%d %H:%M') wb1 done"
