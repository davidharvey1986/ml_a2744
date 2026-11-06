#!/bin/bash
# wb1 (bahamas_1 in fine-tuning) vs nob1: response curves overlaid, per-seed A2744 P(SIDM), SBI table -> sbi/results/wb1/
source $(dirname $0)/common.sh
LOG=$PWD/sbi/logs_wb1
for t in final best; do
    step compare_response_$t $PY sbi/jitter_response.py --title "κ<0.7 per-galaxy χ, ba_$t: without vs with bahamas_1" \
        --summaries sbi/summaries_spin2cx_${t}_n5 sbi/summaries_wb1_${t}_n5 --labels "nob1" "wb1" \
        --out $R/wb1/response_nob1_vs_wb1_$t.png
done
$PY sbi/pipelines_20261104/compare_wb1_nob1.py | tee $R/wb1/compare_nob1_wb1.txt
