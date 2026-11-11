#!/bin/bash
# 4x4 stride-1 occlusion sensitivity of all 29 pairs of a2744_analogue_pairs_nc4_all.json (existing outputs skipped)
cd /Users/davidharvey/Work/ml_check
for n in $(seq 1 29); do
    [ -e sbi/results/pair${n}_sensitivity_w4s1.png ] && { echo "pair $n: exists"; continue; }
    echo "== $(date '+%H:%M') pair $n"
    astro-env/bin/python -u sbi/pair_sensitivity.py --pair $n --window 4 --stride 1 --draws 3 2>&1 \
        | grep -v "DEFAULT ZS\|Source redshift\|WARN\|RADECSYS\|it/s\]" > sbi/logs_pair_sensitivity/pair${n}_w4s1.txt
    grep -E "P\(SIDM\)|Traceback|Error" sbi/logs_pair_sensitivity/pair${n}_w4s1.txt
done
echo "== $(date '+%H:%M') all done"
