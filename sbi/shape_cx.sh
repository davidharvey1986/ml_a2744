#!/bin/bash
# Shape-measurement bias, m and c each from -0.05 to 0.05 on e1 and e2 together, for ba_best and ba_final;
# every other test is read from cache. Logs to sbi/logs_rerun_cx_final/systematics_<tag>_shape.txt.
cd "$(dirname "$0")/.."
caffeinate -i -w $$ &
PY=astro-env/bin/python; LOG=sbi/logs_rerun_cx_final; INC=spin2_cx_ft_; R=sbi/results
step() { name=$1; shift; echo "== $(date +%H:%M) $name"; "$@" > $LOG/$name.txt 2>&1 || echo "   FAILED (see $LOG/$name.txt)"; }
for t in best final; do
    step systematics_${t}_shape $PY -u sbi/systematics.py --model tweakedalign_k0p9_ctr_ba_$t --include-str $INC \
        --min-ncomp 2 --summaries $PWD/sbi/summaries_spin2cx_${t}_n5 --out-dir $R/systematics_spin2cx_$t \
        --dz-summaries $PWD/sbi/summaries_spin2cx_${t}_n5_dz
    grep -E "^  (fiducial|shape)" $LOG/systematics_${t}_shape.txt
done
echo "== $(date +%H:%M) done"
