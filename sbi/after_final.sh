#!/bin/bash
# After rerun_cx_final.sh: the systematics for ba_best and ba_final on the doubled grids with the
# kappa-scaled contamination (cached points reused), then 4x4-window stride-1 sensitivity maps for
# A2744 and 20 clusters per cross-section, best and final. One catalogue reader at a time.
cd "$(dirname "$0")/.."
PY=astro-env/bin/python; LOG=sbi/logs_rerun_cx_final; INC=spin2_cx_ft_; R=sbi/results
step() { name=$1; shift; echo "== $(date +%H:%M) $name"; "$@" > $LOG/$name.txt 2>&1 || echo "   FAILED (see $LOG/$name.txt)"; }
while pgrep -f "sbi/rerun_cx_final.sh" > /dev/null; do sleep 120; done
for t in best final; do
    step systematics_${t}_fine $PY -u sbi/systematics.py --model tweakedalign_k0p9_ctr_ba_$t --include-str $INC \
        --min-ncomp 2 --summaries $PWD/sbi/summaries_spin2cx_${t}_n5 --out-dir $R/systematics_spin2cx_$t
    grep -E "^  (fiducial|contamination)" $LOG/systematics_${t}_fine.txt
done
for t in best final; do
    step sensitivity_stride1_$t $PY sbi/sensitivity.py --model tweakedalign_k0p9_ctr_ba_$t --include-str $INC --nmodels 5 \
        --window 4 --stride 1 --min-ncomp 2 --domains bahamas darkskies --summaries $PWD/sbi/summaries_spin2cx_${t}_n5 \
        --out-dir $PWD/$R/sensitivity_spin2cx_${t}_w4s1
    grep -E "P\(SIDM\)|pushes" $LOG/sensitivity_stride1_$t.txt | tail -4
done
echo "== $(date +%H:%M) done"
