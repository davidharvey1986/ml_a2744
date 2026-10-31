#!/bin/bash
# Cluster-member photo-z cut (0.40 ... 0.80) added to the data systematics for ba_best and ba_final;
# every other test is read from cache. Waits for tng_cx.sh (get_obs_data rewrites the A2744 filtered
# fits). Logs to sbi/logs_rerun_cx_final/systematics_<tag>_zcut.txt.
cd "$(dirname "$0")/.."
caffeinate -i -w $$ &
while pgrep -f "sbi/tng_cx.sh" > /dev/null; do sleep 120; done
PY=astro-env/bin/python; LOG=sbi/logs_rerun_cx_final; INC=spin2_cx_ft_; R=sbi/results
step() { name=$1; shift; echo "== $(date +%H:%M) $name"; "$@" > $LOG/$name.txt 2>&1 || echo "   FAILED (see $LOG/$name.txt)"; }
for t in best final; do
    step systematics_${t}_zcut $PY -u sbi/systematics.py --model tweakedalign_k0p9_ctr_ba_$t --include-str $INC \
        --min-ncomp 2 --summaries $PWD/sbi/summaries_spin2cx_${t}_n5 --out-dir $R/systematics_spin2cx_$t
    grep -E "^  (fiducial|galaxy_redshift)" $LOG/systematics_${t}_zcut.txt
done
echo "== $(date +%H:%M) done"
