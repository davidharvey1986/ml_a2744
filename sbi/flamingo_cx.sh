#!/bin/bash
# FLAMINGO (fiducial, low and high AGN) through the spin2 re-fine-tuned models, ba_best and ba_final,
# then the CDM test against A2744 (flamingo_test.py). Waits for after_final.sh (get_obs_data rewrites
# the A2744 filtered fits). Logs to sbi/logs_rerun_cx_final/flamingo_<tag>.txt.
cd "$(dirname "$0")/.."
PY=astro-env/bin/python; LOG=sbi/logs_rerun_cx_final; INC=spin2_cx_ft_
step() { name=$1; shift; echo "== $(date +%H:%M) $name"; "$@" > $LOG/$name.txt 2>&1 || echo "   FAILED (see $LOG/$name.txt)"; }
while pgrep -f "sbi/after_final.sh" > /dev/null; do sleep 120; done
for t in best final; do
    F=sbi/summaries_spin2cx_${t}_n5_flamingo
    step flamingo_summaries_$t $PY -u sbi/make_summaries.py --model tweakedalign_k0p9_ctr_ba_$t --include-str $INC \
        --n-noise 5 --no-latents --domains flamingo flamingo_lo flamingo_hi --out-dir $PWD/$F
    step flamingo_test_$t $PY sbi/flamingo_test.py --summaries sbi/summaries_spin2cx_${t}_n5 --flamingo $F \
        --label "spin2 re-fine-tuned, ba_$t"
    grep "n_comp" $LOG/flamingo_test_$t.txt
done
echo "== $(date +%H:%M) done"
