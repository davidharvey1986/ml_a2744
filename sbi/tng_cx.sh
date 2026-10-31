#!/bin/bash
# TNG-Cluster through the spin2 re-fine-tuned models: sample the convergence maps on the A2744
# galaxies (add_shear_to_data.main -> data/100/obs/concat/tng.pkl), summaries for ba_best and
# ba_final (into the FLAMINGO summaries directories), then the CDM test with every simulation.
# Waits for flamingo_cx.sh (one catalogue reader at a time); keeps the Mac awake until done.
cd "$(dirname "$0")/.."
caffeinate -i -w $$ &
PY=astro-env/bin/python; LOG=sbi/logs_rerun_cx_final; INC=spin2_cx_ft_
step() { name=$1; shift; echo "== $(date +%H:%M) $name"; "$@" > $LOG/$name.txt 2>&1 || echo "   FAILED (see $LOG/$name.txt)"; }
while pgrep -f "sbi/flamingo_cx.sh" > /dev/null; do sleep 120; done
step tng_obs_maps $PY -c "from add_shear_to_data import main; main(search_path='data/100/convergence/tng.pkl', sample_data=True, data_dir='data/100/a2744')"
for t in best final; do
    F=sbi/summaries_spin2cx_${t}_n5_flamingo
    step tng_summaries_$t $PY -u sbi/make_summaries.py --model tweakedalign_k0p9_ctr_ba_$t --include-str $INC \
        --n-noise 5 --no-latents --domains tng --out-dir $PWD/$F
    step cdm_test_$t $PY sbi/flamingo_test.py --summaries sbi/summaries_spin2cx_${t}_n5 --flamingo $F \
        --label "spin2 re-fine-tuned, ba_$t"
    grep "n_comp" $LOG/cdm_test_$t.txt
done
echo "== $(date +%H:%M) done"
