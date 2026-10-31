#!/bin/bash
# Velocity-dependent SIDM BAHAMAS (vd_bahamas.pkl, vdSIDM+baryons; never trained on) through the spin2
# re-fine-tuned models: sample the maps on the A2744 galaxies (-> data/100/obs/concat/vd_bahamas.pkl),
# summaries for ba_best and ba_final (into the FLAMINGO summaries directories), then the CDM test
# (n_comp >= 3, KDEs) with vdSIDM next to BAHAMAS sigma/m = 0.3 and 1. Keeps the Mac awake.
cd "$(dirname "$0")/.."
caffeinate -i -w $$ &
PY=astro-env/bin/python; LOG=sbi/logs_rerun_cx_final; INC=spin2_cx_ft_
step() { name=$1; shift; echo "== $(date +%H:%M) $name"; "$@" > $LOG/$name.txt 2>&1 || echo "   FAILED (see $LOG/$name.txt)"; }
step vd_obs_maps $PY -c "from add_shear_to_data import main; main(search_path='data/100/convergence/vd_bahamas.pkl', sample_data=True, data_dir='data/100/a2744')"
for t in best final; do
    F=sbi/summaries_spin2cx_${t}_n5_flamingo
    step vd_summaries_$t $PY -u sbi/make_summaries.py --model tweakedalign_k0p9_ctr_ba_$t --include-str $INC \
        --n-noise 5 --no-latents --domains vd_bahamas --out-dir $PWD/$F
    step vd_test_$t $PY sbi/flamingo_test.py --summaries sbi/summaries_spin2cx_${t}_n5 --flamingo $F \
        --label ba_$t --min-ncomp 3 --out $PWD/$F/flamingo_test_nc3_kde
    grep "n_comp" $LOG/vd_test_$t.txt
done
echo "== $(date +%H:%M) done"
