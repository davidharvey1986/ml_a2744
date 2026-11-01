#!/bin/bash
# FLAMINGO projection-depth test (flamingo_dz{25,50,100,200,250,500}.pkl: the same haloes projected
# through 25-500 cMpc, old map format; background from the depth, add_shear_to_data.get_boxsize):
# sample on the A2744 galaxies, summaries for ba_best and ba_final into sbi/summaries_spin2cx_<t>_n5_dz.
cd "$(dirname "$0")/.."
caffeinate -i -w $$ &
PY=astro-env/bin/python; LOG=sbi/logs_rerun_cx_final; INC=spin2_cx_ft_
step() { name=$1; shift; echo "== $(date +%H:%M) $name"; "$@" > $LOG/$name.txt 2>&1 || echo "   FAILED (see $LOG/$name.txt)"; }
step dz_obs_maps $PY -c "from add_shear_to_data import main; main(search_path='data/100/convergence/flamingo_dz*.pkl', sample_data=True, data_dir='data/100/a2744')"
for t in best final; do
    step dz_summaries_$t $PY -u sbi/make_summaries.py --model tweakedalign_k0p9_ctr_ba_$t --include-str $INC \
        --n-noise 5 --no-latents --domains flamingo_dz25 flamingo_dz50 flamingo_dz100 flamingo_dz200 flamingo_dz250 flamingo_dz500 \
        --out-dir $PWD/sbi/summaries_spin2cx_${t}_n5_dz
done
echo "== $(date +%H:%M) done"
