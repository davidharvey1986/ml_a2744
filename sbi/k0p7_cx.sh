#!/bin/bash
# Strong-lensing cut kappa < 0.7 (add_shear_to_data.KAPPA_THRESH) for the existing k0p9-trained ensembles,
# ba_best and ba_final: the A2744 map and every simulated map rebuilt, then summaries (BAHAMAS/DARKSKIES,
# FLAMINGO/TNG/vdSIDM, FLAMINGO depths), centre uncertainty, SBI, the CDM test, the systematics and the
# profile and mass plots. The kappa < 0.9 versions are in archive_k0p9/. One catalogue reader at a time.
# Logs to sbi/logs_k0p7/<step>.txt.
#
#   bash sbi/k0p7_cx.sh
cd "$(dirname "$0")/.."
caffeinate -i -w $$ &
PY=astro-env/bin/python
SBI_PY=../Dark_Matter_DA/sbi/bin/python
LOG=sbi/logs_k0p7
INC=spin2_cx_ft_
R=sbi/results
mkdir -p $LOG
step() { name=$1; shift; echo "== $(date +%H:%M) $name"; "$@" > $LOG/$name.txt 2>&1 || echo "   FAILED (see $LOG/$name.txt)"; }

step obs_map $PY -c "from add_shear_to_data import make_obs_maps; make_obs_maps()"
step sim_maps $PY -c "from add_shear_to_data import main; main(search_path='data/100/convergence/*.pkl', sample_data=True, data_dir='data/100/a2744')"

for t in best final; do
    MODEL=tweakedalign_k0p9_ctr_ba_$t
    S=$PWD/sbi/summaries_spin2cx_${t}_n5
    F=$PWD/sbi/summaries_spin2cx_${t}_n5_flamingo
    D=$PWD/sbi/summaries_spin2cx_${t}_n5_dz
    step summaries_$t $PY -u sbi/make_summaries.py --model $MODEL --include-str $INC --n-noise 5 --no-latents \
        --domains bahamas darkskies --out-dir $S
    step summaries_flamingo_$t $PY -u sbi/make_summaries.py --model $MODEL --include-str $INC --n-noise 5 --no-latents \
        --domains flamingo flamingo_lo flamingo_hi tng vd_bahamas --out-dir $F
    step summaries_dz_$t $PY -u sbi/make_summaries.py --model $MODEL --include-str $INC --n-noise 5 --no-latents \
        --domains flamingo_dz25 flamingo_dz50 flamingo_dz100 flamingo_dz200 flamingo_dz250 flamingo_dz500 --out-dir $D
    grep -h "A2744: ensemble" $LOG/summaries_$t.txt | tr '\r' '\n' | grep A2744
    step centre_uncertainty_$t $PY sbi/centre_uncertainty.py --model $MODEL --include-str $INC
    step cdm_test_$t $PY sbi/flamingo_test.py --summaries $S --flamingo $F --label ba_$t --min-ncomp 3 \
        --out $F/flamingo_test_nc3_kde
    grep "n_comp" $LOG/cdm_test_$t.txt

    B=$R/summaries_spin2cx_${t}_n5_alldraws
    step sbi_${t}_nc2plus $SBI_PY -u sbi/run_sbi.py --summaries $S --noise all --a2744-ncomp 2+ --retrain --out-dir ${B}_nc2plus
    for nc in 3 3+ 4+; do
        d=${B}_nc${nc/+/plus}; mkdir -p "$d"; cp ${B}_nc2plus/nle_*.pt ${B}_nc2plus/npe_conditional_*.pt "$d"/
        step sbi_${t}_nc${nc/+/plus} $SBI_PY -u sbi/run_sbi.py --summaries $S --noise all --a2744-ncomp "$nc" --out-dir "$d"
    done
    grep -h -E "NLE:|conditional|A2744" $LOG/sbi_${t}_*.txt | grep -v -i warn

    step systematics_$t $PY -u sbi/systematics.py --model $MODEL --include-str $INC --min-ncomp 2 --summaries $S \
        --out-dir $R/systematics_spin2cx_$t --dz-summaries $D
    grep -E "^  (fiducial|kappa_cut|shape|merging)" $LOG/systematics_$t.txt
done

step p_vs_mass $PY sbi/p_vs_mass.py
step contamination_profile $PY sbi/contamination_profile.py
step main_halo_profile $PY sbi/main_halo_profile.py --dx-kpc 0 --dy-kpc 0 --summaries $PWD/sbi/summaries_spin2cx_best_n5
echo "== $(date +%H:%M) done"
