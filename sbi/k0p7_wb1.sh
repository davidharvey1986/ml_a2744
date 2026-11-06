#!/bin/bash
# The fiducial ensembles from 4 Nov: as k0p7_pgc.sh but re-fine-tuned with bahamas_1 kept in the training set
# (run name *_nob1_spin2_cx_pgc_ft_tweakedalign_k0p7_ctr_wb1, checkpoints ..._ctr_wb1_ba_{best,final}.pth, already
# in models/concat). Outputs are named *_wb1_* (summaries_wb1_<t>_n5, results/*_wb1_*, results/wb1/); the nob1
# ones (*_spin2cx_*, results/chifix/) are left as they are. Steps whose output exists are skipped, so the main
# summaries / SBI / response curves made by sbi/pipelines_20261104/wb1.sh are reused. ba_final first.
# Logs to sbi/logs_k0p7_wb1/.
#
#   bash sbi/k0p7_wb1.sh
cd "$(dirname "$0")/.."
caffeinate -is -w $$ &
PY=$PWD/astro-env/bin/python
SBI_PY=../Dark_Matter_DA/sbi/bin/python
LOG=$PWD/sbi/logs_k0p7_wb1
INC=spin2_cx_pgc_ft_
R=sbi/results
mkdir -p $LOG $R/wb1
step() { name=$1; shift; echo "== $(date '+%d %H:%M') $name"; "$@" > $LOG/$name.txt 2>&1 || echo "   FAILED (see $LOG/$name.txt)"; }
have() { [ -e "$1" ] && echo "   $(basename $1) exists, skipping"; }

n=$(ls models/concat/*${INC}tweakedalign_k0p7_ctr_wb1_ba_final.pth | wc -l)
[ "$n" -eq 30 ] || { echo "expected 30 wb1 ba_final checkpoints, found $n; stopping"; exit 1; }

for t in final best; do
    MODEL=tweakedalign_k0p7_ctr_wb1_ba_$t
    S=$PWD/sbi/summaries_wb1_${t}_n5
    F=${S}_flamingo
    D=${S}_dz
    have $S/obs.npz || step summaries_$t $PY -u sbi/make_summaries.py --model $MODEL --include-str $INC --n-noise 5 \
        --no-latents --domains bahamas darkskies --out-dir $S
    step summaries_flamingo_$t $PY -u sbi/make_summaries.py --model $MODEL --include-str $INC --n-noise 5 --no-latents \
        --domains flamingo flamingo_lo flamingo_hi tng vd_bahamas --out-dir $F
    step summaries_dz_$t $PY -u sbi/make_summaries.py --model $MODEL --include-str $INC --n-noise 5 --no-latents \
        --domains flamingo_dz25 flamingo_dz50 flamingo_dz100 flamingo_dz200 flamingo_dz250 flamingo_dz500 --out-dir $D
    C=$R/centre_uncertainty/${MODEL}_${INC}.json
    have $C || step centre_uncertainty_$t $PY sbi/centre_uncertainty.py --model $MODEL --include-str $INC
    step cdm_test_$t $PY sbi/flamingo_test.py --summaries $S --flamingo $F --label "wb1 ba_$t" --min-ncomp 3 \
        --out $F/flamingo_test_nc3_kde
    grep "n_comp" $LOG/cdm_test_$t.txt

    B=$R/summaries_wb1_${t}_n5_alldraws
    if ! have ${B}_nc4plus/results.json; then
        step sbi_${t}_nc2plus $SBI_PY -u sbi/run_sbi.py --summaries $S --noise all --a2744-ncomp 2+ --retrain --out-dir ${B}_nc2plus
        for nc in 3 3+ 4+; do
            d=${B}_nc${nc/+/plus}; mkdir -p "$d"; cp ${B}_nc2plus/nle_*.pt ${B}_nc2plus/npe_conditional_*.pt "$d"/
            step sbi_${t}_nc${nc/+/plus} $SBI_PY -u sbi/run_sbi.py --summaries $S --noise all --a2744-ncomp "$nc" --out-dir "$d"
        done
        out=$R/wb1/response_combined; [ $t = final ] && out=${out}_final
        step response_$t $PY sbi/response_plot.py --summaries $S --label "wb1 κ<0.7 per-galaxy χ, ba_$t" --centre-json $C
        for nc in 2 3; do
            step response_combined_${t}_nc$nc $PY sbi/response_combined.py --summaries $S --centre-json $C --min-ncomp $nc \
                --label "wb1 κ<0.7 per-galaxy χ, ba_$t (30 models)" --out $out
        done
    fi

    step sensitivity_w4s1_$t $PY sbi/sensitivity.py --model $MODEL --include-str $INC --nmodels 5 \
        --window 4 --stride 1 --min-ncomp 2 --domains bahamas darkskies --summaries $S \
        --out-dir $PWD/$R/sensitivity_wb1_${t}_w4s1
    grep -E "P\(SIDM\)|pushes" $LOG/sensitivity_w4s1_$t.txt | tail -4
    step sensitivity_all_bahamas_$t $PY sbi/sensitivity.py --model $MODEL --include-str $INC --nmodels 5 --min-ncomp 2 \
        --domains bahamas --all-clusters --summaries $S --out-dir $PWD/$R/sensitivity_wb1_${t}_all_bahamas

    step systematics_$t $PY -u sbi/systematics.py --model $MODEL --include-str $INC --min-ncomp 2 --summaries $S \
        --out-dir $R/systematics_wb1_$t --dz-summaries $D
    grep -E "^  (fiducial|kappa_cut|shape|merging)" $LOG/systematics_$t.txt
done

step summaries_latents_final $PY -u sbi/make_summaries.py --model tweakedalign_k0p7_ctr_wb1_ba_final --include-str $INC \
    --n-noise 1 --domains bahamas darkskies --out-dir $PWD/sbi/summaries_wb1_final_n5_latents
have $R/wb1/response_best_vs_final.png || step response_compare $PY sbi/jitter_response.py \
    --title "wb1 κ<0.7 per-galaxy χ: best vs final checkpoints" \
    --summaries sbi/summaries_wb1_best_n5 sbi/summaries_wb1_final_n5 --labels "ba_best" "ba_final" \
    --out $R/wb1/response_best_vs_final.png
step eb_swap $PY sbi/eb_swap_test.py --model tweakedalign_k0p7_ctr_wb1_ba_best --include-str $INC \
    --summaries $PWD/sbi/summaries_wb1_best_n5 --out-dir $PWD/$R/eb_swap_test_wb1
MODEL=tweakedalign_k0p7_ctr_wb1_ba_best; out=$R/sensitivity_wb1_best_stride1
step sensitivity_best_stride1 $PY sbi/sensitivity.py --model $MODEL --include-str $INC --nmodels 5 --stride 1 \
    --domains --summaries $PWD/sbi/summaries_wb1_best_n5 --out-dir $out
step sensitivity_best_stride1_plot $PY sbi/sensitivity.py --a2744-only --model $MODEL --include-str $INC \
    --sigma-range 2 5 --no-excess-contours --out-dir $out

# these write to fixed paths: keep the nob1 versions
A=$R/archive_nob1_fiducial_20261104; mkdir -p $A
[ -e $A/contamination_profile ] || cp -a $R/contamination_profile $R/main_halo_profile $A/
[ -e notebooks/plots_nob1_20261104 ] || cp -a notebooks/plots notebooks/plots_nob1_20261104
step p_vs_mass $PY sbi/p_vs_mass.py --fiducial wb1 --out $R/wb1/p_vs_mass
step contamination_profile $PY sbi/contamination_profile.py --summaries $PWD/sbi/summaries_wb1_best_n5
step main_halo_profile $PY sbi/main_halo_profile.py --dx-kpc 0 --dy-kpc 0 --summaries $PWD/sbi/summaries_wb1_best_n5 \
    --centre-json $R/centre_uncertainty/tweakedalign_k0p7_ctr_wb1_ba_best_${INC}.json
cd notebooks
step paper_figures $PY plots_for_paper.py
grep -E "Figure|Saved|sigma|rho|Discrepancy" $LOG/paper_figures.txt
echo "== $(date '+%d %H:%M') done"
