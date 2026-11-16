#!/bin/bash
# Every number and figure in the A2744 paper from the trained CNNs (the wb1 fiducial: run name
# *_nob1_spin2_cx_pgc_ft_tweakedalign_k0p7_ctr_wb1, checkpoints ..._ctr_wb1_ba_{best,final}.pth in models/concat,
# A2744 mass from mass_weighting.FIDUCIAL_STUDY = jauzac). ba_final is the paper's ensemble; ba_best is redone as a
# check. Outputs go to notebooks/results/ (summaries_wb1_<t>_n5*, *_wb1_*, wb1/); steps whose output exists are
# skipped. Logs to notebooks/logs/paper_pipeline/. The figures are made last by paper_figures.py.
#
# Was sbi/k0p7_wb1.sh (+ sbi/pipelines_20261104/jauzac_systematics.sh) before the sbi/ folder was folded into
# notebooks/ on 16 Nov 2026.
#
#   bash notebooks/paper_pipeline.sh
cd "$(dirname "$0")/.."
caffeinate -is -w $$ &
PY=$PWD/astro-env/bin/python
NB=notebooks
R=$NB/results
LOG=$PWD/$NB/logs/paper_pipeline
INC=spin2_cx_pgc_ft_
STUDY=jauzac
mkdir -p $LOG $R/wb1
step() { name=$1; shift; echo "== $(date '+%d %H:%M') $name"; "$@" > $LOG/$name.txt 2>&1 || echo "   FAILED (see $LOG/$name.txt)"; }
have() { [ -e "$1" ] && echo "   $(basename $1) exists, skipping"; }

n=$(ls models/concat/*${INC}tweakedalign_k0p7_ctr_wb1_ba_final.pth | wc -l)
[ "$n" -eq 30 ] || { echo "expected 30 wb1 ba_final checkpoints, found $n; stopping"; exit 1; }

for t in final best; do
    MODEL=tweakedalign_k0p7_ctr_wb1_ba_$t
    S=$PWD/$R/summaries_wb1_${t}_n5
    F=${S}_flamingo
    D=${S}_dz
    have $S/obs.npz || step summaries_$t $PY -u $NB/make_summaries.py --model $MODEL --include-str $INC --n-noise 5 \
        --no-latents --domains bahamas darkskies --out-dir $S
    have $F/obs.npz || step summaries_flamingo_$t $PY -u $NB/make_summaries.py --model $MODEL --include-str $INC \
        --n-noise 5 --no-latents --domains flamingo flamingo_lo flamingo_hi tng vd_bahamas --out-dir $F
    have $D/obs.npz || step summaries_dz_$t $PY -u $NB/make_summaries.py --model $MODEL --include-str $INC --n-noise 5 \
        --no-latents --domains flamingo_dz25 flamingo_dz50 flamingo_dz100 flamingo_dz200 flamingo_dz250 flamingo_dz500 \
        --out-dir $D
    C=$R/centre_uncertainty/${MODEL}_${INC}.json
    have $C || step centre_uncertainty_$t $PY $NB/centre_uncertainty.py --model $MODEL --include-str $INC
    step cdm_test_$t $PY $NB/flamingo_test.py --summaries $S --flamingo $F --label "wb1 ba_$t" --min-ncomp 3 \
        --out $F/flamingo_test_nc3_kde
    grep "n_comp" $LOG/cdm_test_$t.txt

    B=$R/summaries_wb1_${t}_n5_alldraws
    if ! have ${B}_nc4plus/results.json; then
        step sbi_${t}_nc2plus $PY -u $NB/run_sbi.py --summaries $S --noise all --a2744-ncomp 2+ --retrain --out-dir ${B}_nc2plus
        for nc in 3 3+ 4+; do
            d=${B}_nc${nc/+/plus}; mkdir -p "$d"; cp ${B}_nc2plus/nle_*.pt ${B}_nc2plus/npe_conditional_*.pt "$d"/
            step sbi_${t}_nc${nc/+/plus} $PY -u $NB/run_sbi.py --summaries $S --noise all --a2744-ncomp "$nc" --out-dir "$d"
        done
        out=$R/wb1/response_combined; [ $t = final ] && out=${out}_final
        step response_$t $PY $NB/response_plot.py --summaries $S --label "wb1 κ<0.7 per-galaxy χ, ba_$t" --centre-json $C
        for nc in 2 3; do
            step response_combined_${t}_nc$nc $PY $NB/response_combined.py --summaries $S --centre-json $C --min-ncomp $nc \
                --label "wb1 κ<0.7 per-galaxy χ, ba_$t (30 models)" --out $out
        done
    fi

    have $PWD/$R/sensitivity_wb1_${t}_w4s1/sensitivity.npz || step sensitivity_w4s1_$t $PY $NB/sensitivity.py \
        --model $MODEL --include-str $INC --nmodels 5 --window 4 --stride 1 --min-ncomp 2 --domains bahamas darkskies \
        --summaries $S --out-dir $PWD/$R/sensitivity_wb1_${t}_w4s1
    have $PWD/$R/sensitivity_wb1_${t}_all_bahamas || step sensitivity_all_bahamas_$t $PY $NB/sensitivity.py \
        --model $MODEL --include-str $INC --nmodels 5 --min-ncomp 2 --domains bahamas --all-clusters --summaries $S \
        --out-dir $PWD/$R/sensitivity_wb1_${t}_all_bahamas

    # results are cached per test, so an interrupted run resumes
    step systematics_$t $PY -u $NB/systematics.py --model $MODEL --include-str $INC --min-ncomp 2 --summaries $S \
        --out-dir $R/systematics_wb1_${t}_$STUDY --dz-summaries $D --mass-study $STUDY
    grep -E "clusters simulated|^  (fiducial|kappa_cut|shape|merging)" $LOG/systematics_$t.txt
done

have $PWD/$R/summaries_wb1_final_n5_latents/obs.npz || step summaries_latents_final $PY -u $NB/make_summaries.py \
    --model tweakedalign_k0p7_ctr_wb1_ba_final --include-str $INC --n-noise 1 --domains bahamas darkskies \
    --out-dir $PWD/$R/summaries_wb1_final_n5_latents
have $R/wb1/response_best_vs_final.png || step response_compare $PY $NB/jitter_response.py \
    --title "wb1 κ<0.7 per-galaxy χ: best vs final checkpoints" \
    --summaries $R/summaries_wb1_best_n5 $R/summaries_wb1_final_n5 --labels "ba_best" "ba_final" \
    --out $R/wb1/response_best_vs_final.png
step eb_swap $PY $NB/eb_swap_test.py --model tweakedalign_k0p7_ctr_wb1_ba_best --include-str $INC \
    --summaries $PWD/$R/summaries_wb1_best_n5 --out-dir $PWD/$R/eb_swap_test_wb1
MODEL=tweakedalign_k0p7_ctr_wb1_ba_best; out=$R/sensitivity_wb1_best_stride1
have $out/sensitivity.npz || step sensitivity_best_stride1 $PY $NB/sensitivity.py --model $MODEL --include-str $INC \
    --nmodels 5 --stride 1 --domains --summaries $PWD/$R/summaries_wb1_best_n5 --out-dir $out
step sensitivity_best_stride1_plot $PY $NB/sensitivity.py --a2744-only --model $MODEL --include-str $INC \
    --sigma-range 2 5 --no-excess-contours --out-dir $out

step p_vs_mass $PY $NB/p_vs_mass.py --fiducial wb1 --out $R/wb1/p_vs_mass
step contamination_profile $PY $NB/contamination_profile.py --summaries $PWD/$R/summaries_wb1_best_n5
step main_halo_profile $PY $NB/main_halo_profile.py --dx-kpc 0 --dy-kpc 0 --summaries $PWD/$R/summaries_wb1_best_n5 \
    --centre-json $R/centre_uncertainty/tweakedalign_k0p7_ctr_wb1_ba_best_${INC}.json
for w in rho noise-floor a-distance; do step paper_numbers_$w $PY $NB/paper_numbers.py $w; done

step paper_figures $PY $NB/paper_figures.py --mass-study $STUDY
grep -E "FAILED|written|\(" $LOG/paper_figures.txt
echo "== $(date '+%d %H:%M') done"
