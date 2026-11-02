#!/bin/bash
# The kappa < 0.7 ensembles re-fine-tuned with the shape noise combined per galaxy (train_seed.sh PER_GALAXY_CHI=1,
# run name *_nob1_spin2_cx_pgc_ft_tweakedalign_k0p7_ctr; dataset.apply_intrinsic_ell per_galaxy_chi, now the
# default for evaluation too). Waits for the 30 bonsai jobs, copies the ba_best/ba_final checkpoints, then reruns
# everything after the data steps of k0p7_models.sh (the A2744 map with the 2-D G1, the noise model and the
# simulated maps it built are unchanged by the noise fix). ba_final first. Logs to sbi/logs_k0p7_pgc/.
#
#   bash sbi/k0p7_pgc.sh
cd "$(dirname "$0")/.."
caffeinate -i -w $$ &
PY=$PWD/astro-env/bin/python
SBI_PY=../Dark_Matter_DA/sbi/bin/python
LOG=$PWD/sbi/logs_k0p7_pgc
INC=spin2_cx_pgc_ft_
R=sbi/results
REMOTE=bonsai:/srv/scratch/dharvey/ml_nature/models/concat/
mkdir -p $LOG
step() { name=$1; shift; echo "== $(date +%H:%M) $name"; "$@" > $LOG/$name.txt 2>&1 || echo "   FAILED (see $LOG/$name.txt)"; }

# wait for the fine-tuning: 30 ba_final checkpoints and no incr_learn jobs left
while true; do
    n=$(ssh -o BatchMode=yes bonsai "ls /srv/scratch/dharvey/ml_nature/models/concat/*${INC}tweakedalign_k0p7_ctr_ba_final.pth 2>/dev/null | wc -l" 2>/dev/null)
    q=$(ssh -o BatchMode=yes bonsai "squeue -u dharvey -h -n incr_learn | wc -l" 2>/dev/null)
    [ -n "$n" ] && [ -n "$q" ] || { sleep 600; continue; }
    [ "$q" -eq 0 ] && break
    sleep 600
done
echo "== $(date +%H:%M) bonsai: $n ba_final checkpoints, no incr_learn jobs left"
[ "$n" -eq 30 ] || { echo "   expected 30: check the bonsai logs; stopping"; exit 1; }
rsync -a --include="*${INC}tweakedalign_k0p7_ctr_ba_best.pth" --include="*${INC}tweakedalign_k0p7_ctr_ba_final.pth" \
    --exclude='*' $REMOTE models/concat/ 2>/dev/null
echo "   copied: $(ls models/concat/*${INC}tweakedalign_k0p7_ctr_ba_*.pth | wc -l) checkpoints"
[ "$(md5 -q data/100/a2744/ngal_concat.pkl)" = 397a78276be7777fdc573e71b6c219f5 ] \
    && echo "   ngal_concat.pkl identical to the bonsai copy" || echo "   ngal_concat.pkl DIFFERS from the bonsai copy"

for t in final best; do
    MODEL=tweakedalign_k0p7_ctr_ba_$t
    S=$PWD/sbi/summaries_spin2cx_${t}_n5
    F=${S}_flamingo
    D=${S}_dz
    step summaries_$t $PY -u sbi/make_summaries.py --model $MODEL --include-str $INC --n-noise 5 --no-latents \
        --domains bahamas darkskies --out-dir $S
    grep -h "A2744: ensemble" $LOG/summaries_$t.txt | tr '\r' '\n' | grep A2744
    step summaries_flamingo_$t $PY -u sbi/make_summaries.py --model $MODEL --include-str $INC --n-noise 5 --no-latents \
        --domains flamingo flamingo_lo flamingo_hi tng vd_bahamas --out-dir $F
    step summaries_dz_$t $PY -u sbi/make_summaries.py --model $MODEL --include-str $INC --n-noise 5 --no-latents \
        --domains flamingo_dz25 flamingo_dz50 flamingo_dz100 flamingo_dz200 flamingo_dz250 flamingo_dz500 --out-dir $D
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
    grep -h -E "NLE:|conditional|empirical" $LOG/sbi_${t}_*.txt | grep -v -i warn

    C=$R/centre_uncertainty/${MODEL}_${INC}.json
    step response_$t $PY sbi/response_plot.py --summaries $S --label "spin2 κ<0.7 per-galaxy χ, ba_$t" --centre-json $C
    out=$R/chifix/response_combined; [ $t = final ] && out=${out}_final
    for nc in 2 3; do
        step response_combined_${t}_nc$nc $PY sbi/response_combined.py --summaries $S --centre-json $C --min-ncomp $nc \
            --label "spin2 κ<0.7 per-galaxy χ, ba_$t (30 models)" --out $out
    done
    step sensitivity_w4s1_$t $PY sbi/sensitivity.py --model $MODEL --include-str $INC --nmodels 5 \
        --window 4 --stride 1 --min-ncomp 2 --domains bahamas darkskies --summaries $S \
        --out-dir $PWD/$R/sensitivity_spin2cx_${t}_w4s1
    grep -E "P\(SIDM\)|pushes" $LOG/sensitivity_w4s1_$t.txt | tail -4
    step sensitivity_all_bahamas_$t $PY sbi/sensitivity.py --model $MODEL --include-str $INC --nmodels 5 --min-ncomp 2 \
        --domains bahamas --all-clusters --summaries $S --out-dir $PWD/$R/sensitivity_spin2cx_${t}_all_bahamas

    step systematics_$t $PY -u sbi/systematics.py --model $MODEL --include-str $INC --min-ncomp 2 --summaries $S \
        --out-dir $R/systematics_spin2cx_$t --dz-summaries $D
    grep -E "^  (fiducial|kappa_cut|shape|merging)" $LOG/systematics_$t.txt
done

step summaries_latents_final $PY -u sbi/make_summaries.py --model tweakedalign_k0p7_ctr_ba_final --include-str $INC \
    --n-noise 1 --domains bahamas darkskies --out-dir $PWD/sbi/summaries_spin2cx_final_n5_latents
step response_compare $PY sbi/jitter_response.py --title "spin2 κ<0.7 per-galaxy χ: best vs final checkpoints" \
    --summaries sbi/summaries_spin2cx_best_n5 sbi/summaries_spin2cx_final_n5 --labels "ba_best" "ba_final" \
    --out $R/chifix/response_best_vs_final.png
step eb_swap $PY sbi/eb_swap_test.py --model tweakedalign_k0p7_ctr_ba_best --include-str $INC \
    --summaries $PWD/sbi/summaries_spin2cx_best_n5 --out-dir $PWD/$R/eb_swap_test_spin2cx
MODEL=tweakedalign_k0p7_ctr_ba_best; out=$R/sensitivity_spin2cx_best_stride1
step sensitivity_best_stride1 $PY sbi/sensitivity.py --model $MODEL --include-str $INC --nmodels 5 --stride 1 \
    --domains --summaries $PWD/sbi/summaries_spin2cx_best_n5 --out-dir $out
step sensitivity_best_stride1_plot $PY sbi/sensitivity.py --a2744-only --model $MODEL --include-str $INC \
    --sigma-range 2 5 --no-excess-contours --out-dir $out

step p_vs_mass $PY sbi/p_vs_mass.py
step contamination_profile $PY sbi/contamination_profile.py
step main_halo_profile $PY sbi/main_halo_profile.py --dx-kpc 0 --dy-kpc 0 --summaries $PWD/sbi/summaries_spin2cx_best_n5
cd notebooks
step paper_figures $PY plots_for_paper.py
grep -E "Figure|Saved|sigma|rho|Discrepancy" $LOG/paper_figures.txt
echo "== $(date +%H:%M) done"
