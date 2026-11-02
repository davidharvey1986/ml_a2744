#!/bin/bash
# After k0p7_cx.sh, k0p7_extra.sh and k0p7_figures.sh: switch the A2744 catalogue to the 2-D shear
# susceptibility G1(size, S/N) (shear_susceptibility.py, applied in add_shear_to_data.get_obs_data),
# rebuild the A2744 map and the shape-noise model (ngal_concat.pkl), and rerun what depends on the
# A2744 map: its P(SIDM) for ba_best and ba_final (obs.npz in every summaries directory), centre
# uncertainty, CDM test, SBI, response curves, E/B swap, A2744 sensitivity maps, the profile and mass
# plots and the paper figures. The simulation summaries are not redone (their shape noise stays at the
# pre-fix level, 0.9% higher); nor are the systematics sweeps. Everything overwritten is first copied to
# archive_k0p7_pyrrg/. One catalogue reader at a time. Logs to sbi/logs_g1fix/.
#
#   bash sbi/g1fix_after.sh
cd "$(dirname "$0")/.."
caffeinate -i -w $$ &
PY=$PWD/astro-env/bin/python
SBI_PY=../Dark_Matter_DA/sbi/bin/python
LOG=$PWD/sbi/logs_g1fix
INC=spin2_cx_ft_
R=sbi/results
A=archive_k0p7_pyrrg
mkdir -p $LOG
step() { name=$1; shift; echo "== $(date +%H:%M) $name"; "$@" > $LOG/$name.txt 2>&1 || echo "   FAILED (see $LOG/$name.txt)"; }
while pgrep -f "sbi/k0p7_cx.sh|sbi/k0p7_extra.sh|sbi/k0p7_figures.sh" > /dev/null; do sleep 120; done

echo "== $(date +%H:%M) archive to $A"
SUMS=$(ls -d sbi/summaries_spin2cx_{best,final}_n5{,_flamingo,_dz,_latents} 2>/dev/null)
rsync -aR data/100/a2744/obs_data_concat.pkl data/100/a2744/ngal_concat.pkl data/100/a2744/a2744_*_filtered.fits \
    $(for d in $SUMS; do echo $d/obs.npz; ls $d/flamingo_test* 2>/dev/null; done) \
    $R/summaries_spin2cx_{best,final}_n5_alldraws_* $R/centre_uncertainty $R/chifix $R/eb_swap_test_spin2cx \
    $R/sensitivity_spin2cx_best_stride1 $R/sensitivity_spin2cx_final_w4s1 $R/main_halo_profile $R/contamination_profile \
    $R/p_vs_mass.* notebooks/plots add_shear_to_data.py notebooks/plots_for_paper.py $A/ 2>&1 | grep -v "No such file" | head

step apply_patch $PY sbi/g1fix/apply_patch.py
step obs_map $PY -c "from add_shear_to_data import make_obs_maps; make_obs_maps()"
step noise_model $PY -c "from add_shear_to_data import make_noise_model; make_noise_model()"
cmp -s data/100/a2744/ngal_concat.pkl /private/tmp/claude-502/-Users-davidharvey-Applications-Imperial-Job/2fd953fc-3740-40a9-b0d1-91ed4c06ee8a/scratchpad/g1fix/a2744/ngal_concat.pkl \
    && echo "   ngal_concat.pkl identical to the copy uploaded to bonsai" || echo "   ngal_concat.pkl DIFFERS from the bonsai copy"

for t in best final; do
    MODEL=tweakedalign_k0p9_ctr_ba_$t
    S=$PWD/sbi/summaries_spin2cx_${t}_n5
    F=${S}_flamingo
    step rescore_$t $PY sbi/g1fix/rescore_obs.py --model $MODEL --include-str $INC \
        --dirs $(ls -d sbi/summaries_spin2cx_${t}_n5{,_flamingo,_dz,_latents} 2>/dev/null)
    grep "P(SIDM)" $LOG/rescore_$t.txt
    step centre_uncertainty_$t $PY sbi/centre_uncertainty.py --model $MODEL --include-str $INC
    step cdm_test_$t $PY sbi/flamingo_test.py --summaries $S --flamingo $F --label ba_$t --min-ncomp 3 \
        --out $F/flamingo_test_nc3_kde
    grep "n_comp" $LOG/cdm_test_$t.txt
    B=$R/summaries_spin2cx_${t}_n5_alldraws
    for nc in 2+ 3 3+ 4+; do      # the trained NLE/NPE in each directory are reused
        step sbi_${t}_nc${nc/+/plus} $SBI_PY -u sbi/run_sbi.py --summaries $S --noise all --a2744-ncomp "$nc" \
            --out-dir ${B}_nc${nc/+/plus}
    done
    grep -h -E "NLE:|conditional|empirical" $LOG/sbi_${t}_*.txt | grep -v -i warn
done

# response curves and E/B swap (as k0p7_extra.sh)
for t in best final; do
    S=$PWD/sbi/summaries_spin2cx_${t}_n5
    C=$R/centre_uncertainty/tweakedalign_k0p9_ctr_ba_${t}_${INC}.json
    step response_$t $PY sbi/response_plot.py --summaries $S --label "spin2 re-fine-tuned, ba_$t, κ < 0.7" --centre-json $C
    out=$R/chifix/response_combined; [ $t = final ] && out=${out}_final
    for nc in 2 3; do
        step response_combined_${t}_nc$nc $PY sbi/response_combined.py --summaries $S --centre-json $C --min-ncomp $nc \
            --label "spin2 re-fine-tuned, ba_$t (30 models), κ < 0.7" --out $out
    done
done
step response_compare $PY sbi/jitter_response.py --title "spin2 re-fine-tuned: best vs final checkpoints, κ < 0.7" \
    --summaries sbi/summaries_spin2cx_best_n5 sbi/summaries_spin2cx_final_n5 --labels "ba_best" "ba_final" \
    --out $R/chifix/response_best_vs_final.png
step eb_swap $PY sbi/eb_swap_test.py --model tweakedalign_k0p9_ctr_ba_best --include-str $INC \
    --summaries $PWD/sbi/summaries_spin2cx_best_n5 --out-dir $PWD/$R/eb_swap_test_spin2cx

# A2744 sensitivity maps: ba_best stride 1 (A2744 only) and the ba_final w4s1 map used in the paper
MODEL=tweakedalign_k0p9_ctr_ba_best; out=$R/sensitivity_spin2cx_best_stride1
step sensitivity_best_stride1 $PY sbi/sensitivity.py --model $MODEL --include-str $INC --nmodels 5 --stride 1 \
    --domains --summaries $PWD/sbi/summaries_spin2cx_best_n5 --out-dir $out
step sensitivity_best_stride1_plot $PY sbi/sensitivity.py --a2744-only --model $MODEL --include-str $INC \
    --sigma-range 2 5 --no-excess-contours --out-dir $out
step sensitivity_w4s1_final $PY sbi/sensitivity.py --model tweakedalign_k0p9_ctr_ba_final --include-str $INC --nmodels 5 \
    --window 4 --stride 1 --min-ncomp 2 --domains bahamas darkskies --summaries $PWD/sbi/summaries_spin2cx_final_n5 \
    --out-dir $PWD/$R/sensitivity_spin2cx_final_w4s1
grep -E "P\(SIDM\)|pushes" $LOG/sensitivity_w4s1_final.txt | tail -4

step p_vs_mass $PY sbi/p_vs_mass.py
step contamination_profile $PY sbi/contamination_profile.py
step main_halo_profile $PY sbi/main_halo_profile.py --dx-kpc 0 --dy-kpc 0 --summaries $PWD/sbi/summaries_spin2cx_best_n5
cd notebooks
step paper_figures $PY plots_for_paper.py
grep -E "Figure|Saved|sigma|rho|Discrepancy" $LOG/paper_figures.txt
echo "== $(date +%H:%M) done"
