#!/bin/bash
# The user replaced data/100/convergence/flamingo{,_lo,_hi}.pkl at 14:40 on 5 Nov. After ba40.sh (it rewrites the
# A2744 catalogue, as this does), resample them on the A2744 galaxies into data/100/obs/concat, then redo the wb1
# _flamingo summaries, the CDM test, p_vs_mass and the paper figures. The old obs maps and summaries are moved to
# archive_flamingo_pre_20261105 dirs.
#
#   nohup bash sbi/pipelines_20261104/flamingo_redo.sh > sbi/pipelines_20261104/flamingo_redo.log 2>&1 &
source "$(dirname "$0")/common.sh"
caffeinate -is -w $$ &
LOG=$PWD/sbi/logs_flamingo_redo; mkdir -p $LOG
INC=spin2_cx_pgc_ft_

echo "== $(date '+%d %H:%M') waiting for ba40.sh"
until grep -q "== .* all done" sbi/pipelines_20261104/ba40.log && ! pgrep -f ba40.sh > /dev/null; do sleep 120; done

O=data/100/obs/concat; A=data/100/obs/archive_flamingo_pre_20261105; mkdir -p $A
for f in flamingo flamingo_lo flamingo_hi; do [ -e $A/$f.pkl ] || mv $O/$f.pkl $A/; done
for f in flamingo flamingo_lo flamingo_hi; do
    step obs_maps_$f $PY -c "from add_shear_to_data import main; main(search_path='data/100/convergence/$f.pkl', sample_data=True, data_dir='data/100/a2744')"
done
ls -l $O/flamingo.pkl $O/flamingo_lo.pkl $O/flamingo_hi.pkl || { echo "obs maps missing; stopping"; exit 1; }

SA=sbi/archive_flamingo_pre_20261105; mkdir -p $SA
for t in final best; do
    S=$PWD/sbi/summaries_wb1_${t}_n5; F=${S}_flamingo
    [ -e $SA/$(basename $F) ] || mv $F $SA/
    summaries summaries_flamingo_$t tweakedalign_k0p7_ctr_wb1_ba_$t $INC $F flamingo flamingo_lo flamingo_hi tng vd_bahamas
    step cdm_test_$t $PY sbi/flamingo_test.py --summaries $S --flamingo $F --label "wb1 ba_$t" --min-ncomp 3 \
        --out $F/flamingo_test_nc3_kde
    grep "n_comp" $LOG/cdm_test_$t.txt
done
step p_vs_mass $PY sbi/p_vs_mass.py --fiducial wb1 --out $R/wb1/p_vs_mass
cd notebooks
step paper_figures $PY plots_for_paper.py
grep -E "Figure|Saved|sigma|rho|Discrepancy" $LOG/paper_figures.txt
echo "== $(date '+%d %H:%M') done"
