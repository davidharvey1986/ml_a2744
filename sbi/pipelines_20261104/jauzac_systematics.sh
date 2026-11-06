#!/bin/bash
# Systematics with the Jauzac A2744 mass (14.735 +- 0.116), clusters selected within SUBSET_DEX of it, for ba_final
# then ba_best. Replaces the re-weighted Harvey caches of notebooks_jauzac/systematics_jauzac.py (archived first).
#
#   nohup bash sbi/pipelines_20261104/jauzac_systematics.sh > sbi/pipelines_20261104/jauzac_systematics.log 2>&1 &
source "$(dirname "$0")/common.sh"
caffeinate -is -w $$ &
LOG=$PWD/sbi/logs_jauzac_systematics; mkdir -p $LOG
INC=spin2_cx_pgc_ft_
A=$R/archive_systematics_jauzac_reweighted_20261106; mkdir -p $A

for t in final best; do
    O=$R/systematics_wb1_${t}_jauzac
    [ -d $O ] && [ ! -e $O/fiducial.npz ] && mv $O $A/
    S=$PWD/sbi/summaries_wb1_${t}_n5
    step systematics_jauzac_$t $PY -u sbi/systematics.py --model tweakedalign_k0p7_ctr_wb1_ba_$t --include-str $INC \
        --min-ncomp 2 --summaries $S --out-dir $O --dz-summaries ${S}_dz --mass-study jauzac
    grep -E "clusters simulated|^  (fiducial|merging|contamination 0.2|contamination -0.2)" $LOG/systematics_jauzac_$t.txt
    for f in data model; do for e in png pdf; do
        [ -e $O/systematics_$f.$e ] && cp $O/systematics_$f.$e notebooks_jauzac/plots/systematics_${f}_ba_$t.$e; done; done
done
echo "== $(date '+%d %H:%M') all done"
