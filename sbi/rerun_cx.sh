#!/bin/bash
# Rerun the SBI, the systematics (now with the strong-lensing cut 1.0 -> 0.5 and the A2744 halo
# centring) and the sensitivity maps over all mass-matched BAHAMAS clusters for the spin2 models
# re-fine-tuned with the corrected shape noise (*spin2_cx_ft_*_ba_best, 30 seeds), on the current
# maps (each data set cut on its own convergence, kappa >= 0.9). One step at a time:
# get_obs_data rewrites the A2744 filtered fits. Logs to sbi/logs_rerun_cx/<step>.txt.
#
#   bash sbi/rerun_cx.sh
cd "$(dirname "$0")/.."
PY=astro-env/bin/python
SBI_PY=../Dark_Matter_DA/sbi/bin/python
LOG=sbi/logs_rerun_cx
INC=spin2_cx_ft_
MODEL=tweakedalign_k0p9_ctr_ba_best
S=sbi/summaries_spin2cx_best_n5
R=sbi/results
mkdir -p $LOG
step() { name=$1; shift; echo "== $(date +%H:%M) $name"; "$@" > $LOG/$name.txt 2>&1 || echo "   FAILED (see $LOG/$name.txt)"; }

# SBI: estimators trained once (n_comp 2+), then A2744 conditioned on each merger class
B=$R/summaries_spin2cx_best_n5_alldraws
step sbi_nc2plus $SBI_PY -u sbi/run_sbi.py --summaries $S --noise all --a2744-ncomp 2+ --retrain --out-dir ${B}_nc2plus
for nc in 3 3+ 4+; do
    d=${B}_nc${nc/+/plus}; mkdir -p "$d"; cp ${B}_nc2plus/nle_*.pt ${B}_nc2plus/npe_conditional_*.pt "$d"/
    step sbi_nc${nc/+/plus} $SBI_PY -u sbi/run_sbi.py --summaries $S --noise all --a2744-ncomp "$nc" --out-dir "$d"
done
grep -h -E "NLE:|conditional|A2744" $LOG/sbi_*.txt | grep -v -i warn

step systematics $PY -u sbi/systematics.py --model $MODEL --include-str $INC --min-ncomp 2 --summaries $S \
    --out-dir $R/systematics_spin2cx_best --redo
grep -E "^  (fiducial|kappa_cut|centring)" $LOG/systematics.txt

out=$R/sensitivity_spin2cx_best_all_bahamas
step sensitivity_all_bahamas $PY sbi/sensitivity.py --model $MODEL --include-str $INC --nmodels 5 --min-ncomp 2 \
    --domains bahamas --all-clusters --summaries $PWD/$S --out-dir $PWD/$out
grep -E "P\(SIDM\)|pushes|clusters" $LOG/sensitivity_all_bahamas.txt | tail -12
echo "== $(date +%H:%M) done"
