#!/bin/bash
# After k0p7_extra.sh: ba_final summaries with latents (one noise draw) for Figure A3, then every paper
# figure from the current pipeline (notebooks/plots_for_paper.py, all_figures) into notebooks/plots/.
# Logs to sbi/logs_k0p7/.
cd "$(dirname "$0")/.."
caffeinate -i -w $$ &
PY=$PWD/astro-env/bin/python; LOG=$PWD/sbi/logs_k0p7; INC=spin2_cx_ft_
step() { name=$1; shift; echo "== $(date +%H:%M) $name"; "$@" > $LOG/$name.txt 2>&1 || echo "   FAILED (see $LOG/$name.txt)"; }
while pgrep -f "sbi/k0p7_cx.sh|sbi/k0p7_extra.sh" > /dev/null; do sleep 120; done
step summaries_latents_final $PY -u sbi/make_summaries.py --model tweakedalign_k0p9_ctr_ba_final --include-str $INC \
    --n-noise 1 --domains bahamas darkskies --out-dir $PWD/sbi/summaries_spin2cx_final_n5_latents
cd notebooks
step paper_figures $PY plots_for_paper.py
grep -E "Figure|Saved|sigma|rho|Discrepancy" $LOG/paper_figures.txt
echo "== $(date +%H:%M) done"
