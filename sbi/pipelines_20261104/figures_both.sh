#!/bin/bash
# Rest of flamingo_redo.sh (killed in summaries_flamingo_best), then the paper figures for ba_final and ba_best:
# best FLAMINGO summaries + CDM test, systematics data/model plots redrawn from systematics.json (grey bands),
# p_vs_mass, all figures for final, and for best every figure except A3 (no best latents).
#
#   nohup bash sbi/pipelines_20261104/figures_both.sh > sbi/pipelines_20261104/figures_both.log 2>&1 &
source "$(dirname "$0")/common.sh"
caffeinate -is -w $$ &
LOG=$PWD/sbi/logs_figures_both; mkdir -p $LOG
INC=spin2_cx_pgc_ft_

S=$PWD/sbi/summaries_wb1_best_n5; F=${S}_flamingo
summaries summaries_flamingo_best tweakedalign_k0p7_ctr_wb1_ba_best $INC $F flamingo flamingo_lo flamingo_hi tng vd_bahamas
step cdm_test_best $PY sbi/flamingo_test.py --summaries $S --flamingo $F --label "wb1 ba_best" --min-ncomp 3 \
    --out $F/flamingo_test_nc3_kde
grep "n_comp" $LOG/cdm_test_best.txt

for t in final best; do
    D=$PWD/$R/systematics_wb1_$t
    step replot_systematics_$t $PY -c "
import json, sys; sys.path.insert(0, 'sbi')
from pathlib import Path
import systematics as sy
d = Path('$D'); out = json.load(open(d / 'systematics.json'))
sy.plot_data(out, d / 'systematics_data'); sy.plot_model(out, d / 'systematics_model')"
done

step p_vs_mass $PY sbi/p_vs_mass.py --fiducial wb1 --out $R/wb1/p_vs_mass
cd notebooks
step paper_figures_final $PY plots_for_paper.py
step paper_figures_best $PY -c "
import plots_for_paper as p
t = 'best'
P = p.paper_paths(t)
p.figure2_summaries(P['summaries'], {d: P['flamingo'] for d in ('flamingo', 'tng')}, tag=f'ba_{t}')
p.figure3_sensitivity(t); p.figure4_sensitivity(t)
ls, le = p.figure5_summaries(t)
p.figureA2_summaries(t); p.figureA4A5_systematics(t)"
grep -E "Figure|Saved|sigma|rho|Discrepancy" $LOG/paper_figures_*.txt
echo "== $(date '+%d %H:%M') all done"
