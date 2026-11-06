#!/bin/bash
# After flamingo_redo.sh (all of these rewrite the A2744 catalogue on import, so one at a time): redraw the data
# systematics plots with the galaxy-number grey bands from systematics.json (the --from-cache replot fails in
# merging()), copy wb1 final's into notebooks/plots for Figure A4, then boost the A2744 data with B(r) for the wb1
# ba_final ensemble (sbi/boost_data.py: profile + P(SIDM) plot).
#
#   nohup bash sbi/pipelines_20261104/boost_after.sh > sbi/pipelines_20261104/boost_after.log 2>&1 &
source "$(dirname "$0")/common.sh"
caffeinate -is -w $$ &
LOG=$PWD/sbi/logs_boost_after; mkdir -p $LOG

echo "== $(date '+%d %H:%M') waiting for flamingo_redo.sh"
until grep -q "== .* done\|stopping" sbi/pipelines_20261104/flamingo_redo.log && ! pgrep -f flamingo_redo.sh > /dev/null; do
    sleep 120
done

cd sbi
step replot_systematics $PY -c "
import json, pathlib, sys; sys.argv = ['x']
import systematics as S
for t in ['wb1_final', 'wb1_best', 'spin2cx_final', 'spin2cx_best']:
    d = pathlib.Path('/Users/davidharvey/Work/ml_check/sbi/results/systematics_' + t)
    S.plot_data(json.load(open(d / 'systematics.json')), d / 'systematics_data')
    print('redrew', d)"
cd ..
for ext in pdf png; do cp sbi/results/systematics_wb1_final/systematics_data.$ext notebooks/plots/data_systematics_ba_final.$ext; done

step boost_data_wb1_final $PY -u sbi/boost_data.py --model tweakedalign_k0p7_ctr_wb1_ba_final --include-str spin2_cx_pgc_ft_ \
    --summaries sbi/summaries_wb1_final_n5 --out-dir sbi/results/boost_data_wb1_final \
    --fiducial-from sbi/results/systematics_wb1_final
grep -E "fiducial|A2744 P\(SIDM\)|A = " $LOG/boost_data_wb1_final.txt | cut -c1-220
echo "== $(date '+%d %H:%M') done"
