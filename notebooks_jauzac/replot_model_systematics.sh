#!/bin/bash
# After the figures pipeline and the Jauzac figures (both rewrite the catalogue on import): redraw the model-systematics
# plots with the "× leak" contamination labels and copy them to the paper plots.
cd /Users/davidharvey/Work/ml_check
until grep -q JAUZAC_DONE notebooks_jauzac/run_jauzac.log 2>/dev/null && ! pgrep -f "figures_both.sh|run_jauzac.py" > /dev/null; do sleep 30; done
for t in final best; do
    astro-env/bin/python -c "
import json, sys; sys.path.insert(0, 'sbi')
from pathlib import Path
import systematics as sy
d = Path('$PWD/sbi/results/systematics_wb1_$t'); sy.plot_model(json.load(open(d / 'systematics.json')), d / 'systematics_model')" 2>&1 | grep -v -i "warn\|radesys\|radecsys"
    for e in pdf png; do cp sbi/results/systematics_wb1_$t/systematics_model.$e notebooks/plots/summarised_systematics_ba_$t.$e; done
done
echo REPLOT_DONE
