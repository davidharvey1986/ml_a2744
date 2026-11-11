#!/bin/bash
# after the ratio scan and figures 2/7: sensitivity panel, then the cut-systematics rerun (Jauzac) and its plots
cd /Users/davidharvey/Work/ml_check
J=$PWD/notebooks_jauzac/plots; PY=astro-env/bin/python
while pgrep -f "ratio_scan_run.py|fig2_obs.sh" >/dev/null; do sleep 20; done
echo "== $(date +%H:%M) ratio + figure 2 done"
echo "== $(date +%H:%M) sensitivity panel"
$PY sbi/pairs_obs_panel.py --pairs 7 10 12 15 16 --out $J/pairs_obs_sensitivity 2>&1 | grep -i "wrote\|error"
echo "== $(date +%H:%M) cut systematics (jauzac)"
S=sbi/summaries_wb1_final_n5; O=sbi/results/systematics_wb1_final_jauzac
$PY -u sbi/systematics.py --model tweakedalign_k0p7_ctr_wb1_ba_final --include-str spin2_cx_pgc_ft_ --min-ncomp 2 \
    --summaries $S --out-dir $O --dz-summaries ${S}_dz --mass-study jauzac > notebooks_jauzac/systematics_cuts.log 2>&1
tail -2 notebooks_jauzac/systematics_cuts.log
$PY - <<PYEOF 2>&1 | grep -v -i "warn\|radesys"
import json, sys
sys.path.insert(0, 'sbi')
from pathlib import Path
import systematics as sy
out = json.load(open('$O/systematics.json'))
P = Path('$J')
sy.plot_ranges({**out, "contamination": [r for r in out.get("contamination", []) if abs(r["value"]) <= sy.CONTAMINATION_MAX + 1e-9]},
               P / 'systematics_model_paper_ba_final', list(sy.MODEL_GROUPS), None, show_total=False, show_shifts=False)
sy.plot_data(out, P / 'systematics_data_paper_ba_final', paper=True, skip=("galaxy_low_mag", "galaxy_high_mag", "galaxy_size"))
sy.plot_data(out, P / 'systematics_data_all_cuts_ba_final', paper=True)
print('wrote jauzac systematics plots')
PYEOF
echo "== $(date +%H:%M) ALL_DONE"
