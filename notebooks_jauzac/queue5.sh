#!/bin/bash
cd /Users/davidharvey/Work/ml_check
J=$PWD/notebooks_jauzac/plots; PY=$PWD/astro-env/bin/python; F=/Users/davidharvey/Papers/jwst_a2744/aa_draft/Figures
while pgrep -f paper_fig.py >/dev/null; do sleep 15; done
echo "== $(date +%H:%M) figure 2"
cd notebooks_jauzac
for args in "ba_final_with_a2744 True"; do set -- $args
  $PY paper_fig.py figure2_summaries 'p.paper_paths("final")["summaries"]' '{d: p.paper_paths("final")["flamingo"] for d in ("flamingo", "tng")}' 2 "\"$1\"" $2 2>&1 | grep -i "saved\|error"
done
cd ..
cd /Users/davidharvey/Work/ml_check
echo "== $(date +%H:%M) sensitivity panel"
$PY sbi/pairs_obs_panel.py --pairs 7 10 12 15 16 --out $J/pairs_obs_sensitivity 2>&1 | grep -i "wrote\|error"
echo "== $(date +%H:%M) push"
cp $J/final_model_weighting_with_data_ba_final.pdf $F/final_model_weighting_with_data.pdf
cp $J/pairs_obs_sensitivity.pdf $F/sensitivity_map.pdf
(cd $F/.. && git add Figures && git commit -q -m "Figure 2 right panel: per-cluster CDM discrepancy; sensitivity map with kappa S/N 2,4,8,12

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"; for i in $(seq 1 10); do git pull -q --rebase --autostash && git push -q origin main 2>/dev/null && { echo pushed; break; }; sleep 5; done)
echo "== $(date +%H:%M) ratio scatter"
MASS_STUDY=jauzac $PY notebooks_harvey_g1v2/ratio_scan_run.py --summaries $PWD/sbi/summaries_wb1_final_n5 --num-kpc 50 100 --den-start 100 --den-width 200 --mass-window 0.25 --out-dir $J/ratio_scan_50-100 2>&1 | grep "wrote\|Error"
cp $J/ratio_scan_50-100/ratio_scan_scatter.pdf $F/shear_ratio_scatter_50-100.pdf
(cd $F/.. && git add Figures && git commit -q -m "Shear-ratio scatter: 200 kpc outer annuli, +-0.25 dex mass window, paper style

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"; for i in $(seq 1 10); do git pull -q --rebase --autostash && git push -q origin main 2>/dev/null && { echo pushed; break; }; sleep 5; done)
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
