#!/bin/bash
cd /Users/davidharvey/Work/ml_check
J=$PWD/notebooks_jauzac/plots; PY=$PWD/astro-env/bin/python; F=/Users/davidharvey/Papers/jwst_a2744/aa_draft/Figures
while pgrep -f ratio_scan_run.py >/dev/null; do sleep 15; done
ls $J/ratio_scan_50-100/ratio_scan_scatter.png && echo "wrote scatter"
cp $J/ratio_scan_50-100/ratio_scan_scatter.pdf $F/shear_ratio_scatter_50-100.pdf
(cd $F/.. && git add Figures && git commit -q -m "Shear-ratio scatter: 200 kpc outer annuli, +-0.25 dex mass window, paper style

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"; for i in $(seq 1 10); do git pull -q --rebase --autostash && git push -q origin main 2>/dev/null && { echo pushed; break; }; sleep 5; done)
echo "== $(date +%H:%M) figure 2, mass width x2"
cd notebooks_jauzac
for args in "ba_final_2sig_with_a2744 True" "ba_final_2sig False"; do set -- $args
  $PY paper_fig.py figure2_summaries 'p.paper_paths("final")["summaries"]' '{d: p.paper_paths("final")["flamingo"] for d in ("flamingo", "tng")}' 2 "\"$1\"" $2 True '"final"' 2.0 2>&1 | grep -i "saved\|error\|pooled\|scatter"
done
cd ..
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
