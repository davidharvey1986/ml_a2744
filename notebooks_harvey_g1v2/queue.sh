#!/bin/bash
# one catalogue user at a time: figure 1 -> ratio scatter (no titles) -> Jauzac systematics replot -> cut-systematics rerun
cd /Users/davidharvey/Work/ml_check
while kill -0  2>/dev/null; do sleep 20; done
cp notebooks/plots/intrinsic_ell.{png,pdf} notebooks_harvey_g1v2/plots/; cp notebooks/plots/intrinsic_ell.{png,pdf} notebooks_jauzac/plots/
echo "== $(date +%H:%M) fig1 done"
for hi in 100 200; do astro-env/bin/python notebooks_harvey_g1v2/ratio_scan_run.py --summaries $PWD/sbi/summaries_wb1_final_n5 --num-kpc 50 $hi --out-dir $PWD/notebooks_harvey_g1v2/plots/ratio_scan_50-$hi 2>&1 | grep wrote; done
echo "== $(date +%H:%M) ratio done"
astro-env/bin/python - <<PYEOF 2>&1 | grep -v -i "warn\|radesys"
import json, sys
sys.path.insert(0, 'sbi')
from pathlib import Path
import systematics as sy
out = json.load(open('sbi/results/systematics_wb1_final_jauzac/systematics.json'))
P = Path('notebooks_jauzac/plots')
sy.plot_ranges({**out, "contamination": [r for r in out.get("contamination", []) if abs(r["value"]) <= sy.CONTAMINATION_MAX + 1e-9]},
               P / 'systematics_model_paper_ba_final', list(sy.MODEL_GROUPS), None, show_total=False, show_shifts=False)
sy.plot_data(out, P / 'systematics_data_paper_ba_final', paper=True, skip=("galaxy_low_mag", "galaxy_high_mag", "galaxy_size"))
print('jauzac systematics plots done')
PYEOF
echo "== $(date +%H:%M) cut systematics start"
S=sbi/summaries_wb1_final_n5; O=sbi/results/systematics_wb1_final_harvey_g1v2
astro-env/bin/python -u sbi/systematics.py --model tweakedalign_k0p7_ctr_wb1_ba_final --include-str spin2_cx_pgc_ft_ --min-ncomp 2     --summaries $S --out-dir $O --dz-summaries ${S}_dz --mass-study harvey_g1v2 > notebooks_harvey_g1v2/systematics_cuts.log 2>&1
echo "== $(date +%H:%M) ALL_DONE"
