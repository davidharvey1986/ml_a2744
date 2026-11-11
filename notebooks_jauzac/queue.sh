#!/bin/bash
# Jauzac fiducial, one catalogue user at a time
cd /Users/davidharvey/Work/ml_check
J=$PWD/notebooks_jauzac/plots; PY=astro-env/bin/python
while pgrep -f ratio_scan_run.py >/dev/null; do sleep 20; done
echo "== $(date +%H:%M) profile"
$PY sbi/main_halo_lenstool.py --no-lenstool --sims --rmax 800 --nbins 26 --mass-study jauzac --out $PWD/sbi/results/main_halo_profile/main_halo_sims_jauzac 2>&1 | grep "clusters\|wrote"
cp sbi/results/main_halo_profile/main_halo_sims_jauzac.{png,pdf} $J/
echo "== $(date +%H:%M) figure1"
(cd notebooks && ../$PY -c "import plots_for_paper as p; p.figureA1()" 2>&1 | grep -i "error\|n(z)")
cp notebooks/plots/intrinsic_ell.{png,pdf} $J/; echo "wrote $J/intrinsic_ell.png"
echo "== $(date +%H:%M) ratio"
for hi in 100 200; do MASS_STUDY=jauzac $PY notebooks_harvey_g1v2/ratio_scan_run.py --summaries $PWD/sbi/summaries_wb1_final_n5 --num-kpc 50 $hi --out-dir $J/ratio_scan_50-$hi 2>&1 | grep "wrote\|Error"; done
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
