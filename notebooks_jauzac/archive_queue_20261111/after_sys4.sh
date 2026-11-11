#!/bin/bash
# after the post-run steps: rerun the additive shear bias test (c only on occupied pixels), redraw the model plot, push
cd /Users/davidharvey/Work/ml_check; R=$PWD
sleep 90
while pgrep -f "notebooks_jauzac/after_sys.sh|notebooks_jauzac/after_sys2.sh|notebooks_jauzac/after_sys3.sh|notebooks_jauzac/queue9.sh" >/dev/null; do sleep 60; done
O=sbi/results/systematics_wb1_final_jauzac; mkdir -p $O/archive_shape_c_allpixels && mv $O/shape_c_*.npz $O/archive_shape_c_allpixels/
echo "== $(date +%H:%M) additive bias rerun"
S=sbi/summaries_wb1_final_n5
astro-env/bin/python -u sbi/systematics.py --model tweakedalign_k0p7_ctr_wb1_ba_final --include-str spin2_cx_pgc_ft_ --min-ncomp 2 \
    --summaries $S --out-dir $O --dz-summaries ${S}_dz --mass-study jauzac > notebooks_jauzac/systematics_shape_c.log 2>&1
tr '\r' '\n' < notebooks_jauzac/systematics_shape_c.log | grep "^  shape_c" | cut -c1-120
astro-env/bin/python - <<PYEOF 2>&1 | grep -v -i "warn\|radesys\|Source redshift"
import json, sys
sys.path[:0] = ['$R/sbi', '$R/notebooks', '$R']
from pathlib import Path
import systematics as sy
gp = json.load(open('$R/notebooks_jauzac/plots/gp_posterior_ba_final.json'))
out = json.load(open('$R/$O/systematics.json'))
sy.plot_ranges({**out, "contamination": [r for r in out.get("contamination", []) if abs(r["value"]) <= sy.CONTAMINATION_MAX + 1e-9]},
               Path('$R/notebooks_jauzac/plots/systematics_model_paper_ba_final'), list(sy.MODEL_GROUPS), None, show_total=False, show_shifts=False,
               stat=(gp["lo"], gp["hi"]), centre=gp["ml"])
sy.plot_data(out, Path('$R/notebooks_jauzac/plots/systematics_data_paper_ba_final'), paper=True, skip=("galaxy_low_mag", "galaxy_high_mag", "galaxy_size"))
print("plots done")
PYEOF
cp notebooks_jauzac/plots/systematics_model_paper_ba_final.pdf /Users/davidharvey/Papers/jwst_a2744/aa_draft/Figures/systematics_model.pdf
cp notebooks_jauzac/plots/systematics_data_paper_ba_final.pdf /Users/davidharvey/Papers/jwst_a2744/aa_draft/Figures/systematics_data.pdf
cd /Users/davidharvey/Papers/jwst_a2744/aa_draft && git add Figures && git commit -q -m "Systematics: additive shear bias only on occupied pixels" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in $(seq 1 10); do git pull -q --rebase --autostash && git push -q origin main 2>/dev/null && { echo pushed; break; }; sleep 5; done
echo "== $(date +%H:%M) done"
