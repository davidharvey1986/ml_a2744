#!/bin/bash
# projection-depth test back in (absolute --dz-summaries), redraw both systematics plots, push
cd /Users/davidharvey/Work/ml_check; R=$PWD; O=sbi/results/systematics_wb1_final_jauzac; S=sbi/summaries_wb1_final_n5
sleep 180
while pgrep -f "notebooks_jauzac/after_sys5.sh|notebooks_jauzac/after_sys6.sh|plots_for_paper|systematics.py|pairs_obs_panel" >/dev/null; do sleep 30; done
echo "== $(date +%H:%M) projection rerun"
astro-env/bin/python -u sbi/systematics.py --model tweakedalign_k0p7_ctr_wb1_ba_final --include-str spin2_cx_pgc_ft_ --min-ncomp 2 \
    --summaries $S --out-dir $O --dz-summaries $R/${S}_dz --mass-study jauzac > notebooks_jauzac/systematics_projection.log 2>&1
tr '\r' '\n' < notebooks_jauzac/systematics_projection.log | grep -i "projection" | cut -c1-110
astro-env/bin/python - <<PYEOF 2>&1 | grep -v -i "warn\|radesys\|Source redshift\|DEFAULT ZS"
import json, sys
sys.path[:0] = ['$R/sbi', '$R/notebooks', '$R']
from pathlib import Path
import systematics as sy
gp = json.load(open('$R/notebooks_jauzac/plots/gp_posterior_ba_final.json'))
out = json.load(open('$R/$O/systematics.json'))
print("projection rows:", len(out.get("projection", [])))
sy.plot_ranges({**out, "contamination": [r for r in out.get("contamination", []) if abs(r["value"]) <= sy.CONTAMINATION_MAX + 1e-9]},
               Path('$R/notebooks_jauzac/plots/systematics_model_paper_ba_final'), list(sy.MODEL_GROUPS), None, show_total=False, show_shifts=False,
               stat=(gp["lo"], gp["hi"]), centre=gp["ml"])
sy.plot_data(out, Path('$R/notebooks_jauzac/plots/systematics_data_paper_ba_final'), paper=True, skip=("galaxy_low_mag", "galaxy_high_mag", "galaxy_size"))
print("plots done")
PYEOF
cp notebooks_jauzac/plots/systematics_model_paper_ba_final.pdf /Users/davidharvey/Papers/jwst_a2744/aa_draft/Figures/systematics_model.pdf
cp notebooks_jauzac/plots/systematics_data_paper_ba_final.pdf /Users/davidharvey/Papers/jwst_a2744/aa_draft/Figures/systematics_data.pdf
cd /Users/davidharvey/Papers/jwst_a2744/aa_draft && git add Figures && git commit -q -m "Systematics: projection depth restored" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in $(seq 1 10); do git pull -q --rebase --autostash && git push -q origin main 2>/dev/null && { echo pushed; break; }; sleep 5; done
echo "== $(date +%H:%M) done"
