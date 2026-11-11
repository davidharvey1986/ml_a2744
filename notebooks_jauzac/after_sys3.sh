#!/bin/bash
# the automatic post-run redraw uses the systematics' own statistical range: redraw the model plot with the GP one, push
cd /Users/davidharvey/Work/ml_check; R=$PWD
sleep 60
while pgrep -f "notebooks_jauzac/after_sys.sh|notebooks_jauzac/after_sys2.sh|notebooks_jauzac/queue9.sh" >/dev/null; do sleep 60; done
astro-env/bin/python - <<PYEOF 2>&1 | grep -v -i "warn\|radesys\|Source redshift"
import json, sys
sys.path[:0] = ['$R/sbi', '$R/notebooks', '$R']
import add_shear_to_data as asd
asd.get_source_redshift = lambda *a, **k: 2.2484412345861355
from pathlib import Path
import systematics as sy
gp = json.load(open('$R/notebooks_jauzac/plots/gp_posterior_ba_final.json'))
out = json.load(open('$R/sbi/results/systematics_wb1_final_jauzac/systematics.json'))
sy.plot_ranges({**out, "contamination": [r for r in out.get("contamination", []) if abs(r["value"]) <= sy.CONTAMINATION_MAX + 1e-9]},
               Path('$R/notebooks_jauzac/plots/systematics_model_paper_ba_final'), list(sy.MODEL_GROUPS), None, show_total=False, show_shifts=False,
               stat=(gp["lo"], gp["hi"]), centre=gp["ml"])
PYEOF
cp notebooks_jauzac/plots/systematics_model_paper_ba_final.pdf /Users/davidharvey/Papers/jwst_a2744/aa_draft/Figures/systematics_model.pdf
cd /Users/davidharvey/Papers/jwst_a2744/aa_draft && git add Figures && git commit -q -m "Model systematics: GP statistical range after the cut rerun" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in $(seq 1 10); do git pull -q --rebase --autostash && git push -q origin main 2>/dev/null && { echo pushed; break; }; sleep 5; done
