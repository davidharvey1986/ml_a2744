#!/bin/bash
# All paper plots with the Harvey mass, core rescaled to the new Lenstool O2 (mass_weighting "harvey_g1v2"), final model.
# One step at a time (each rewrites the shared catalogue). Plots in notebooks_harvey_g1v2/plots/.
cd /Users/davidharvey/Work/ml_check
PY=$PWD/astro-env/bin/python; SBI_PY=$PWD/../Dark_Matter_DA/sbi/bin/python
D=$PWD/notebooks_harvey_g1v2; P=$D/plots; R=$PWD/sbi/results; S=$PWD/sbi/summaries_wb1_final_n5
F="grep -v -i warn|radesys|radecsys|DEFAULT ZS|Source redshift"
step() { echo "== $(date '+%H:%M') $1"; }
cp notebooks_jauzac/plots/intrinsic_ell.{png,pdf} notebooks_jauzac/plots/correlated_models_ba_final.{png,pdf} $P/
cp sbi/results/main_halo_profile/main_halo_sims.{png,pdf} $P/
step figure4; (cd $D && $PY paper_fig.py figure2_summaries 'p.paper_paths("final")["summaries"]' \
    '{d: p.paper_paths("final")["flamingo"] for d in ("flamingo", "tng")}' 2 '"ba_final"' False) 2>&1 | grep -i "saved\|error"
step figure8; (cd $D && $PY - <<'PYEOF'
import os, sys
sys.path[:0] = ["../notebooks", ".."]
import mass_weighting
_g = mass_weighting.a2744_mass_gaussian
mass_weighting.a2744_mass_gaussian = lambda study=None: _g("harvey_g1v2")
import numpy as np
import plots_for_paper as p
os.chdir(os.path.dirname(os.path.abspath("paper_fig.py")))
mu, sig = mass_weighting.a2744_mass_gaussian()
paths = p.paper_paths("final")
sigmas, curve = p.combined_curve(paths["summaries"], mu, sig, 2)
p_obs = np.load(f"{paths['summaries']}/obs.npz")["softmax"][:, 0]
gp, x, post, (med, lo, hi) = p.sigma_posterior(sigmas, curve, p_obs)
ml = x[np.argmax(post)]
p.figure6(ml, np.array([ml - lo, hi - ml]), merger=False)
PYEOF
) 2>&1 | grep -i "sigma\|error"
step p_vs_mass; $PY sbi/p_vs_mass.py --tags final --ncomp 2 --no-lenstool-ks --harvey harvey_g1v2 --no-jauzac --xlim 14 15.1 \
    --no-title --out $P/p_vs_mass_ba_final | tail -1
step pooled_tail; $PY sbi/cdm_pooled_tail.py --studies harvey_g1v2 jauzac --out $P/cdm_pooled_tail_final_nc2 2>&1 | grep "pooled\|wrote"
step empirical_posterior; $PY sbi/empirical_posterior.py --mass-study harvey_g1v2 --out $P/empirical_posterior_ba_final | tail -4
step sbi; MASS_STUDY=harvey_g1v2 $SBI_PY -u sbi/run_sbi_jauzac.py --summaries sbi/summaries_wb1_final_n5 --noise all \
    --a2744-ncomp 2+ --retrain --out-dir $R/summaries_wb1_final_n5_alldraws_nc2plus_harvey_g1v2 > $D/sbi.log 2>&1
cp $R/summaries_wb1_final_n5_alldraws_nc2plus_harvey_g1v2/{posterior,discrepancy}.png $P/ 2>/dev/null; tail -2 $D/sbi.log
step sensitivity_panel; $PY sbi/pairs_obs_panel.py --pairs 7 10 12 15 16 --out $P/pairs_obs_sensitivity 2>&1 | grep "wrote\|Error"
step systematics; O=$R/systematics_wb1_final_harvey_g1v2
$PY -u sbi/systematics.py --model tweakedalign_k0p7_ctr_wb1_ba_final --include-str spin2_cx_pgc_ft_ --min-ncomp 2 \
    --summaries $S --out-dir $O --dz-summaries ${S}_dz --mass-study harvey_g1v2 > $D/systematics.log 2>&1
tail -3 $D/systematics.log
$PY - <<PYEOF 2>&1 | grep -v -i "warn|radesys"
import json, sys
sys.path.insert(0, 'sbi')
from pathlib import Path
import systematics as sy
out = json.load(open('$O/systematics.json'))
P = Path('$P')
sy.plot_ranges({**out, "contamination": [r for r in out.get("contamination", []) if abs(r["value"]) <= sy.CONTAMINATION_MAX + 1e-9]},
               P / 'systematics_model_ba_final', list(sy.MODEL_GROUPS), None, show_total=False, show_shifts=False)
sy.plot_data(out, P / 'systematics_data_ba_final', paper=True, skip=("galaxy_low_mag", "galaxy_high_mag", "galaxy_size"))
print('systematics plots done')
PYEOF
echo "== $(date '+%H:%M') ALL_DONE"
