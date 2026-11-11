#!/bin/bash
# after the cut-systematics run: sensitivity panel (catalogue) and push systematics + sensitivity to Overleaf
cd /Users/davidharvey/Work/ml_check
J=$PWD/notebooks_jauzac/plots; F=/Users/davidharvey/Papers/jwst_a2744/aa_draft/Figures
until grep -q "ALL_DONE" notebooks_jauzac/queue9.log; do sleep 60; done
astro-env/bin/python sbi/pairs_obs_panel.py --pairs 7 10 12 15 16 --out $J/pairs_obs_sensitivity 2>&1 | grep -i "wrote\|error"
cp $J/pairs_obs_sensitivity.pdf $F/sensitivity_map.pdf
cp $J/systematics_model_paper_ba_final.pdf $F/systematics_model.pdf
cp $J/systematics_data_paper_ba_final.pdf $F/systematics_data.pdf
cd $F/.. && git add Figures && git commit -q -m "Systematics (magnitude/size cuts rebuilt from the raw catalogues, Jauzac); sensitivity map row label CDM" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in $(seq 1 10); do git pull -q --rebase --autostash && git push -q origin main 2>/dev/null && { echo pushed; break; }; sleep 5; done
