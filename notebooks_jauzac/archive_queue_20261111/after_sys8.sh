#!/bin/bash
# after the other post-run steps: sensitivity grid with the fixed labels and scale bar, push
cd /Users/davidharvey/Work/ml_check
J=$PWD/notebooks_jauzac/plots; F=/Users/davidharvey/Papers/jwst_a2744/aa_draft/Figures
sleep 120
while pgrep -f "notebooks_jauzac/after_sys6.sh|notebooks_jauzac/after_sys7.sh|plots_for_paper|systematics.py" >/dev/null; do sleep 30; done
astro-env/bin/python sbi/pairs_obs_panel.py --pairs 7 10 12 15 16 --out $J/pairs_obs_sensitivity 2>&1 | grep -i "wrote\|error"
cp $J/pairs_obs_sensitivity.pdf $F/sensitivity_map.pdf
cd $F/.. && git add Figures && git commit -q -m "Sensitivity map: A2744 label on one line" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in $(seq 1 10); do git pull -q --rebase --autostash && git push -q origin main 2>/dev/null && { echo pushed; break; }; sleep 5; done
