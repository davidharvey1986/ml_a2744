#!/bin/bash
# push figures 1, 2 and 7 (Jauzac) to Overleaf once figure 1 and fig2_obs.sh are done
J=/Users/davidharvey/Work/ml_check/notebooks_jauzac/plots; F=/Users/davidharvey/Papers/jwst_a2744/aa_draft/Figures
until grep -q "wrote .*intrinsic_ell.png" /Users/davidharvey/Work/ml_check/notebooks_jauzac/queue.log; do sleep 15; done
while pgrep -f fig2_obs.sh >/dev/null; do sleep 15; done
cp $J/intrinsic_ell.pdf $F/intrinsic_ell.pdf
cp $J/final_model_weighting_with_data_ba_final.pdf $F/final_model_weighting_with_data.pdf
cp $J/output_to_model_ba_final.pdf $F/output_to_model.pdf
cd /Users/davidharvey/Papers/jwst_a2744/aa_draft && git add Figures && git status -s Figures && git commit -q -m "Figures 1, 2, 7 (Jauzac mass): profile matched to the Jauzac main-halo mass, P(SIDM) labels

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in $(seq 1 10); do git pull -q --rebase --autostash && git push -q origin main 2>/dev/null && { echo pushed; git log --oneline -1; break; }; sleep 5; done
