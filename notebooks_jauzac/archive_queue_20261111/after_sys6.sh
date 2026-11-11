#!/bin/bash
# after the other post-run steps: figure 1 without the inner-leak reference line, push
cd /Users/davidharvey/Work/ml_check
J=$PWD/notebooks_jauzac/plots; F=/Users/davidharvey/Papers/jwst_a2744/aa_draft/Figures
sleep 150
while pgrep -f "notebooks_jauzac/after_sys4.sh|notebooks_jauzac/after_sys5.sh|plots_for_paper|systematics.py|pairs_obs_panel" >/dev/null; do sleep 30; done
(cd notebooks && ../astro-env/bin/python -c "import plots_for_paper as p; p.figureA1()" 2>&1 | grep -i "error\|member leak")
cp notebooks/plots/intrinsic_ell.{png,pdf} $J/; cp $J/intrinsic_ell.pdf $F/intrinsic_ell.pdf
cd $F/.. && git add Figures && git commit -q -m "Figure 1: leakage panel without the inner-leak reference line" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in $(seq 1 10); do git pull -q --rebase --autostash && git push -q origin main 2>/dev/null && { echo pushed; break; }; sleep 5; done
