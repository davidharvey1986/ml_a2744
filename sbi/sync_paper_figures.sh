#!/bin/bash
# Copy the paper figures (notebooks/plots, from plots_for_paper.py) into the Overleaf clone under the
# names aa_a2744.tex uses; only figures that exist are copied. Commit/push is done by hand.
#
#   bash sbi/sync_paper_figures.sh [tag=final]
cd "$(dirname "$0")/../notebooks/plots"
T=${1:-final}
DST=/Users/davidharvey/Papers/jwst_a2744/aa_draft/Figures
while read src dst; do
    if [ -f "$src" ]; then cp -p "$src" "$DST/$dst" && echo "copied $src -> $dst"; else echo "missing $src"; fi
done <<MAP
data_examples.png data_examples.png
intrinsic_ell.pdf intrinsic_ell.pdf
final_model_weighting_with_data_ba_$T.pdf final_model_weighting_with_data.pdf
sensitivity_map_ba_$T.pdf sensitivity_map.pdf
Sensitivty_relation_ba_$T.pdf Sensitivty_relation.pdf
output_to_model_ba_$T.pdf output_to_model.pdf
particle_physics.pdf particle_physics.pdf
correlated_models_ba_$T.pdf correlated_models.pdf
latent_overlap_ba_$T.pdf latent_overlap.pdf
goodness_of_fit_ba_$T.pdf goodness_of_fit.pdf
data_systematics_ba_$T.pdf galaxy_selection.pdf
summarised_systematics_ba_$T.png summarised_systematics.png
MAP
