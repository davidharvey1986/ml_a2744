#!/bin/bash
# Figures 2 (without and with A2744) and 7 with P(SIDM) labels, Jauzac mass: after figure 1, once the ratio scan has imported
cd /Users/davidharvey/Work/ml_check
until grep -q "== .* ratio" notebooks_jauzac/queue.log; do sleep 15; done; sleep 90
cd notebooks_jauzac
F() { ../astro-env/bin/python paper_fig.py "$@" 2>&1 | grep -i "saved\|error\|traceback"; }
F figure2_summaries 'p.paper_paths("final")["summaries"]' '{d: p.paper_paths("final")["flamingo"] for d in ("flamingo", "tng")}' 2 '"ba_final"' False
F figure2_summaries 'p.paper_paths("final")["summaries"]' '{d: p.paper_paths("final")["flamingo"] for d in ("flamingo", "tng")}' 2 '"ba_final_with_a2744"' True
F figure5_summaries '"final"' 2 True
ls -l plots/final_model_weighting_with_data_ba_final*.png plots/output_to_model_ba_final.png
