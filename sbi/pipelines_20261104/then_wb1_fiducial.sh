#!/bin/bash
# After all.sh (nob1 seed-12 rerun + wb1/nob1 comparison): switch response_combined.py's defaults to wb1, then
# build the rest of the wb1 fiducial products with sbi/k0p7_wb1.sh.
cd /Users/davidharvey/Work/ml_check
until grep -q "all done" sbi/pipelines_20261104/all.log; do sleep 120; done
sed -i '' -e 's#summaries_spin2cx_best_n5#summaries_wb1_best_n5#' \
    -e 's#tweakedalign_k0p7_ctr_ba_best_spin2_cx_pgc_ft_.json#tweakedalign_k0p7_ctr_wb1_ba_best_spin2_cx_pgc_ft_.json#' \
    -e 's#"results" / "chifix" / "response_combined"#"results" / "wb1" / "response_combined"#' sbi/response_combined.py
echo "== $(date '+%d %H:%M') response_combined.py defaults -> wb1"
bash sbi/k0p7_wb1.sh
