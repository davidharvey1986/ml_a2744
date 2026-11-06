#!/bin/bash
# 4 Nov: wb1 pipeline, then the rest of the nob1 seed-12 rerun, then the wb1 vs nob1 comparison. One catalogue rewriter at a time.
D=$(cd $(dirname $0) && pwd)
caffeinate -is -w $$ &
bash $D/wb1.sh && bash $D/nob1_seed12_rest.sh && bash $D/compare.sh
echo "== $(date '+%d %H:%M') all done"
