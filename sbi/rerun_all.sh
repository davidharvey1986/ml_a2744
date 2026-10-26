#!/bin/bash
# Rerun everything downstream of add_shear_to_data.py:
#   summaries for the six model groups -> conditioned SBI (ba best + n_comp
#   sensitivity, ba final) -> model-group comparison -> noise analysis -> systematics.
#
# Usage (from ml_check/):
#   bash sbi/rerun_all.sh                      # everything
#   SKIP_SYSTEMATICS=1 bash sbi/rerun_all.sh   # without the (~2 h) systematics suite
#   WAIT_FOR_PID=<pid> bash sbi/rerun_all.sh   # first wait for add_shear_to_data to finish
#
# Needs: astro-env (CNNs, summaries) and the Dark_Matter_DA/sbi environment (sbi package).
set -u
cd "$(dirname "$0")/.."
PY=astro-env/bin/python
SBI_PY=../Dark_Matter_DA/sbi/bin/python
R=sbi/results

step() { echo "##### $(date +%H:%M) $*"; }
filt() { tr '\r' '\n' | grep -e "models, seeds" -e "A2744:" -e "===" -e "clusters x" -e empirical \
                              -e "NLE:" -e conditional -e Error -e Traceback; }

if [ -n "${WAIT_FOR_PID:-}" ]; then
    step "waiting for process $WAIT_FOR_PID (add_shear_to_data) to finish"
    # maps must be newer than the start of that process (ps etime: [[dd-]hh:]mm:ss)
    et=$(ps -p "$WAIT_FOR_PID" -o etime= | tr -d ' '); d=0
    [[ $et == *-* ]] && { d=${et%%-*}; et=${et#*-}; }
    IFS=: read -r -a p <<< "$et"; secs=0; for v in "${p[@]}"; do secs=$((secs * 60 + 10#$v)); done
    START=$(( $(date +%s) - secs - d * 86400 ))
    while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do sleep 60; done
    stale=$(find data/100/obs/concat -name "bahamas*.pkl" -o -name "darkskies*.pkl" | while read f; do
                [ "$(stat -f %m "$f")" -lt "$START" ] && echo "$f"; done)
    if [ -n "$stale" ]; then
        echo "These obs maps were not rewritten after $(date -r "$START" +%H:%M), stopping:"; echo "$stale"; exit 1
    fi
fi

for g in ba_best ba_final; do
    step "summaries $g (5 noise draws)"
    $PY sbi/make_summaries.py --model tweakedalign_$g --out-dir sbi/summaries_${g}_n5 --n-noise 5 --no-latents 2>&1 | filt
done
for pair in "best:summaries" "final:summaries_final" "ta_final:summaries_ta_final"; do
    g=${pair%%:*}; d=${pair#*:}
    step "summaries $g"
    $PY sbi/make_summaries.py --model tweakedalign_$g --out-dir sbi/$d --no-latents 2>&1 | filt
done
step "summaries ta_best (with latents)"
$PY sbi/make_summaries.py --model tweakedalign_ta_best --out-dir sbi/summaries_ta 2>&1 | filt

run_sbi() { $SBI_PY -u sbi/run_sbi.py --noise all "$@" 2>&1 | grep -v -i warn | filt; }
step "SBI ba_best, n_comp 3+"
run_sbi --summaries sbi/summaries_ba_best_n5 --retrain
B=summaries_ba_best_n5_alldraws
for nc in 3 2+ 4+; do                     # same estimators, different conditioning
    d=$R/${B}_nc${nc/+/plus}; mkdir -p "$d"; cp $R/$B/nle_*.pt $R/$B/npe_conditional_*.pt "$d"/
    step "SBI ba_best, n_comp $nc"
    run_sbi --summaries sbi/summaries_ba_best_n5 --a2744-ncomp "$nc"
done
step "SBI ba_final, n_comp 3+"
run_sbi --summaries sbi/summaries_ba_final_n5 --retrain

step "model-group comparison"
$PY sbi/compare_model_groups.py 2>&1 | grep -v -i warn | tail -9
step "noise analysis"
$PY sbi/noise_variance.py sbi/summaries_ba_best_n5 sbi/summaries_ba_final_n5 2>&1 | grep -v -i warn | tail -25

if [ -z "${SKIP_SYSTEMATICS:-}" ]; then
    step "systematics"
    $PY -u sbi/systematics.py --redo 2>&1 | grep -v -i -e warn -e Missing -e "DEFAULT ZS" -e "Source redshift" \
        | tr '\r' '\n' | grep -v "it/s\]\|s/it\]\|^ *$"
fi
step "done"
