# shared by the 4 Nov pipelines: same commands as sbi/k0p7_pgc.sh
cd /Users/davidharvey/Work/ml_check
PY=$PWD/astro-env/bin/python
SBI_PY=../Dark_Matter_DA/sbi/bin/python
R=sbi/results
step() { name=$1; shift; echo "== $(date '+%d %H:%M') $name"; "$@" > $LOG/$name.txt 2>&1 || echo "   FAILED (see $LOG/$name.txt)"; }
summaries() { # name model inc outdir domains...
    local name=$1 model=$2 inc=$3 out=$4; shift 4
    step $name $PY -u sbi/make_summaries.py --model $model --include-str $inc --n-noise 5 --no-latents --domains "$@" --out-dir $out
}
sbi_response() { # tag model inc summaries alldraws_base combined_out label
    local t=$1 MODEL=$2 INC=$3 S=$4 B=$5 out=$6 label=$7
    step centre_uncertainty_$t $PY sbi/centre_uncertainty.py --model $MODEL --include-str $INC
    step sbi_${t}_nc2plus $SBI_PY -u sbi/run_sbi.py --summaries $S --noise all --a2744-ncomp 2+ --retrain --out-dir ${B}_nc2plus
    for nc in 3 3+ 4+; do
        d=${B}_nc${nc/+/plus}; mkdir -p "$d"; cp ${B}_nc2plus/nle_*.pt ${B}_nc2plus/npe_conditional_*.pt "$d"/
        step sbi_${t}_nc${nc/+/plus} $SBI_PY -u sbi/run_sbi.py --summaries $S --noise all --a2744-ncomp "$nc" --out-dir "$d"
    done
    grep -h -E "NLE:|conditional|empirical" $LOG/sbi_${t}_*.txt | grep -v -i warn
    local C=$R/centre_uncertainty/${MODEL}_${INC}.json
    step response_$t $PY sbi/response_plot.py --summaries $S --label "$label, ba_$t" --centre-json $C
    for nc in 2 3; do
        step response_combined_${t}_nc$nc $PY sbi/response_combined.py --summaries $S --centre-json $C --min-ncomp $nc \
            --label "$label, ba_$t (30 models)" --out $out
    done
}
