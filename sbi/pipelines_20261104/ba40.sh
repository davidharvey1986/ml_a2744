#!/bin/bash
# The 40-epoch last stage (run name ..._ctr_wb1_ba_40, trained on bonsai 5 Nov): once all 30 ba_40 best/final
# checkpoints are in models/concat and the wb1 fiducial pipeline (sbi/k0p7_wb1.sh) has finished (both rewrite
# the A2744 catalogue), make the summaries -> sbi/summaries_wb1_40_<t>_n5 and compare A2744 P(SIDM) with wb1.
# Then redo the wb1 ba_best systematics cases that may have read a catalogue rewritten at 08:05 / 08:10 by
# an import of systematics.py (archived to results/archive_systematics_wb1_best_0810), and redraw the data
# systematics plots with the galaxy-number grey bands.
#
#   nohup bash sbi/pipelines_20261104/ba40.sh > sbi/pipelines_20261104/ba40.log 2>&1 &
source "$(dirname "$0")/common.sh"
caffeinate -is -w $$ &
LOG=$PWD/sbi/logs_ba40; mkdir -p $LOG
INC=spin2_cx_pgc_ft_

echo "== $(date '+%d %H:%M') waiting for the ba_40 checkpoints and the wb1 fiducial pipeline"
until [ $(ls models/concat/*${INC}tweakedalign_k0p7_ctr_wb1_ba_40_final.pth 2>/dev/null | wc -l) -ge 30 ] &&
      [ $(ls models/concat/*${INC}tweakedalign_k0p7_ctr_wb1_ba_40_best.pth 2>/dev/null | wc -l) -ge 30 ] &&
      grep -q "== .* done" sbi/pipelines_20261104/wb1_fiducial.log && ! pgrep -f k0p7_wb1.sh > /dev/null; do
    sleep 300
done

for t in final best; do
    summaries summaries_$t tweakedalign_k0p7_ctr_wb1_ba_40_$t $INC $PWD/sbi/summaries_wb1_40_${t}_n5 bahamas darkskies
done
$PY - <<'EOF'
import numpy as np
for t in ["final", "best"]:
    a = np.load(f"sbi/summaries_wb1_{t}_n5/obs.npz")["softmax"][:, 0]
    b = np.load(f"sbi/summaries_wb1_40_{t}_n5/obs.npz")["softmax"][:, 0]
    print(f"A2744 P(SIDM) ba_{t}: 20 epochs {a.mean():.3f} (median {np.median(a):.3f}, sd {a.std():.3f}) -> "
          f"40 epochs {b.mean():.3f} (median {np.median(b):.3f}, sd {b.std():.3f})")
EOF

S=$R/systematics_wb1_best; A=$R/archive_systematics_wb1_best_0810; mkdir -p $A
for f in fiducial galaxy_low_mag_21.00 galaxy_low_mag_21.30; do [ -e $S/$f.npz ] && mv $S/$f.npz $A/; done
step systematics_wb1_best $PY -u sbi/systematics.py --model tweakedalign_k0p7_ctr_wb1_ba_best --include-str $INC \
    --min-ncomp 2 --summaries $PWD/sbi/summaries_wb1_best_n5 --out-dir $S --dz-summaries $PWD/sbi/summaries_wb1_best_n5_dz
grep -E "^  (fiducial|galaxy_low_mag 21\.[03])" $LOG/systematics_wb1_best.txt
step replot_wb1_final $PY -u sbi/systematics.py --from-cache --model tweakedalign_k0p7_ctr_wb1_ba_final --include-str $INC \
    --min-ncomp 2 --summaries $PWD/sbi/summaries_wb1_final_n5 --out-dir $R/systematics_wb1_final \
    --dz-summaries $PWD/sbi/summaries_wb1_final_n5_dz
step replot_nob1_final $PY -u sbi/systematics.py --from-cache --model tweakedalign_k0p7_ctr_ba_final --include-str $INC \
    --min-ncomp 2 --summaries $PWD/sbi/summaries_spin2cx_final_n5 --out-dir $R/systematics_spin2cx_final \
    --dz-summaries $PWD/sbi/summaries_spin2cx_final_n5_dz
echo "== $(date '+%d %H:%M') all done"
