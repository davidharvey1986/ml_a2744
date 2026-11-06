'''wb1 vs nob1: A2744 P(SIDM) per seed and SBI significances, ba_final and ba_best.'''
import json
from pathlib import Path
import numpy as np
S = Path(__file__).resolve().parents[1]
for t in ("final", "best"):
    p = {}
    for g, d in (("nob1", f"summaries_spin2cx_{t}_n5"), ("wb1", f"summaries_wb1_{t}_n5")):
        info = json.load(open(S / d / "info.json"))
        p[g] = dict(zip(info["seeds"], np.load(S / d / "obs.npz")["softmax"][:, 0]))
    seeds = sorted(set(p["nob1"]) & set(p["wb1"]))
    a, b = np.array([p["nob1"][s] for s in seeds]), np.array([p["wb1"][s] for s in seeds])
    print(f"\nba_{t}: A2744 P(SIDM), {len(seeds)} seeds")
    print(f"  nob1 {a.mean():.3f} ± {a.std():.3f}   wb1 {b.mean():.3f} ± {b.std():.3f}   "
          f"diff {np.mean(b - a):+.3f} ± {np.std(b - a) / np.sqrt(len(seeds)):.3f}   n(>0.65) {np.sum(a > 0.65)} -> {np.sum(b > 0.65)}")
    print("  seed  " + " ".join(f"{s:>5}" for s in seeds))
    print("  nob1  " + " ".join(f"{x:5.2f}" for x in a))
    print("  wb1   " + " ".join(f"{x:5.2f}" for x in b))
    print(f"  SBI ({'nc':>6}: nob1 -> wb1)      empirical σ     NLE P(CDM)   NPE P(CDM)")
    for nc in ("2plus", "3", "3plus", "4plus"):
        row = []
        for d in (f"summaries_spin2cx_{t}_n5_alldraws_nc{nc}", f"summaries_wb1_{t}_n5_alldraws_nc{nc}"):
            f = S / "results" / d / "results.json"
            if not f.exists():
                row.append(None); continue
            r = json.load(open(f))["bahamas"]
            row.append((r["empirical"]["significance_sigma"], r["nle"]["posterior_p_cdm"], r["npe_conditional"]["posterior_p_cdm"]))
        if None in row:
            print(f"  {nc:>6}: missing"); continue
        print(f"  {nc:>6}: " + "   ".join(f"{x:.2f} -> {y:.2f}" for x, y in zip(*row)))
