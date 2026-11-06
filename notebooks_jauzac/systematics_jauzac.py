# Systematics (data + model plots) re-summarised from the cached network outputs of sbi/results/systematics_wb1_<tag>
# with the Jauzac A2744 mass prior. Nothing is re-evaluated: the cached clusters are the Harvey selection
# (within SUBSET_DEX = 0.45 dex of 14.52), so the Jauzac Gaussian (14.73 +- 0.12) is cut at +2 sigma (14.97).
# Writes sbi/results/systematics_wb1_<tag>_jauzac/{systematics.json, systematics_data, systematics_model}.
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sbi"))
import systematics as sy
from mass_weighting import a2744_mass_gaussian
MU_H, SIG_H = a2744_mass_gaussian("harvey")
MU_J, SIG_J = a2744_mass_gaussian("jauzac")
_select = sy.select_clusters
sy.select_clusters = lambda summaries, domain, mu, min_ncomp=None: _select(summaries, domain, MU_H, min_ncomp)  # match the caches

for tag in sys.argv[1:] or ["final", "best"]:
    S = REPO / "sbi" / f"summaries_wb1_{tag}_n5"
    cache = REPO / "sbi" / "results" / f"systematics_wb1_{tag}"
    out_dir = REPO / "sbi" / "results" / f"systematics_wb1_{tag}_jauzac"; out_dir.mkdir(exist_ok=True)
    sys.argv = ["systematics.py", "--model", f"tweakedalign_k0p7_ctr_wb1_ba_{tag}", "--include-str", "spin2_cx_pgc_ft_",
                "--min-ncomp", "2", "--summaries", str(S), "--out-dir", str(cache), "--dz-summaries", f"{S}_dz", "--from-cache"]
    opts = sy.parse()
    sy.A2744_NCOMP = tuple(range(opts.min_ncomp, 7))
    R = sy.Runner(opts, cache)
    R.maps = {}
    R.mu, R.sig = MU_J, SIG_J
    print(f"== ba_{tag}: Jauzac {MU_J:.3f} +- {SIG_J:.3f} (clusters selected about Harvey {MU_H:.3f})")
    out = sy.run_tests(R, opts.tests)
    out["settings"] = {"model": opts.model, "a2744_ncomp": list(sy.A2744_NCOMP), "a2744_log_mass": [R.mu, R.sig],
                       "mass_study": "jauzac", "cluster_selection_mu": MU_H, "from_cache": str(cache)}
    json.dump(out, open(out_dir / "systematics.json", "w"), indent=2)
    sy.plot_data(out, out_dir / "systematics_data")
    sy.plot_model(out, out_dir / "systematics_model")
    print(f"wrote {out_dir}")
