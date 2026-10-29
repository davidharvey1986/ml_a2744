#!/usr/bin/env python
'''
Response functions for the masses of A2744's three halos separately (mass_weighting.A2744_MASSES:
core = Lenstool O2, NW = O1, N = O3) next to the summed mass used everywhere else.

Each curve is response_plot.curves with the cluster weights N(log M750; mu, sigma) / n(log M750)
centred on one halo's mass. The quoted mass errors are 0.02-0.05 dex, which would leave a handful
of clusters, so sigma is floored at --min-sigma. Curves whose mass lies outside the simulated
range, or with fewer than --min-neff effective clusters per sigma/m, are left out. Note that the
halo masses are compared with the simulations' M(<750 kpc) of the whole cluster.

Optionally the P(SIDM) of A2744 with the stamp centred on each Lenstool potential
(sensitivity.py --recentre) is drawn as a dashed line in the halo's colour: it is not calibrated
(the simulations are sampled around the main-halo centre), so it is for orientation only.

    astro-env/bin/python sbi/halo_response.py --summaries sbi/summaries_spin2_best_n5 --label "spin2 best"
'''
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "sbi")]
from mass_weighting import A2744_ERRORS, A2744_MASSES, a2744_mass_gaussian  # noqa: E402
from response_plot import X_CDM, curves  # noqa: E402

HALOS = {"core": ("O2", "#000000"), "nw": ("O1", "#CC79A7"), "n": ("O3", "#009E73")}


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summaries", required=True)
    p.add_argument("--study", default="harvey", choices=sorted(A2744_MASSES))
    p.add_argument("--min-ncomp", type=int, default=2)
    p.add_argument("--min-sigma", type=float, default=0.1, help="floor on the width of each halo's log-mass Gaussian")
    p.add_argument("--min-neff", type=float, default=5)
    p.add_argument("--recentred", default=str(REPO / "sbi" / "results" / "sensitivity_spin2_best_centre_{}" / "sensitivity.json"),
                   help="sensitivity.json per Lenstool halo ({} = O1, O2, O3); '' to skip")
    p.add_argument("--label", default=None)
    return p.parse_args()


def main():
    opts = parse()
    S_dir = Path(opts.summaries).resolve()
    out = S_dir.parent / "results" / S_dir.name / f"halo_response_{opts.study}"
    obs = np.load(S_dir / "obs.npz")["softmax"]
    p_a = float(obs[:, 0].mean()) if obs.ndim == 2 else float(obs[:, 0, 0].mean())

    gauss = {"total": a2744_mass_gaussian(opts.study)}
    for k, m in A2744_MASSES[opts.study].items():
        lo, hi = A2744_ERRORS[opts.study][k]
        gauss[k] = (np.log10(m * 1e13), 0.5 * np.log10((m + hi) / (m - lo)))
    res = {"study": opts.study, "min_sigma": opts.min_sigma, "min_ncomp": opts.min_ncomp, "a2744_p_sidm": p_a,
           "mass_gaussians": {k: list(map(float, v)) for k, v in gauss.items()}}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True, sharey=True)
    for ax, dom in zip(axes, ("bahamas", "darkskies")):
        m = np.load(S_dir / f"params_{dom}.npz")["log10_mass_750"]
        lo, hi = np.nanmin(m), np.nanmax(m)
        res[dom] = {}
        for k, (mu, sig) in gauss.items():
            name, col = ("total (O1+O2+O3)", "#0072B2") if k == "total" else (f"{k} ({HALOS[k][0]})", HALOS[k][1])
            s = max(sig, opts.min_sigma)
            if not lo <= mu <= hi:
                print(f"{dom} {k}: log M {mu:.2f} outside the simulated range {lo:.2f}-{hi:.2f}, skipped")
                res[dom][k] = None
                continue
            c, *_ = curves(S_dir, dom, opts.min_ncomp, mu, s)
            if min(c["n_eff"]) < opts.min_neff:
                print(f"{dom} {k}: n_eff {min(c['n_eff']):.1f} < {opts.min_neff:g}, skipped")
                res[dom][k] = None
                continue
            res[dom][k] = c
            x = np.where(np.array(c["sigma"]) == 0, X_CDM, c["sigma"])
            ax.fill_between(x, c["q16"], c["q84"], color=col, alpha=0.08, lw=0)
            ax.errorbar(x, c["all"], c["all_err"], fmt="o-", color=col, ms=5, capsize=2, lw=2 if k == "total" else 1.5,
                        label=f"{name}: log M {mu:.2f} ± {s:.2f}, n_eff {min(c['n_eff']):.0f}-{max(c['n_eff']):.0f}")
            print(f"{dom} {k:5s} log M {mu:.2f}±{s:.2f}: " + " | ".join(
                f"σ={sg:g} {a:.3f}±{e:.3f} (n_eff {n:.0f})" for sg, a, e, n in zip(c["sigma"], c["all"], c["all_err"], c["n_eff"])))
        ax.axhline(p_a, color="#0072B2", lw=1.5, ls="-", alpha=0.6, label=f"A2744, adopted centre: {p_a:.3f}")
        if opts.recentred:
            for k, (h, col) in HALOS.items():
                f = Path(opts.recentred.format(h))
                if f.exists():
                    p = json.load(open(f))["a2744_p_sidm"]
                    ax.axhline(p, color=col, lw=1.2, ls="--", label=f"A2744 centred on {h} (uncalibrated): {p:.3f}")
        ax.set_xscale("log")
        ax.set_xlim(0.003, 1.4)
        ax.set_xticks([X_CDM, 0.01, 0.05, 0.1, 0.2, 0.3, 1], ["CDM", "0.01", "0.05", "0.1", "0.2", "0.3", "1"])
        ax.minorticks_off()
        ax.set_xlabel("σ/m [cm²/g]")
        ax.set_title(f"{dom.upper()} (simulated log M750 {lo:.2f}-{hi:.2f}), n_comp ≥ {opts.min_ncomp}", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("ensemble P(SIDM), mass-matched")
    fig.suptitle(f"{opts.label or S_dir.name}: response for each A2744 halo mass ({opts.study}; "
                 f"widths floored at {opts.min_sigma} dex; 16-84% of clusters shaded)", fontsize=10)
    for e in ("png", "pdf"):
        fig.savefig(out.with_suffix("." + e), dpi=150)
    json.dump(res, open(out.with_suffix(".json"), "w"), indent=1)
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
