#!/usr/bin/env python
'''
Ensemble P(SIDM) against halo mass (log10 M750) for each BAHAMAS simulation (CDM, sigma/m = 0.1,
0.3, 1 and velocity-dependent SIDM) and the CDM-only FLAMINGO (fiducial, low and high AGN) and TNG-Cluster, over all noise draws: binned mean with its error and 16-84%
of clusters shaded, for every cluster and for n_comp >= 3, with A2744 at its mass.

    astro-env/bin/python sbi/p_vs_mass.py --tag final
'''
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from mass_weighting import a2744_mass_gaussian  # noqa: E402

# Lenstool convergence (sl_kappa_zs2.28.fits) within 750 kpc of the adopted centre: the map covers 45% of
# the aperture; 14.84 is the covered part alone, 15.15 fills each annulus with its covered mean kappa
LENSTOOL_750 = (15.15, 14.84)
KS_JSON = REPO / "sbi" / "results" / "ks_mass.json"         # sbi/ks_mass.py: calibrated Kaiser-Squires masses
SIMS = [("bahamas", 0, "CDM", "#000000"), ("bahamas", 0.1, "σ/m = 0.1", "#9ECAE1"),
        ("bahamas", 0.3, "σ/m = 0.3", "#4292C6"), ("bahamas", 1, "σ/m = 1", "#08306B"),
        ("vd_bahamas", None, "vdSIDM", "#E41A1C"),
        ("flamingo", 0, "FLAMINGO CDM", "#009E73"), ("flamingo_lo", 0, "FLAMINGO low AGN", "#8FD175"),
        ("flamingo_hi", 0, "FLAMINGO high AGN", "#00593F"), ("tng", 0, "TNG-Cluster CDM", "#CC79A7")]
OTHER = ("vd_bahamas", "flamingo", "flamingo_lo", "flamingo_hi", "tng")   # in the _flamingo summaries


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tags", nargs="+", default=["best", "final"])
    p.add_argument("--fiducial", default="wb1", help="summaries_<fiducial>_<tag>_n5 (nob1: spin2cx)")
    p.add_argument("--bins", type=float, nargs="+", default=list(np.arange(13.7, 15.46, 0.15)))
    p.add_argument("--min-count", type=int, default=10, help="clusters per bin to draw it")
    p.add_argument("--out", default=str(REPO / "sbi" / "results" / "p_vs_mass"))
    p.add_argument("--jauzac-2sigma", action="store_true", help="also mark the Jauzac mass +-2 sigma (hatched, dashed edges)")
    return p.parse_args()


def load(tag, sim, sigma, fid="wb1"):
    d = REPO / "sbi" / (f"summaries_{fid}_{tag}_n5" + ("_flamingo" if sim in OTHER else ""))
    P = np.load(d / f"params_{sim}.npz")
    p = np.load(d / f"softmax_{sim}.npy").mean(0)                    # (n_noise, N) ensemble P(SIDM)
    keep = np.ones(p.shape[1], bool) if sigma is None else np.isclose(P["cross_section"], sigma)
    return p[:, keep], P["log10_mass_750"][keep], P["ncomponents"][keep]


def main():
    opts = parse()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    studies = [(st, *a2744_mass_gaussian(st), c) for st, c in (("harvey", "0.55"), ("jauzac", "#D95F02"))]
    edges = np.array(opts.bins)
    mids = 0.5 * (edges[1:] + edges[:-1])
    cuts = [(1, "all clusters"), (3, "n_comp ≥ 3")]
    fig, axes = plt.subplots(len(opts.tags), len(cuts), figsize=(6.2 * len(cuts), 4.4 * len(opts.tags)),
                             sharex=True, sharey=True, constrained_layout=True, squeeze=False)
    for i, tag in enumerate(opts.tags):
        p_obs = float(np.load(REPO / "sbi" / f"summaries_{opts.fiducial}_{tag}_n5" / "obs.npz")["softmax"][:, 0].mean())
        for j, (nc, title) in enumerate(cuts):
            ax = axes[i, j]
            for st, mu, sig, c in studies:
                ax.axvspan(mu - sig, mu + sig, color=c, alpha=0.18, lw=0)
                if st == "jauzac" and opts.jauzac_2sigma:
                    ax.axvspan(mu - 2 * sig, mu + 2 * sig, facecolor="none", edgecolor=c, hatch="//", alpha=0.35, lw=0)
                    for x in (mu - 2 * sig, mu + 2 * sig):
                        ax.axvline(x, color=c, ls="--", lw=1.2)
                ax.errorbar(mu, p_obs, xerr=sig, fmt="*", color=c, ms=14, zorder=5,
                            label=f"A2744 ({p_obs:.3f}), {st.capitalize()} mass {mu:.2f}")
            lm, lo = LENSTOOL_750
            ax.axvline(lo, color="#1B9E77", ls=":", lw=1.2)
            ax.errorbar(lm, p_obs, xerr=[[lm - lo], [0]], fmt="*", color="#1B9E77", ms=14, zorder=5,
                        label=f"A2744, Lenstool M(<750 kpc) {lm:.2f} (dotted: covered part only, {lo:.2f})")
            if KS_JSON.exists():                # note: a 500 kpc aperture, not M750
                ks = json.load(open(KS_JSON))["a2744"]["500"]
                lm = np.log10(ks["M_calibrated"])
                lo, hi = np.log10(ks["M_calibrated_16_84"])
                noise = ks["M_calibrated_err_noise"] / ks["M_calibrated"] / np.log(10)
                xerr = [[np.hypot(lm - lo, noise)], [np.hypot(hi - lm, noise)]]
                ax.errorbar(lm, p_obs, xerr=xerr, fmt="*", color="#7570B3", ms=14, zorder=6, capsize=3,
                            label=f"A2744, KS M(<500 kpc) {lm:.2f} (calibrated on BAHAMAS)")
            for sim, sigma, name, col in SIMS:
                p, m, ncomp = load(tag, sim, sigma, opts.fiducial)
                sel = (ncomp >= nc) & np.isfinite(m)
                pc, mc = p[:, sel].mean(0), m[sel]                     # per cluster, mean over noise draws
                idx = np.digitize(mc, edges) - 1
                rows = []
                for b in range(len(mids)):
                    x = pc[idx == b]
                    if len(x) >= opts.min_count:
                        rows.append((mids[b], x.mean(), x.std() / np.sqrt(len(x)), *np.percentile(x, [16, 84])))
                if not rows:
                    continue
                r = np.array(rows).T
                ls = "--" if sigma is None else ":" if sim not in ("bahamas", "vd_bahamas") else "-"
                ax.fill_between(r[0], r[3], r[4], color=col, alpha=0.08, lw=0)
                ax.errorbar(r[0], r[1], r[2], fmt="o" + ls, color=col, ms=4, lw=1.8, capsize=2,
                            label=f"{name} (N={sel.sum()})")
            ax.set_title(f"ba_{tag}: {title}", fontsize=10)
            ax.set_xlabel(r"$\log_{10} M_{750}$ [M$_\odot$]")
            ax.set_xlim(13.65, 15.45)
            ax.set_ylim(0.3, 0.85)
            if j == 0:
                ax.set_ylabel("ensemble P(SIDM)")
            h, l = ax.get_legend_handles_labels()
            star = [i for i, t in enumerate(l) if t.startswith("A2744")]
            ax.legend([h[i] for i in star], [l[i] for i in star], fontsize=7.5, loc="upper left", frameon=False)
            sims = [(h[i], t.split(" (N=")[0]) for i, t in enumerate(l) if i not in star]
    fig.legend(*zip(*sims), loc="outside lower center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle("P(SIDM) against halo mass: BAHAMAS solid, vdSIDM dashed, FLAMINGO/TNG CDM dotted\n(binned mean ± error; shaded: 16-84% of clusters; bands: A2744 mass ±1σ, Harvey grey, Jauzac orange" + (", hatched/dashed ±2σ" if opts.jauzac_2sigma else "") + "; green: Lenstool; purple: KS)",
                 fontsize=11)
    for e in ("png", "pdf"):
        fig.savefig(f"{opts.out}.{e}", dpi=150)
    print(f"wrote {opts.out}.png")


if __name__ == "__main__":
    main()
