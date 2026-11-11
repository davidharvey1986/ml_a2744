#!/usr/bin/env python
'''
One-tailed probability that a simulated cluster has an ensemble P(SIDM) at least as large as A2744's, for every
simulated cross-section: clusters with n_comp >= --min-ncomp, weighted to the A2744 mass (gaussian_mass_weights),
every noise draw a sample; plotted as the one-tailed Gaussian sigma with a bootstrap over clusters. CDM-only
simulations (FLAMINGO, TNG) are stars at the CDM position, vdSIDM a band.

    astro-env/bin/python sbi/tail_vs_cross_section.py
'''
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from mass_weighting import a2744_mass_gaussian, gaussian_mass_weights  # noqa: E402

SIMS = [("bahamas", "", "BAHAMAS", "#56B4E9"), ("darkskies", "", "DARKSKIES", "#E69F00"),
        ("flamingo", "_flamingo", "FLAMINGO", "#D55E00"), ("tng", "_flamingo", "TNG", "#009E73"),
        ("vd_bahamas", "_flamingo", "vdSIDM", "#E41A1C")]
CDM_X = 3e-3


def tail(p, w, p_obs):
    return float(np.mean([(w * (row >= p_obs)).sum() for row in p]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="final")
    ap.add_argument("--min-ncomp", type=int, default=2)
    ap.add_argument("--sig-scale", type=float, default=1.0, help="multiply the A2744 mass error (2: twice the Jauzac width)")
    ap.add_argument("--exclude", nargs="*", default=["vd_bahamas"], help="simulations left out")
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--out", default=str(REPO / "notebooks_jauzac" / "plots" / "tail_vs_cross_section"))
    opts = ap.parse_args()
    mu, sig = a2744_mass_gaussian()
    sig *= opts.sig_scale
    print(f"A2744 mass {mu:.3f} +- {sig:.3f} dex")
    base = REPO / "sbi" / f"summaries_wb1_{opts.tag}_n5"
    p_obs = float(np.load(base / "obs.npz")["softmax"][:, 0].mean())
    rng = np.random.default_rng(0)
    res = {}
    sims = [x for x in SIMS if x[0] not in opts.exclude]
    for sim, suffix, name, col in sims:
        d = Path(f"{base}{suffix}")
        P = np.load(d / f"params_{sim}.npz")
        p = np.load(d / f"softmax_{sim}.npy").mean(0)                  # (n_noise, N): mean over the models
        res[sim] = []
        for c in np.unique(P["cross_section"]):
            rows = (P["cross_section"] == c) & (P["ncomponents"] >= opts.min_ncomp) & np.isfinite(P["log10_mass_750"])
            w = gaussian_mass_weights(P["log10_mass_750"][rows], mu, sig)
            pp, w = p[:, rows], w / w.sum()
            f = tail(pp, w, p_obs)
            boot = []
            for _ in range(opts.nboot):
                i = rng.integers(0, len(w), len(w))
                boot.append(norm.isf(np.clip(tail(pp[:, i], w[i] / w[i].sum(), p_obs), 1e-6, 1 - 1e-6)))
            lo, hi = np.percentile(boot, [16, 84])
            res[sim].append({"sigma_m": float(c), "p": f, "z": float(norm.isf(f)), "z_lo": float(lo), "z_hi": float(hi),
                             "n": int(rows.sum())})
            print(f"{name:10s} sigma/m {c:<5g} p {100 * f:5.2f}%  {norm.isf(f):+.2f} sigma [{lo:+.2f}, {hi:+.2f}]")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401
    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
    for sim, _, name, col in sims:
        r = res[sim]
        z, lo, hi = (np.array([x[k] for x in r]) for k in ("z", "z_lo", "z_hi"))
        if sim == "vd_bahamas":
            ax.axhspan(lo[0], hi[0], color=col, alpha=0.15, lw=0)
            ax.axhline(z[0], color=col, ls="--", lw=1.5, label=name)
            continue
        x = np.array([CDM_X if v["sigma_m"] == 0 else v["sigma_m"] for v in r])
        yerr = np.array([z - lo, hi - z])
        if len(r) > 1:
            ax.errorbar(x, z, yerr, fmt="-o", color=col, ms=7, capsize=2, label=name)
            ax.errorbar(x[:1], z[:1], yerr[:, :1], fmt="*", color=col, ms=15, capsize=2)
        else:
            ax.errorbar(x, z, yerr, fmt="*", color=col, ms=15, capsize=2, label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(1.5e-3, 1.6)
    ax.set_xticks([CDM_X, 1e-2, 1e-1, 1], ["CDM", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"])
    ax.set_xlabel(r"Self-interaction cross-section $[\mathrm{cm}^2\,\mathrm{g}^{-1}]$", fontsize=15)
    ax.set_ylabel(r"A2744 tension, one-tailed [$\sigma$]", fontsize=15)
    ax.legend(fontsize=10, loc="upper right", ncols=2)
    for ext in ("png", "pdf"):
        fig.savefig(f"{opts.out}.{ext}", dpi=200)
    json.dump({"p_obs": p_obs, "mass": [mu, sig], "results": res}, open(f"{opts.out}.json", "w"), indent=1)
    print(f"wrote {opts.out}.png")


if __name__ == "__main__":
    main()
