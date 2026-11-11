#!/usr/bin/env python
'''
Distribution of the ensemble-mean P(SIDM) of one simulation's clusters at each cross-section (every noise draw),
n_comp >= --min-ncomp, weighted to the A2744 mass (gaussian_mass_weights, --mass-study), with A2744's value.

    astro-env/bin/python sbi/p_sidm_distributions.py --sim darkskies
'''
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "sbi")]
from mass_weighting import a2744_mass_gaussian, gaussian_mass_weights  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", default="darkskies")
    p.add_argument("--summaries", default=str(REPO / "sbi" / "summaries_wb1_final_n5"))
    p.add_argument("--mass-study", default="jauzac")
    p.add_argument("--min-ncomp", type=int, default=2)
    p.add_argument("--out", default=None)
    opts = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = Path(opts.summaries)
    P = np.load(S / f"params_{opts.sim}.npz")
    pc = np.load(S / f"softmax_{opts.sim}.npy").mean(0)               # (n_noise, N) ensemble mean
    p_obs = np.load(S / "obs.npz")["softmax"][:, 0].mean()
    mu, sig = a2744_mass_gaussian(opts.mass_study)
    cs, m, nc = P["cross_section"], P["log10_mass_750"], P["ncomponents"]
    sigmas = np.unique(cs)
    cmap = plt.get_cmap("viridis")
    bins = np.linspace(0.2, 0.9, 36)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for k, s in enumerate(sigmas):
        rows = (cs == s) & (nc >= opts.min_ncomp) & np.isfinite(m)
        w = gaussian_mass_weights(m[rows], mu, sig)
        x = pc[:, rows].ravel()
        ww = np.tile(w, pc.shape[0])
        col = "k" if s == 0 else cmap(k / max(len(sigmas) - 1, 1))
        mean = np.sum(x * ww) / ww.sum()
        ax.hist(x, bins=bins, weights=ww, density=True, histtype="step", lw=2, color=col,
                label=("CDM" if s == 0 else rf"$\sigma/m={s:g}$") + f" (N={rows.sum()}, mean {mean:.3f})")
        print(f"{opts.sim} sigma/m={s:g}: {rows.sum()} clusters, weighted mean P(SIDM) {mean:.3f}")
    ax.axvline(p_obs, color="#D55E00", lw=2, ls="--", label=f"A2744 ({p_obs:.2f})")
    ax.set_xlabel("Ensemble P(SIDM)", fontsize=13)
    ax.set_ylabel("Probability distribution", fontsize=13)
    ax.legend(fontsize=9)
    out = opts.out or str(REPO / "sbi" / "results" / f"p_sidm_distributions_{opts.sim}_{S.name}_{opts.mass_study}")
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=180)
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
