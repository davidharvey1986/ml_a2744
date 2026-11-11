#!/usr/bin/env python
'''
Posterior on sigma/m over the simulated cross-sections from the empirical likelihood: at each cross-section, a
weighted Gaussian KDE of logit ensemble-mean P(SIDM) over every (cluster, noise draw) with n_comp >= --min-ncomp,
weighted to the A2744 mass (gaussian_mass_weights, --mass-study), evaluated at A2744's logit P(SIDM); flat prior
over the simulated values. Combined: the two simulations' weighted samples pooled (each normalised to equal weight)
at every cross-section either has.

    astro-env/bin/python sbi/empirical_posterior.py
'''
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "sbi")]
from mass_weighting import a2744_mass_gaussian, gaussian_mass_weights  # noqa: E402

SIMS = ("bahamas", "darkskies")


def samples(S, d, mu, sig, min_ncomp):
    P = np.load(S / f"params_{d}.npz")
    pc = np.load(S / f"softmax_{d}.npy").mean(0)                     # (n_noise, N) ensemble mean
    out = {}
    for s in np.unique(P["cross_section"]):
        r = (P["cross_section"] == s) & (P["ncomponents"] >= min_ncomp) & np.isfinite(P["log10_mass_750"])
        w = np.tile(gaussian_mass_weights(P["log10_mass_750"][r], mu, sig), pc.shape[0])
        x = np.clip(pc[:, r].ravel(), 1e-6, 1 - 1e-6)
        out[float(s)] = (np.log(x / (1 - x)), w / w.sum())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summaries", default=str(REPO / "sbi" / "summaries_wb1_final_n5"))
    p.add_argument("--mass-study", default="jauzac")
    p.add_argument("--min-ncomp", type=int, default=2)
    p.add_argument("--out", default=None)
    opts = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = Path(opts.summaries)
    mu, sig = a2744_mass_gaussian(opts.mass_study)
    po = np.load(S / "obs.npz")["softmax"][:, 0].mean()
    lo = np.log(po / (1 - po))
    smp = {d: samples(S, d, mu, sig, opts.min_ncomp) for d in SIMS}
    pooled = {}
    for s in sorted(set().union(*[smp[d] for d in SIMS])):
        parts = [smp[d][s] for d in SIMS if s in smp[d]]
        pooled[s] = (np.concatenate([a for a, _ in parts]), np.concatenate([w / len(parts) for _, w in parts]))
    smp["combined"] = pooled
    res = {}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)
    for ax, (d, dd) in zip(axes, smp.items()):
        sig_vals = sorted(dd)
        like = np.array([gaussian_kde(*[dd[s][0]], weights=dd[s][1])(lo)[0] for s in sig_vals])
        post = like / like.sum()
        res[d] = dict(zip([f"{s:g}" for s in sig_vals], post.round(4).tolist()))
        print(d, "  ".join(f"{s:g}: {q:.3f}" for s, q in zip(sig_vals, post)))
        ax.bar(range(len(sig_vals)), post, color="#D55E00", width=0.6)
        ax.set_xticks(range(len(sig_vals)), ["CDM" if s == 0 else f"{s:g}" for s in sig_vals])
        ax.set_xlabel(r"$\sigma/m$ [cm$^2$ g$^{-1}$]", fontsize=12)
        ax.text(0.03, 0.95, d.upper(), transform=ax.transAxes, ha="left", va="top", fontsize=12)
        ax.set_ylim(0, 1.2 * post.max())
    axes[0].set_ylabel("Posterior probability", fontsize=12)
    out = opts.out or str(REPO / "sbi" / "results" / f"empirical_posterior_{S.name}_{opts.mass_study}")
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=180)
    json.dump({"settings": vars(opts), "a2744_p_sidm": float(po), "posterior": res}, open(f"{out}.json", "w"), indent=1)
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
