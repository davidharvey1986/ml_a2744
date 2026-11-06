#!/usr/bin/env python
'''
The per-cluster CDM distribution of the ensemble P(SIDM), pooled over BAHAMAS, DARKSKIES, FLAMINGO and TNG, and the
fraction of it above A2744. Selection as Figure 2 (plots_for_paper.figure2_summaries): CDM clusters with
n_comp >= --min-ncomp, weighted to the A2744 mass with gaussian_mass_weights per simulation; every shape-noise draw
is a sample. Each simulation is normalised to the same total weight, so the pooled distribution is the mean of the
four. Faded: each simulation's weighted KDE; thick: the pooled KDE, its part above A2744 shaded.

    astro-env/bin/python sbi/cdm_pooled_tail.py --tag final
'''
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde, norm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from mass_weighting import a2744_mass_gaussian, effective_sample_size, gaussian_mass_weights  # noqa: E402

COLOURS = {"bahamas": "#56B4E9", "darkskies": "#E69F00", "flamingo": "#D55E00", "tng": "#009E73"}
NAMES = {"bahamas": "BAHAMAS", "darkskies": "DARKSKIES", "flamingo": "FLAMINGO", "tng": "TNG-Cluster (z = 0)"}


def load(tag, sim, mu, sig, min_ncomp):
    d = REPO / "sbi" / (f"summaries_wb1_{tag}_n5" + ("_flamingo" if sim in ("flamingo", "tng") else ""))
    P = np.load(d / f"params_{sim}.npz")
    p = np.load(d / f"softmax_{sim}.npy").mean(0)                    # (n_noise, N) ensemble P(SIDM)
    rows = (P["cross_section"] == 0) & (P["ncomponents"] >= min_ncomp) & np.isfinite(P["log10_mass_750"])
    w = gaussian_mass_weights(P["log10_mass_750"][rows], mu, sig)
    p = p[:, rows]
    return p.ravel(), np.tile(w, p.shape[0]) / (w.sum() * p.shape[0]), int(rows.sum()), effective_sample_size(w)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="final")
    ap.add_argument("--min-ncomp", type=int, default=2)
    ap.add_argument("--out", default=None)
    opts = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_obs = float(np.load(REPO / "sbi" / f"summaries_wb1_{opts.tag}_n5" / "obs.npz")["softmax"][:, 0].mean())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True, constrained_layout=True)
    x = np.linspace(0.1, 0.95, 800)
    for ax, study in zip(axes, ("harvey", "jauzac")):
        mu, sig = a2744_mass_gaussian(study)
        print(f"== ba_{opts.tag}, {study} mass {mu:.2f} ± {sig:.2f}, n_comp >= {opts.min_ncomp}, A2744 {p_obs:.3f}")
        pooled, P, W = np.zeros_like(x), [], []
        for sim in COLOURS:
            p, w, n, n_eff = load(opts.tag, sim, mu, sig, opts.min_ncomp)
            k = gaussian_kde(p, weights=w)(x)
            pooled += k / len(COLOURS)
            P.append(p), W.append(w / len(COLOURS))
            f = w[p >= p_obs].sum()
            print(f"  {NAMES[sim]:20s} N {n:5d} (N_eff {n_eff:6.0f})  weighted mean {np.sum(w * p):.3f}  "
                  f"above A2744 {100 * f:.2f}% ({norm.isf(max(f, 1e-6)):.2f} sigma)  raw clusters above {np.sum(p >= p_obs) / (len(p) / n):.1f}")
            ax.fill_between(x, k, color=COLOURS[sim], alpha=0.15, lw=0)
            ax.plot(x, k, color=COLOURS[sim], lw=1, alpha=0.6, label=NAMES[sim])
        P, W = np.concatenate(P), np.concatenate(W)
        f = W[P >= p_obs].sum()
        n_above = sum(np.sum(p >= p_obs) for p in [P]) / 5                 # per noise draw (5 draws)
        f_kde = np.trapz(pooled[x >= p_obs], x[x >= p_obs])
        print(f"  pooled: weighted fraction above A2744 {100 * f:.2f}% ({norm.isf(f):.2f} sigma); KDE tail {100 * f_kde:.2f}%; "
              f"raw cluster-draws above / noise draw {n_above:.1f}")
        ax.plot(x, pooled, color="k", lw=2.5, label="All CDM (pooled)")
        m = x >= p_obs
        ax.fill_between(x[m], pooled[m], color="k", alpha=0.35, lw=0,
                        label=f"above A2744: {100 * f:.1f}% ({norm.isf(f):.2f}σ)")
        ax.axvline(p_obs, color="k", ls="--", lw=1.2)
        ax.annotate(f"A2744 {p_obs:.3f}", (p_obs, 0.97), xycoords=("data", "axes fraction"), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="top", fontsize=9)
        ax.set_title(f"{study.capitalize()} mass: log M750 = {mu:.2f} ± {sig:.2f}", fontsize=11)
        ax.set_xlabel("ensemble P(SIDM) of each CDM cluster")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xlim(0.15, 0.9)
        ax.set_ylim(0, None)
    axes[0].set_ylabel("density (mass-weighted to A2744)")
    fig.suptitle(f"wb1 ba_{opts.tag}: CDM clusters with n_comp ≥ {opts.min_ncomp}, pooled over four simulations "
                 "(each given equal weight)", fontsize=11)
    out = Path(opts.out or REPO / "sbi" / "results" / "wb1" / f"cdm_pooled_tail_{opts.tag}_nc{opts.min_ncomp}")
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=170)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
