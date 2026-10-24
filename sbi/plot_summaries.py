#!/usr/bin/env python
"""
Plot the SBI summaries written by make_summaries.py.

    (a) ensemble P(SIDM) against sigma/m for each simulation, with A2744, using
        the figure-2 selection (A2744 mass range, merging, mass-function weights)
    (b,c) the latent space of one model (PCA), CDM vs the largest cross-section
    (d,e) P(SIDM) against projected mass inside 750 kpc, one line per sigma/m
    (f) P(SIDM) against the number of merging components

Errors on ensemble means use return_error_in_mean (correlated models), as in
the paper figures; binned curves show the standard error over clusters.

Usage
-----
    cd ml_check && astro-env/bin/python sbi/plot_summaries.py
    astro-env/bin/python sbi/plot_summaries.py --held-out     # only clusters each model never trained on
    astro-env/bin/python sbi/plot_summaries.py --summaries sbi/summaries_ta --out sbi/plots/sbi_summaries_ta
    astro-env/bin/python sbi/plot_summaries.py --no-mass-cut  # panel (a) over all clusters, unweighted
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from add_shear_to_data import return_error_in_mean  # noqa: E402

DOMAINS = ["bahamas", "darkskies"]
# Same colours as COLOUR_SCHEME in notebooks/get_model_probabilities.py
COLOUR = {"bahamas": "#56B4E9", "darkskies": "#E69F00"}
RAMP = {"bahamas": plt.cm.Blues, "darkskies": plt.cm.Oranges}
INK, MUTED = "#222222", "#6b6b6b"
CDM_X = 3e-3          # where CDM sits on the log sigma/m axis (as in figure 2)


def a2744_log_mass(nsigma=2):
    '''
    log10 of the A2744 core + NW + N mass and its nsigma range, as
    get_model_probabilities.get_mass_cut(study='harvey') (not imported: that
    module reruns the catalogue cuts on import).
    '''
    mass = {"core": 16., "nw": 10.8, "n": 6.5}                       # 1e13 Msun
    err = {"core": (0.9, 0.6), "nw": (1.0, 0.3), "n": (0.9, 0.7)}
    est = sum(mass.values())
    lo, hi = (np.sqrt(sum((err[k][i] / mass[k])**2 for k in mass)) * est for i in (0, 1))
    return np.log10(np.array([est, est - nsigma * lo, est + nsigma * hi]) * 1e13)


A2744_LOG_MASS = a2744_log_mass()


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summaries", default=str(REPO / "sbi" / "summaries"))
    p.add_argument("--out", default=str(REPO / "sbi" / "plots" / "sbi_summaries"))
    p.add_argument("--held-out", action="store_true", help="use only rows each model did not train on")
    p.add_argument("--noise", type=int, default=0, help="noise realisation to plot")
    p.add_argument("--no-mass-cut", action="store_true",
                   help="panel (a): average all clusters instead of the figure-2 selection")
    return p.parse_args()


def setup_style():
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "grid"])
    except (ImportError, OSError):
        pass
    plt.rcParams.update({
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "grid.color": "#e3e3e3", "grid.linewidth": 0.6,
        "axes.titlesize": 11, "axes.labelsize": 11, "legend.frameon": False,
    })


def load(summaries, noise):
    S = Path(summaries)
    data = {}
    for d in DOMAINS:
        P = dict(np.load(S / f"params_{d}.npz"))
        P["softmax"] = np.load(S / f"softmax_{d}.npy")[:, noise]          # (n_models, N)
        latent_file = S / f"latents_{d}.npy"
        P["latents"] = np.load(latent_file, mmap_mode="r")[:, noise] if latent_file.exists() else None
        data[d] = P
    obs = dict(np.load(S / "obs.npz"))
    return data, obs


def masked(P, use_held_out):
    '''(n_models, N) array of P(SIDM) with rows a model trained on set to nan.'''
    s = P["softmax"].astype(float)
    return np.where(P["held_out"], s, np.nan) if use_held_out else s


def ensemble(values):
    '''Mean over models of a per-model statistic, with the correlated-model error.'''
    values = np.asarray(values)
    return values.mean(0), return_error_in_mean(values)


def figure2_weights(P, min_components=1):
    '''
    Per-cluster weights reproducing the selection of figure 2, i.e.
    get_threshold_for_cross(mass_cut=get_mass_cut(nsigma=2), integrated_mass=True,
    ncomponents=1, mass_weights=get_massfunction_weights(domain)):
    mass inside 750 kpc within the A2744 range, more than one merging component,
    weighted by the inverse CDM mass function of that same mass (30 bins from
    10^14 to 10^15.5).
    '''
    m750 = P["log10_mass_750"]
    sel = (m750 > A2744_LOG_MASS[1]) & (m750 < A2744_LOG_MASS[2]) & (P["ncomponents"] > min_components)
    mass_function, edges = np.histogram(m750[P["cross_section"] == 0], bins=np.linspace(14, 15.5, 30))
    mass_function[mass_function == 0] = 1
    ibin = np.clip(np.digitize(m750[sel], edges) - 1, 0, len(mass_function) - 1)
    weights = np.zeros(len(m750))
    weights[sel] = 1. / mass_function[ibin]
    return weights


def panel_response(ax, data, obs, use_held_out, mass_cut=True):
    for d in DOMAINS:
        P = data[d]
        s = masked(P, use_held_out)
        w = figure2_weights(P) if mass_cut else np.ones(s.shape[1])
        sigmas = np.unique(P["cross_section"])
        per_model = []
        for c in sigmas:
            wc = np.where(np.isfinite(s) & (P["cross_section"] == c), w, 0.)
            per_model.append(np.nansum(wc * s, axis=1) / wc.sum(1))
        mean, err = ensemble(np.array(per_model).T)
        if mass_cut:
            print(f"  (a) {d}: {np.sum(w > 0)} clusters pass the figure-2 selection")
        x = np.where(sigmas == 0, CDM_X, sigmas)
        ax.errorbar(x, mean, err, fmt="-o", color=COLOUR[d], lw=2, ms=8, mec="white",
                    mew=1.5, capsize=3, label=d.upper(), zorder=3)

    obs_mean, obs_err = ensemble(obs["softmax"][:, :1])
    ax.axhspan(obs_mean[0] - obs_err[0], obs_mean[0] + obs_err[0], color="k", alpha=0.12, lw=0)
    ax.axhline(obs_mean[0], color="k", lw=2)
    ax.text(CDM_X * 0.8, obs_mean[0] + obs_err[0], f"A2744  {obs_mean[0]:.3f} $\\pm$ {obs_err[0]:.3f}",
            va="bottom", ha="left", fontsize=9, color=INK)

    ax.set_xscale("log")
    ax.set_xlim(CDM_X * 0.6, 3)
    ax.set_xticks([CDM_X, 1e-2, 1e-1, 1])
    ax.set_xticklabels(["CDM", "0.01", "0.1", "1"])
    ax.set_xlabel(r"$\sigma/m$ [cm$^2$ g$^{-1}$]")
    ax.set_ylabel("Ensemble P(SIDM)")
    ax.set_title("A2744 mass range, merging, mass-function weighted" if mass_cut else "All clusters",
                 color=INK)
    ax.legend(loc="lower right")


def panel_latent(ax, d, P, obs):
    if P["latents"] is None:
        ax.text(0.5, 0.5, "latents not found\n(run make_summaries.py)", ha="center", va="center",
                transform=ax.transAxes, color=MUTED)
        ax.set_axis_off()
        return
    # The model whose A2744 output is closest to the ensemble median
    m = int(np.argmin(np.abs(obs["softmax"][:, 0] - np.median(obs["softmax"][:, 0]))))
    Z = np.asarray(P["latents"][m])
    pca = PCA(n_components=2).fit(Z)
    xy = pca.transform(Z)
    lo, hi = np.percentile(xy, [1, 99], axis=0)
    grid = np.mgrid[lo[0]:hi[0]:120j, lo[1]:hi[1]:120j]

    sigmas = np.unique(P["cross_section"])
    for c, shade, style in [(sigmas[0], 0.45, "--"), (sigmas[-1], 0.9, "-")]:
        pts = xy[P["cross_section"] == c]
        kde = gaussian_kde(pts.T)
        dens = kde(grid.reshape(2, -1)).reshape(grid.shape[1:])
        levels = np.quantile(kde(pts.T), [0.32, 0.68])     # 68% and 32% enclosed
        ax.contour(grid[0], grid[1], dens, levels=levels, colors=[RAMP[d](shade)],
                   linewidths=2, linestyles=style)

    obs_xy = pca.transform(obs["latents"][m, :1])
    ax.plot(*obs_xy[0], "*", ms=16, color="k", mec="white", mew=1.2, zorder=5)
    ax.annotate("A2744", obs_xy[0], xytext=(8, 4), textcoords="offset points", fontsize=9, color=INK)
    handles = [Line2D([], [], color=RAMP[d](0.45), lw=2, ls="--", label="CDM"),
               Line2D([], [], color=RAMP[d](0.9), lw=2, label=f"$\\sigma/m={sigmas[-1]:g}$")]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_title(f"{d.upper()} latent space (seed {int(P['seeds'][m])})", color=INK)
    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.0f}\\%)")
    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.0f}\\%)")
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def binned(x, y, edges):
    '''Mean and standard error of y in bins of x (bins with < 10 clusters dropped).'''
    idx = np.digitize(x, edges) - 1
    centres, mean, err = [], [], []
    for b in range(len(edges) - 1):
        yb = y[(idx == b) & np.isfinite(y)]
        if len(yb) >= 10:
            centres.append(0.5 * (edges[b] + edges[b+1]))
            mean.append(yb.mean())
            err.append(yb.std() / np.sqrt(len(yb)))
    return np.array(centres), np.array(mean), np.array(err)


def panel_mass(ax, d, P, use_held_out):
    s = np.nanmean(masked(P, use_held_out), axis=0)           # ensemble mean per cluster
    logm = P["log10_mass_750"]
    edges = np.linspace(*np.nanpercentile(logm, [2, 98]), 9)
    sigmas = np.unique(P["cross_section"])
    shades = np.linspace(0.35, 0.95, len(sigmas))
    for c, shade in zip(sigmas, shades):
        sel = P["cross_section"] == c
        x, y, e = binned(logm[sel], s[sel], edges)
        colour = RAMP[d](shade)
        ax.fill_between(x, y - e, y + e, color=colour, alpha=0.2, lw=0)
        ax.plot(x, y, "-", color=colour, lw=2, label="CDM" if c == 0 else f"{c:g}")
    ax.axvspan(A2744_LOG_MASS[1], A2744_LOG_MASS[2], color="k", alpha=0.08, lw=0)
    ax.text(A2744_LOG_MASS[0], 0.02, "A2744", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=9, color=INK)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title=r"$\sigma/m$", loc="upper left", fontsize=8,
              title_fontsize=8)
    ax.set_title(f"{d.upper()}: mass dependence", color=INK)
    ax.set_xlabel(r"$\log_{10} M(<750\,\mathrm{kpc})$ [M$_\odot$]")
    ax.set_ylabel("P(SIDM), cluster mean")


def panel_ncomp(ax, data, use_held_out, nmax=4):
    for d in DOMAINS:
        P = data[d]
        s = np.nanmean(masked(P, use_held_out), axis=0)
        ncomp = np.minimum(P["ncomponents"], nmax)
        sigmas = np.unique(P["cross_section"])
        for c, style, fill in [(sigmas[0], "--", "white"), (sigmas[-1], "-", COLOUR[d])]:
            sel = P["cross_section"] == c
            x = np.arange(1, nmax + 1)
            y = np.array([s[sel & (ncomp == n)].mean() for n in x])
            e = np.array([s[sel & (ncomp == n)].std() / np.sqrt(np.sum(sel & (ncomp == n))) for n in x])
            ax.errorbar(x, y, e, fmt="o" + style, color=COLOUR[d], mfc=fill, ms=8, lw=2, capsize=3,
                        label=f"{d.upper()} " + ("CDM" if c == 0 else f"$\\sigma/m={c:g}$"))
    ax.set_xticks(range(1, nmax + 1))
    ax.set_xticklabels([str(n) for n in range(1, nmax)] + [f"{nmax}+"])
    ax.set_xlabel("Merging components")
    ax.set_ylabel("P(SIDM), cluster mean")
    ax.set_title("Dependence on merger state", color=INK)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)


def main():
    opts = parse()
    setup_style()
    data, obs = load(opts.summaries, opts.noise)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    panel_response(axes[0, 0], data, obs, opts.held_out, mass_cut=not opts.no_mass_cut)
    for ax, d in zip(axes[0, 1:], DOMAINS):
        panel_latent(ax, d, data[d], obs)
    for ax, d in zip(axes[1, :2], DOMAINS):
        panel_mass(ax, d, data[d], opts.held_out)
    panel_ncomp(axes[1, 2], data, opts.held_out)

    nmodels = obs["softmax"].shape[0]
    model = json.load(open(Path(opts.summaries) / "info.json"))["options"]["model"]
    rows = "held-out clusters" if opts.held_out else "all clusters"
    fig.suptitle(f"CNN summaries for SBI: {nmodels} {model.replace('_', ' ')} models, {rows}",
                 color=INK, fontsize=13)
    for ax, letter in zip(axes.flat, "abcdef"):
        ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11, color=INK)

    out = Path(opts.out + ("_heldout" if opts.held_out else ""))
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
