#!/usr/bin/env python
"""
Compare the model groups in models/concat on the summaries from make_summaries.py.

For each group and simulation, with merging clusters weighted to the A2744 mass
(mass_weighting.py, as panel (a) of plot_summaries.py):

    ensemble S/N(sigma)   mean over models of P_m(sigma) - P_m(CDM), divided by its
                          error (return_error_in_mean); differencing within a model
                          removes the offsets shared between cross-sections
    single-cluster d'     (mean - CDM mean) / CDM cluster scatter of the
                          ensemble-mean logit P(SIDM): what one cluster can tell apart
    CDM alignment         P_m(CDM, bahamas) - P_m(CDM, darkskies), mean over models
                          and its error
    A2744 vs CDM          weighted fraction of CDM clusters whose ensemble output is
                          at least A2744's, and its Gaussian-equivalent significance

reported at sigma/m = 0.1 (simulated in both) and at each simulation's largest
sigma/m, plus the A2744 ensemble output.

With --match-ncomp the merger state is matched too: merging clusters are split into
n_comp = 2, 3, 4+, mass-weighted within each class, and the classes reweighted to a
common mix (the mass-weighted CDM mix pooled equally over both simulations), so every
simulation and cross-section has the same mass and merger-state distribution.
Outputs then carry the suffix _ncomp.

Usage
-----
    cd ml_check && astro-env/bin/python sbi/compare_model_groups.py
    astro-env/bin/python sbi/compare_model_groups.py --match-ncomp
"""

import argparse

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from add_shear_to_data import return_error_in_mean  # noqa: E402
from mass_weighting import a2744_mass_gaussian, gaussian_mass_weights  # noqa: E402

# group label -> summaries directory
GROUPS = {
    "best": "summaries",
    "final": "summaries_final",
    "ta best": "summaries_ta",
    "ta final": "summaries_ta_final",
    "ba best": "summaries_ba_best_n5",      # 5 noise draws; draw 0 is used here
    "ba final": "summaries_ba_final_n5",
}
DOMAINS = ["bahamas", "darkskies"]
COLOUR = {"bahamas": "#56B4E9", "darkskies": "#E69F00"}
INK, MUTED = "#222222", "#6b6b6b"
EPS = 1e-4
NCOMP_CLASSES = (2, 3, 4)     # merging clusters; 4 means 4 or more
MIN_PER_CLASS = 10


def ncomp_class(P):
    return np.minimum(P["ncomponents"], NCOMP_CLASSES[-1])


def weights(P, mu, sig, target=None):
    '''
    Merging clusters weighted to the A2744 mass, separately per cross-section. With
    target {n_comp class: fraction}, the mass weighting is done within each class and
    the classes are scaled to the target mix.
    '''
    w = np.zeros(len(P["cross_section"]))
    m750, cs = P["log10_mass_750"], P["cross_section"]
    merging = (P["ncomponents"] > 1) & np.isfinite(m750)
    for c in np.unique(cs):
        if target is None:
            sel = (cs == c) & merging
            w[sel] = gaussian_mass_weights(m750[sel], mu, sig)
            continue
        for k, frac in target.items():
            sel = (cs == c) & merging & (ncomp_class(P) == k)
            if sel.sum() < MIN_PER_CLASS:
                print(f"  warning: only {sel.sum()} clusters with n_comp class {k} at sigma/m={c:g}")
                continue
            wk = gaussian_mass_weights(m750[sel], mu, sig)
            w[sel] = frac * wk / wk.sum()
    return w


def ncomp_mix(P, w, sigma=0.):
    '''Weighted n_comp class fractions of one cross-section.'''
    sel = P["cross_section"] == sigma
    cls = ncomp_class(P)[sel]
    return {k: float(w[sel][cls == k].sum() / w[sel].sum()) for k in NCOMP_CLASSES}


def target_ncomp_mix(S, mu, sig):
    '''Mass-weighted CDM n_comp mix, averaged over the two simulations.'''
    mixes = []
    for d in DOMAINS:
        P = dict(np.load(S / f"params_{d}.npz"))
        mixes.append(ncomp_mix(P, weights(P, mu, sig)))
        print(f"  {d:9s} mass-weighted merging CDM n_comp mix: "
              + ", ".join(f"{k}{'+' if k == NCOMP_CLASSES[-1] else ''}: {v:.2f}" for k, v in mixes[-1].items()))
    return {k: 0.5 * (mixes[0][k] + mixes[1][k]) for k in NCOMP_CLASSES}


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def group_metrics(S, mu, sig, target=None):
    out, cdm_per_model = {}, {}
    x_obs = logit(np.load(S / "obs.npz")["softmax"][:, 0].mean())
    for d in DOMAINS:
        P = dict(np.load(S / f"params_{d}.npz"))
        soft = np.load(S / f"softmax_{d}.npy")[:, 0]                     # (n_models, N)
        w = weights(P, mu, sig, target)
        cs = P["cross_section"]
        sigmas = np.unique(cs)
        per_model = {c: (soft[:, cs == c] * w[cs == c]).sum(1) / w[cs == c].sum() for c in sigmas}
        x = logit(soft.mean(0))
        xbar = {c: np.average(x[cs == c], weights=w[cs == c]) for c in sigmas}
        sd_cdm = np.sqrt(np.average((x[cs == 0] - xbar[0])**2, weights=w[cs == 0]))
        res = {"n_models": int(soft.shape[0]),
               "p_mean": {f"{c:g}": float(per_model[c].mean()) for c in sigmas},
               "p_err": {f"{c:g}": float(return_error_in_mean(per_model[c][:, None])[0]) for c in sigmas}}
        for label, c in [("0.1", 0.1), ("max", sigmas.max())]:
            diff = per_model[c] - per_model[0]
            err = return_error_in_mean(diff[:, None])[0]
            res[f"signal_{label}"] = float(diff.mean())
            res[f"ensemble_snr_{label}"] = float(diff.mean() / err)
            res[f"dprime_{label}"] = float((xbar[c] - xbar[0]) / sd_cdm)
        res["sigma_max"] = float(sigmas.max())
        p = float(np.sum(w[cs == 0] * (x[cs == 0] >= x_obs)) / w[cs == 0].sum())
        res["a2744_p_cdm"] = p
        res["a2744_z_cdm"] = float(norm.isf(p))
        out[d] = res
        cdm_per_model[d] = per_model[0]
    offset = cdm_per_model["bahamas"] - cdm_per_model["darkskies"]
    err = return_error_in_mean(offset[:, None])[0]
    obs = np.load(S / "obs.npz")["softmax"][:, 0]
    out["cdm_offset"] = {"bahamas_minus_darkskies": float(offset.mean()), "err": float(err),
                         "in_sigma": float(offset.mean() / err)}
    out["a2744"] = {"p_mean": float(obs.mean()), "err": float(return_error_in_mean(obs[:, None])[0])}
    return out


def plot(M, out, suffix="", label="merging, weighted to A2744 mass"):
    plt.rcParams.update({"axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.labelcolor": INK, "axes.grid": True, "grid.color": "#e6e6e6",
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "font.size": 10})
    groups = list(M)
    y = np.arange(len(groups))[::-1]
    labels = [f"{g} ({M[g]['bahamas']['n_models']})" for g in groups]
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.8), sharey=True, constrained_layout=True)
    panels = [("ensemble_snr_0.1", r"Ensemble S/N, $\sigma/m=0.1$"),
              ("ensemble_snr_max", r"Ensemble S/N, largest $\sigma/m$"),
              ("dprime_max", r"Single-cluster $d'$, largest $\sigma/m$"),
              ("a2744_z_cdm", "A2744 vs CDM clusters")]
    for ax, (key, title) in zip(axes, panels):
        for k, d in enumerate(DOMAINS):
            ax.plot([M[g][d][key] for g in groups], y + (0.12 if k == 0 else -0.12), "o", ms=9,
                    color=COLOUR[d], mec="white", mew=1.2, label=d.upper())
        ax.set_title(title, color=INK)
        ax.set_xlim(left=0)
    axes[1].legend(loc="upper right")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[2].text(0.58, 0.98, "BAHAMAS: CDM vs 1\nDARKSKIES: CDM vs 0.2", transform=axes[2].transAxes,
                 ha="center", va="top", fontsize=8, color=MUTED)

    axes[3].set_xlabel("significance [$\\sigma$]")
    ax = axes[4]
    off = [M[g]["cdm_offset"]["bahamas_minus_darkskies"] for g in groups]
    err = [M[g]["cdm_offset"]["err"] for g in groups]
    ax.axvline(0, color=MUTED, lw=1)
    ax.errorbar(off, y, xerr=err, fmt="o", ms=9, color=INK, mec="white", mew=1.2, capsize=3)
    for yi, o, e in zip(y, off, err):
        ax.annotate(f"{o / e:+.1f}$\\sigma$", (o, yi), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    ax.set_title("CDM alignment: BAHAMAS - DARKSKIES", color=INK)
    ax.set_xlabel("P(SIDM) offset at CDM")
    for ax in axes[:3]:
        ax.set_xlabel("S/N" if "S/N" in ax.get_title() else "separation / CDM scatter")
    fig.suptitle(f"Model groups: sensitivity and domain alignment ({label})",
                 color=INK, fontsize=12)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"model_groups{suffix}.{ext}", dpi=180)


def plot_response(M, out, suffix="", label="merging, weighted to A2744 mass", cdm_x=3e-3):
    '''Mass-matched ensemble P(SIDM) against sigma/m, one panel per model group.'''
    groups = list(M)
    ncol = min(3, len(groups))
    nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow), sharex=True, sharey=True,
                             constrained_layout=True, squeeze=False)
    for ax, g in zip(axes.flat, groups):
        a = M[g]["a2744"]
        ax.axhspan(a["p_mean"] - a["err"], a["p_mean"] + a["err"], color="k", alpha=0.1, lw=0)
        ax.axhline(a["p_mean"], color="k", lw=1.5)
        for d in DOMAINS:
            sig = np.array([float(k) for k in M[g][d]["p_mean"]])
            mean = np.array(list(M[g][d]["p_mean"].values()))
            err = np.array(list(M[g][d]["p_err"].values()))
            x = np.where(sig == 0, cdm_x, sig)
            ax.errorbar(x, mean, err, fmt="-o", color=COLOUR[d], lw=2, ms=7, mec="white", mew=1.2,
                        capsize=3, label=d.upper())
        ax.set_xscale("log")
        ax.set_xticks([cdm_x, 1e-2, 1e-1, 1])
        ax.set_xticklabels(["CDM", "0.01", "0.1", "1"])
        ax.set_xlim(cdm_x * 0.6, 2)
        ax.set_title(f"{g} ({M[g]['bahamas']['n_models']} models)", color=INK)
    for ax in axes.flat[len(groups):]:
        ax.set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel(r"$\sigma/m$ [cm$^2$ g$^{-1}$]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Ensemble P(SIDM)")
    first = axes.flat[0]
    first.legend(loc="lower right")
    a = M[groups[0]]["a2744"]
    first.text(cdm_x * 0.75, a["p_mean"] + a["err"], "A2744", va="bottom", fontsize=9, color=INK)
    fig.suptitle(f"Softmax output vs cross-section ({label}; error in ensemble mean)",
                 color=INK, fontsize=12)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"model_groups_response{suffix}.{ext}", dpi=180)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match-ncomp", action="store_true", help="also match the n_comp mix between simulations")
    ap.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS),
                    help="model groups to include, e.g. --groups 'ba best' 'ba final'")
    opts = ap.parse_args()
    mu, sig = a2744_mass_gaussian()
    target, suffix, label = None, "", "merging, weighted to A2744 mass"
    if opts.match_ncomp:
        # cluster masses and n_comp are the same for every group, so any summaries will do
        target = target_ncomp_mix(REPO / "sbi" / GROUPS["best"], mu, sig)
        print("  target mix: " + ", ".join(f"{k}: {v:.2f}" for k, v in target.items()))
        suffix, label = "_ncomp", "merging, weighted to A2744 mass and a common n_comp mix"
    M = {}
    for g, name in GROUPS.items():
        if g not in opts.groups:
            continue
        S = REPO / "sbi" / name
        if not (S / "softmax_bahamas.npy").exists():
            print(f"skipping {g}: no summaries in {S}")
            continue
        M[g] = group_metrics(S, mu, sig, target)

    header = (f"{'group':10s} {'n':>3s} | {'S/N 0.1 B':>9s} {'S/N 0.1 D':>9s} | {'S/N max B':>9s} {'S/N max D':>9s}"
              f" | {'dp max B':>8s} {'dp max D':>8s} | {'CDM offset':>17s} | {'A2744':>13s} | {'z_CDM B':>7s} {'z_CDM D':>7s}")
    print(header)
    print("-" * len(header))
    for g, m in M.items():
        b, d, o, a = m["bahamas"], m["darkskies"], m["cdm_offset"], m["a2744"]
        print(f"{g:10s} {b['n_models']:3d} | {b['ensemble_snr_0.1']:9.2f} {d['ensemble_snr_0.1']:9.2f} | "
              f"{b['ensemble_snr_max']:9.2f} {d['ensemble_snr_max']:9.2f} | {b['dprime_max']:8.2f} "
              f"{d['dprime_max']:8.2f} | {o['bahamas_minus_darkskies']:+.3f}+/-{o['err']:.3f} "
              f"({o['in_sigma']:+.1f}) | {a['p_mean']:.3f}+/-{a['err']:.3f} | {b['a2744_z_cdm']:7.2f} {d['a2744_z_cdm']:7.2f}")

    out = REPO / "sbi" / "results"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(M, open(out / f"model_groups{suffix}.json", "w"), indent=2)
    plot(M, out, suffix, label)
    plot_response(M, out, suffix, label)
    print(f"\nwrote {out}/model_groups{suffix}.json, model_groups{suffix}.pdf/.png "
          f"and model_groups_response{suffix}.pdf/.png")


if __name__ == "__main__":
    main()
