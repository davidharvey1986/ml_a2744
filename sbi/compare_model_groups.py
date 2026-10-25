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

Usage
-----
    cd ml_check && astro-env/bin/python sbi/compare_model_groups.py
"""

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
    "ba best": "summaries_ba_best",
    "ba final": "summaries_ba_final",
}
DOMAINS = ["bahamas", "darkskies"]
COLOUR = {"bahamas": "#56B4E9", "darkskies": "#E69F00"}
INK, MUTED = "#222222", "#6b6b6b"
EPS = 1e-4


def weights(P, mu, sig):
    '''Merging clusters weighted to the A2744 mass, separately per cross-section.'''
    w = np.zeros(len(P["cross_section"]))
    m750, cs = P["log10_mass_750"], P["cross_section"]
    for c in np.unique(cs):
        sel = (cs == c) & (P["ncomponents"] > 1) & np.isfinite(m750)
        w[sel] = gaussian_mass_weights(m750[sel], mu, sig)
    return w


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def group_metrics(S, mu, sig):
    out, cdm_per_model = {}, {}
    x_obs = logit(np.load(S / "obs.npz")["softmax"][:, 0].mean())
    for d in DOMAINS:
        P = dict(np.load(S / f"params_{d}.npz"))
        soft = np.load(S / f"softmax_{d}.npy")[:, 0]                     # (n_models, N)
        w = weights(P, mu, sig)
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


def plot(M, out):
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
    fig.suptitle("Model groups: sensitivity and domain alignment (merging, weighted to A2744 mass)",
                 color=INK, fontsize=12)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"model_groups.{ext}", dpi=180)


def main():
    mu, sig = a2744_mass_gaussian()
    M = {}
    for g, name in GROUPS.items():
        S = REPO / "sbi" / name
        if not (S / "softmax_bahamas.npy").exists():
            print(f"skipping {g}: no summaries in {S}")
            continue
        M[g] = group_metrics(S, mu, sig)

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
    json.dump(M, open(out / "model_groups.json", "w"), indent=2)
    plot(M, out)
    print(f"\nwrote {out / 'model_groups.json'} and model_groups.pdf/.png")


if __name__ == "__main__":
    main()
