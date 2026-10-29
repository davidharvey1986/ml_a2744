#!/usr/bin/env python
'''
Response of one model group with the simulated clusters matched to A2744 on the tangential
shear profile of its main halo instead of its mass.

Profiles as profile_matching.py: noise-free chi_t of each cluster (stored reduced shear at the
A2744 galaxies, source-redshift rescaled) about the stamp centre, against A2744's with its
shape-noise covariance from random galaxy rotations. No simulated profile fits A2744 within
its shape noise, so instead of the exp(-chi^2/2) likelihood (a handful of clusters) the
--n-nearest clusters with the lowest chi^2 are taken per sigma/m (n_comp >= --min-ncomp),
with equal weights.

Outputs (<summaries>/../results/<summaries name>/): profile_response.png/pdf/json (P(SIDM)
against sigma/m, all and held-out, with the mass-matched curves dotted) and
matched_profiles.png/pdf (A2744's profile and the mean of the matched clusters per sigma/m).

    astro-env/bin/python sbi/profile_response.py --summaries $PWD/sbi/summaries_spin2_best_n5 --label "spin2 best"
'''
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "sbi")]
import profile_matching as pm  # noqa: E402  (chdirs to notebooks/)
from mass_weighting import a2744_mass_gaussian, gaussian_mass_weights  # noqa: E402
from response_plot import COLOUR, X_CDM  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summaries", required=True)
    p.add_argument("--n-nearest", type=int, default=100, help="clusters per sigma/m with the closest profile")
    p.add_argument("--min-ncomp", type=int, default=2)
    p.add_argument("--label", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def summary(v, held, S, w=None):
    w = np.ones(len(v)) if w is None else w
    m = np.sum(w * v) / w.sum()
    n_eff = w.sum() ** 2 / np.sum(w ** 2)
    per = [np.average(S[i][held[i]], weights=w[held[i]]) for i in range(S.shape[0]) if held[i].sum() >= 5]
    return m, np.sqrt(np.sum(w * (v - m) ** 2) / w.sum() / n_eff), np.mean(per), np.std(per) / np.sqrt(len(per))


def main():
    opts = parse()
    S_dir = Path(opts.summaries).resolve()
    out = S_dir.parent / "results" / S_dir.name
    out.mkdir(parents=True, exist_ok=True)
    obs = np.load(S_dir / "obs.npz")["softmax"]
    p_a = float(obs[:, 0].mean()) if obs.ndim == 2 else float(obs[:, 0, 0].mean())
    mu, sig = a2744_mass_gaussian()

    d, cov, occupied = pm.a2744_profile(pm.DEFAULT_OBS, np.random.default_rng(opts.seed))
    icov, err = np.linalg.inv(cov), np.sqrt(np.diag(cov))
    rescale = pm.source_redshift_rescale()
    rk = 10 * (pm.FIT_BINS[1:] + pm.FIT_BINS[:-1])

    fig_r, ax_r = plt.subplots(figsize=(8, 5), constrained_layout=True)
    fig_p, axes_p = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True, sharey=True)
    ax_r.axhline(p_a, color="k", lw=2, label=f"A2744: {p_a:.3f}")
    res = {"a2744_p_sidm": p_a, "n_nearest": opts.n_nearest, "min_ncomp": opts.min_ncomp, "radii_kpc": rk.tolist(),
           "a2744_profile": d.tolist(), "a2744_profile_err": err.tolist()}
    for j, dom in enumerate(("bahamas", "darkskies")):
        P = dict(np.load(S_dir / f"params_{dom}.npz"))
        S = np.load(S_dir / f"softmax_{dom}.npy")[:, 0]              # (models, clusters), noise draw 0
        sm, ho = S.mean(0), P["held_out"]
        prof = pm.simulated_profiles(P, pm.DEFAULT_MAPS, occupied, rescale)
        resid = prof - d
        chi2 = np.einsum("ij,jk,ik->i", resid, icov, resid)
        chi2 = np.where(np.isfinite(chi2), chi2, np.inf)
        cs, m = P["cross_section"], P["log10_mass_750"]
        sel = (P["ncomponents"] >= opts.min_ncomp) & np.isfinite(m)
        sigmas = np.unique(cs)
        cols = (plt.cm.Blues if dom == "bahamas" else plt.cm.Oranges)(np.linspace(0.4, 0.95, len(sigmas)))
        c = {k: [] for k in ("sigma", "profile", "profile_err", "held", "held_err", "q16", "q84", "frac_ge_a2744",
                             "chi2_median", "log10_mass_750_mean", "mass", "mass_err", "mean_profile")}
        ax = axes_p[j]
        for s, col in zip(sigmas, cols):
            rows = np.where(sel & (cs == s))[0]
            near = rows[np.argsort(chi2[rows])[:opts.n_nearest]]
            v = sm[near]
            a, ae, h, he = summary(v, ho[:, near], S[:, near])
            wm = gaussian_mass_weights(m[rows], mu, sig)
            ma, mae, _, _ = summary(sm[rows], ho[:, rows], S[:, rows], wm)
            mp = prof[near].mean(0)
            for key, val in [("sigma", float(s)), ("profile", a), ("profile_err", ae), ("held", h), ("held_err", he),
                             ("q16", np.percentile(v, 16)), ("q84", np.percentile(v, 84)),
                             ("frac_ge_a2744", np.mean(v >= p_a)), ("chi2_median", np.median(chi2[near])),
                             ("log10_mass_750_mean", m[near].mean()), ("mass", ma), ("mass_err", mae)]:
                c[key].append(float(val))
            c["mean_profile"].append(mp.tolist())
            lab = "CDM" if s == 0 else f"σ/m={s:g}"
            ax.fill_between(rk, *np.percentile(prof[near], [16, 84], axis=0), color=col, alpha=0.12, lw=0)
            ax.plot(rk, mp, "-", color=col, lw=2, label=f"{lab}: median χ² {np.median(chi2[near]):.0f}")
            ax.plot(rk, np.average(prof[rows], axis=0, weights=wm), ":", color=col, lw=1.2)
        ax.errorbar(rk, d, err, fmt="o", color="k", ms=6, capsize=2, label="A2744 (shape-noise errors)")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("radius from the stamp centre [kpc]")
        ax.set_title(f"{dom.upper()}: {opts.n_nearest} closest per σ/m (solid, 16-84%), mass-matched (dotted)",
                     fontsize=9)
        ax.legend(fontsize=8)
        res[dom] = c
        x = np.where(np.array(c["sigma"]) == 0, X_CDM, c["sigma"])
        col = COLOUR[dom]
        ax_r.fill_between(x, c["q16"], c["q84"], color=col, alpha=0.12, lw=0, label=f"{dom.upper()}: 16-84% of matched")
        ax_r.errorbar(x, c["profile"], c["profile_err"], fmt="o-", color=col, ms=6, capsize=2, lw=2,
                      label=f"{dom.upper()}: profile-matched")
        ax_r.errorbar(x * 1.06, c["held"], c["held_err"], fmt="s--", color=col, ms=5, capsize=2, mfc="white", lw=1.2,
                      label=f"{dom.upper()}: profile-matched, held-out")
        ax_r.errorbar(x / 1.06, c["mass"], c["mass_err"], fmt="^:", color=col, ms=5, capsize=2, lw=1, alpha=0.7,
                      label=f"{dom.upper()}: mass-matched")
        print(f"{dom}: " + " | ".join(
            f"σ={s:g}: {a:.3f}±{e:.3f} (held-out {h:.3f}, mass-matched {mm:.3f}), ≥A2744 {f:.2f}, χ² {x2:.0f}, "
            f"logM {lm:.2f}" for s, a, e, h, mm, f, x2, lm in zip(
                c["sigma"], c["profile"], c["profile_err"], c["held"], c["mass"], c["frac_ge_a2744"],
                c["chi2_median"], c["log10_mass_750_mean"])))
    axes_p[0].set_ylabel(r"tangential $\chi_t$")
    ax_r.set_xscale("log")
    ax_r.set_xlim(0.003, 1.4)
    ax_r.set_xticks([X_CDM, 0.01, 0.05, 0.1, 0.2, 0.3, 1], ["CDM", "0.01", "0.05", "0.1", "0.2", "0.3", "1"])
    ax_r.minorticks_off()
    ax_r.set_xlabel("σ/m [cm²/g]")
    ax_r.set_ylabel("ensemble P(SIDM)")
    ax_r.set_title(f"{opts.label or S_dir.name}: {opts.n_nearest} closest main-halo shear profiles per σ/m, "
                   f"n_comp ≥ {opts.min_ncomp}", fontsize=10)
    ax_r.legend(fontsize=7, loc="upper left", ncol=2)
    for ext in ("png", "pdf"):
        fig_r.savefig(out / f"profile_response.{ext}", dpi=150)
        fig_p.savefig(out / f"matched_profiles.{ext}", dpi=150)
    json.dump(res, open(out / "profile_response.json", "w"), indent=1)
    print(f"wrote {out}/profile_response.png, matched_profiles.png")


if __name__ == "__main__":
    main()
