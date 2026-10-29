#!/usr/bin/env python
'''
Is the CNN reading how flat the shear is a few hundred kpc out? One number per map:

    R = <chi_t>(200-400 kpc) / <chi_t>(50-200 kpc)

the mean tangential distortion about the stamp centre over occupied pixels in each annulus.
For the simulations it is measured on the noise-free maps sampled at the A2744 galaxies
(data/100/obs/concat, reduced shear converted to chi = 2g / (1 + |g|^2) per pixel) and set against
each cluster's ensemble P(SIDM) from the summaries (mean over models and noise draws). For A2744
it is measured on the data map, with a galaxy bootstrap for its error, at the adopted centre
and with the stamp recentred on each Lenstool potential (P(SIDM) recomputed with the same models).

Within each cross-section, a Spearman correlation between R and P(SIDM) says whether R drives the
output beyond sigma/m; where A2744's R falls among the simulations says whether its P(SIDM) is an
interpolation or an extrapolation.

    astro-env/bin/python sbi/shear_ratio.py --summaries sbi/summaries_spin2_best_n5 --include-str spin2_ft_
'''
import argparse
import json
import pickle as pkl
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from lenspack.utils import bin2d
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import make_summaries as ms  # noqa: E402  (chdirs to notebooks/)
import sensitivity as sens  # noqa: E402
from add_shear_to_data import galaxy_distortion  # noqa: E402
from mass_weighting import a2744_mass_gaussian  # noqa: E402

NPIX = 100
_c = np.arange(NPIX) - (NPIX / 2 - 0.5)
XG, YG = np.meshgrid(_c, _c)                       # maps are indexed [dec, ra]
RG, PHI = np.hypot(XG, YG), np.arctan2(YG, XG)
INNER, OUTER = (2.5, 10), (10, 20)                 # pixels (20 kpc): 50-200 and 200-400 kpc
COLOURS = {0: "#000000", 0.1: "#0072B2", 0.3: "#009E73", 1: "#D55E00"}


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summaries", default=str(REPO / "sbi" / "summaries_spin2_best_n5"))
    p.add_argument("--model", default="tweakedalign_k0p9_ctr_ba_best")
    p.add_argument("--include-str", default="spin2_ft_")
    p.add_argument("--nmodels", type=int, default=None)
    p.add_argument("--min-ncomp", type=int, default=2)
    p.add_argument("--mass-window", type=float, default=0.15, help="|log M750 - A2744| for the matched subset [dex]")
    p.add_argument("--nboot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(REPO / "sbi" / "results" / "shear_ratio"))
    return p.parse_args()


def ratio(maps):
    '''R for maps (..., 2, npix, npix) of chi; empty pixels (both components zero) are left out.'''
    c2, s2 = np.cos(2 * PHI), np.sin(2 * PHI)
    t = -(maps[..., 0, :, :] * c2 + maps[..., 1, :, :] * s2)
    occ = np.any(maps != 0, axis=-3)

    def mean(lo, hi):
        m = occ & (RG >= lo) & (RG < hi)
        return np.sum(t * m, axis=(-2, -1)) / np.maximum(m.sum(axis=(-2, -1)), 1)
    return mean(*OUTER) / mean(*INNER)


def to_chi(g):
    g2 = np.sum(g ** 2, axis=-3, keepdims=True)
    return 2 * g / (1 + g2)


def a2744(models, device, cat, ra, dec, c1, c2, rng, nboot):
    def raw_of(idx):
        return np.array(bin2d(ra[idx], dec[idx], v=(c1[idx], c2[idx]), npix=NPIX, extent=[-50, 50, -50, 50]),
                        dtype=np.float32)
    raw = raw_of(np.arange(len(ra)))
    boot = np.array([ratio(raw_of(rng.integers(0, len(ra), len(ra)))) for _ in range(nboot)])
    p = float(sens.model_outputs(models, sens.normalise(torch.tensor(raw))[0][None], device, 1).mean())
    return {"R": float(ratio(raw)), "R_boot_16_50_84": np.percentile(boot, [16, 50, 84]).tolist(), "p_sidm": p}


def main():
    opts = parse()
    out = Path(opts.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(opts.seed)
    S = Path(opts.summaries).resolve()
    mu, _ = a2744_mass_gaussian()

    sims = {}
    for dom in ("bahamas", "darkskies"):
        P = dict(np.load(S / f"params_{dom}.npz"))
        p_sidm = np.load(S / f"softmax_{dom}.npy").mean((0, 1))
        R = np.full(len(p_sidm), np.nan)
        for f in np.unique(P["filename"]):
            rows = np.where(P["filename"] == f)[0]
            _, obs = pkl.load(open(f"../data/100/obs/concat/{f}", "rb"))
            R[rows] = ratio(to_chi(obs[P["image_idx"][rows], :2]))
        keep = (np.minimum(P["ncomponents"], 6) >= opts.min_ncomp) & np.isfinite(R) & np.isfinite(P["log10_mass_750"])
        matched = keep & (np.abs(P["log10_mass_750"] - mu) < opts.mass_window)
        sims[dom] = {"R": R, "p": p_sidm, "cs": P["cross_section"], "keep": keep, "matched": matched}

    device = ms.get_device()
    models = ms.load_models(SimpleNamespace(filter="concat", model=opts.model, include_str=opts.include_str,
                                            nmodels=opts.nmodels), ms.setup_args("concat"), device)
    print(f"{len(models)} models: {opts.model} containing '{opts.include_str}'")
    raw_a, cat, ra, dec = sens.a2744_map()
    c1, c2 = galaxy_distortion(cat)
    _, inverse = sens.stamp_to_lenstool(cat, ra, dec)
    obs = {"adopted": a2744(models, device, cat, ra, dec, c1, c2, rng, opts.nboot)}
    for h, v in sens.LENSTOOL_HALOS.items():
        dx, dy = inverse(*v)
        obs[h] = a2744(models, device, cat, ra - dx, dec - dy, c1, c2, rng, opts.nboot)
    for k, v in obs.items():
        lo, med, hi = v["R_boot_16_50_84"]
        print(f"A2744 {k:8s}: R = {v['R']:.3f} (bootstrap 16-84%: {lo:.3f}-{hi:.3f}), P(SIDM) {v['p_sidm']:.3f}")

    res = {"inner_kpc": [20 * x for x in INNER], "outer_kpc": [20 * x for x in OUTER], "a2744": obs, "sims": {}}
    for dom, s in sims.items():
        res["sims"][dom] = {}
        R0 = obs["adopted"]["R"]
        print(f"\n{dom.upper()} (n_comp >= {opts.min_ncomp}; matched: |log M - {mu:.2f}| < {opts.mass_window})")
        for c in np.unique(s["cs"]):
            for sub in ("keep", "matched"):
                m = s[sub] & (s["cs"] == c)
                if m.sum() < 10:
                    continue
                rho, pv = spearmanr(s["R"][m], s["p"][m])
                frac = float(np.mean(s["R"][m] >= R0))
                q = np.percentile(s["R"][m], [5, 16, 50, 84, 95])
                res["sims"][dom][f"{c:g}_{sub}"] = {"n": int(m.sum()), "R_5_16_50_84_95": q.tolist(),
                                                    "frac_R_ge_a2744": frac, "spearman_R_p": [float(rho), float(pv)],
                                                    "p_sidm_mean": float(s["p"][m].mean())}
                print(f"  σ={c:<4g} {sub:7s} n={m.sum():5d}  R 16/50/84% {q[1]:.3f}/{q[2]:.3f}/{q[3]:.3f}  "
                      f"frac R >= A2744 {frac:.3f}  Spearman(R, P) {rho:+.2f} (p {pv:.1g})  <P> {s['p'][m].mean():.3f}")
        # does R predict P beyond sigma/m? least squares P ~ R + one offset per sigma, matched clusters
        m = s["matched"]
        X = np.c_[s["R"][m], (s["cs"][m][:, None] == np.unique(s["cs"])[None]).astype(float)]
        coef = np.linalg.lstsq(X, s["p"][m], rcond=None)[0]
        res["sims"][dom]["dP_dR_given_sigma"] = float(coef[0])
        print(f"  P ~ R + offset(σ), matched: dP/dR = {coef[0]:+.3f}")
    json.dump(res, open(out / "shear_ratio.json", "w"), indent=1)
    plot(sims, obs, out / "shear_ratio")
    print(f"wrote {out / 'shear_ratio'}.png")


def plot(sims, obs, out):
    plt = sens.style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True, sharey=True)
    for ax, (dom, s) in zip(axes, sims.items()):
        for c in np.unique(s["cs"]):
            col = COLOURS.get(float(c), None)
            m = s["keep"] & (s["cs"] == c)
            ax.scatter(s["R"][m], s["p"][m], s=2, color=col, alpha=0.08, lw=0)
            m = s["matched"] & (s["cs"] == c)
            ax.scatter(s["R"][m], s["p"][m], s=9, color=col, alpha=0.6, lw=0, label=f"σ/m = {c:g} (mass-matched)")
        for k, v in obs.items():
            lo, _, hi = v["R_boot_16_50_84"]
            ax.errorbar(v["R"], v["p_sidm"], xerr=[[v["R"] - lo], [hi - v["R"]]], fmt="*" if k == "adopted" else "D",
                        ms=14 if k == "adopted" else 7, color="#E69F00" if k == "adopted" else "#56B4E9",
                        mec="k", capsize=2, zorder=5, label=f"A2744 {k}" if k == "adopted" else None)
            if k != "adopted":
                ax.annotate(k, (v["R"], v["p_sidm"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
        lo, hi = np.nanpercentile(s["R"][s["keep"]], [0.5, 99.5])
        ra = [v["R_boot_16_50_84"][i] for v in obs.values() for i in (0, 2)]
        ax.set_xlim(min(lo, *ra) - 0.05, max(hi, *ra) + 0.05)
        ax.set_xlabel(r"R = $\langle\chi_t\rangle$(200-400 kpc) / $\langle\chi_t\rangle$(50-200 kpc)")
        ax.set_title(f"{dom.upper()}, n_comp ≥ 2 (faint: all masses)", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper left", markerscale=1.5)
    axes[0].set_ylabel("ensemble P(SIDM)")
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=150)


if __name__ == "__main__":
    main()
