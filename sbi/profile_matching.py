#!/usr/bin/env python
'''
Match the simulated clusters to A2744 on its tangential shear profile instead of its
M(<750 kpc), in the stamp centred on A2744's main halo (Lenstool O2), and compare where
A2744's P(SIDM) falls among the matched clusters.

For every simulated cluster the noise-free CNN input is chi = 2 g' / (1 + |g'|^2), with g'
the stored reduced shear (data/100/obs/concat_centred, sampled at the A2744 galaxies)
rescaled to the per-pixel source redshift: exactly the test transform with no shape
noise. Its tangential chi profile about the map centre is compared to A2744's, with the
covariance of A2744's profile from random rotations of its galaxies (shape noise only).
The inner 100 kpc is left out: the simulations drop galaxies with kappa >= 0.9, A2744
does not. Each cluster gets the weight exp(-chi^2 / 2).

    astro-env/bin/python sbi/profile_matching.py \
        --summaries $PWD/sbi/summaries_k0p9_ctr_best_n5
'''
import argparse
import json
import pickle as pkl
import sys
from pathlib import Path

import numpy as np
from astropy import cosmology
from lenspack.utils import bin2d, sigma_critical
from scipy.stats import binned_statistic_2d

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import make_summaries as ms  # noqa: E402  (chdirs to notebooks/)
import sensitivity as sens  # noqa: E402
from add_shear_to_data import A2744_STAMP_CENTRE, SHEAR_CUTS, galaxy_distortion, get_obs_data, normalise_map, ra_dec_to_simulation_image_pos  # noqa: E402
from dataset import SIGMA_CRIT_MAP  # noqa: E402
from mass_weighting import a2744_mass_gaussian, gaussian_mass_weights  # noqa: E402
from obs_map_checks import RG, tangential  # noqa: E402

NPIX = 100
O2_PX = A2744_STAMP_CENTRE                      # stamp centre (main halo), catalogue pixels
FIT_BINS = np.array([5, 10, 15, 20, 30, 40, 50, 71])   # pixels (20 kpc); inner 100 kpc left out
N_ROT = 500


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summaries", default=str(REPO / "sbi" / "summaries_k0p9_ctr_best_n5"))
    p.add_argument("--maps", default=str(REPO / "data" / "100" / "obs" / "concat_centred"))
    p.add_argument("--obs", default=str(REPO / "data" / "100" / "a2744" / "obs_data_concat_centred.pkl"))
    p.add_argument("--domains", nargs="+", default=["bahamas", "darkskies"])
    p.add_argument("--rmax", type=int, default=71, help="outer edge of the fitted profile [pixels, 20 kpc]")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(REPO / "sbi" / "results" / "profile_matching"))
    return p.parse_args()


def profile(raw, occupied):
    t, _ = tangential(raw)
    return np.array([t[occupied & (RG >= lo) & (RG < hi)].mean() for lo, hi in zip(FIT_BINS[:-1], FIT_BINS[1:])])


def source_redshift_map():
    '''As get_model_probabilities.get_variable_redshift_dist, about A2744's main halo.'''
    cat = get_obs_data("concat", data_dir="../data/100/a2744/", photoz=True, cuts=SHEAR_CUTS)
    ra, dec = ra_dec_to_simulation_image_pos(cat, ra_0=O2_PX[0], dec_0=O2_PX[1])
    nz = binned_statistic_2d(ra, dec, cat["redshift"], range=[[-50, 50], [-50, 50]],
                             statistic=np.median, bins=NPIX).statistic.T
    nz[~np.isfinite(nz)] = ms.setup_args("concat").default_zs
    return nz


def main():
    opts = parse()
    global FIT_BINS
    FIT_BINS = FIT_BINS[FIT_BINS <= opts.rmax]
    out = Path(opts.out_dir) / f"rmax{20 * opts.rmax}kpc"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(opts.seed)

    # A2744 in the centred stamp, checked against the stored CNN input
    cat = get_obs_data("concat", data_dir="../data/100/a2744/", cuts=SHEAR_CUTS, drop_strong_lensing=True)
    ra, dec = ra_dec_to_simulation_image_pos(cat, ra_0=O2_PX[0], dec_0=O2_PX[1])
    c1, c2 = galaxy_distortion(cat)
    raw_a = np.array(bin2d(ra, dec, v=(c1, c2), npix=NPIX, extent=[-50, 50, -50, 50]))
    _, stored = pkl.load(open(opts.obs, "rb"))
    assert np.allclose(np.array(normalise_map(*raw_a)), stored[0], atol=1e-5), "A2744 map differs from " + opts.obs
    occupied = np.any(raw_a != 0, axis=0)
    d = profile(raw_a, occupied)

    amp = np.hypot(c1, c2)
    rot = []
    for _ in range(N_ROT):
        th = rng.uniform(0, np.pi, amp.size)
        rot.append(profile(np.array(bin2d(ra, dec, v=(amp * np.cos(2 * th), amp * np.sin(2 * th)),
                                         npix=NPIX, extent=[-50, 50, -50, 50])), occupied))
    cov = np.cov(np.array(rot).T)
    icov = np.linalg.inv(cov)
    err = np.sqrt(np.diag(cov))
    rk = 10 * (FIT_BINS[1:] + FIT_BINS[:-1])
    print("A2744 chi_t  " + "  ".join(f"{r:.0f}kpc {v:.3f}±{e:.3f}" for r, v, e in zip(rk, d, err)))

    rescale = SIGMA_CRIT_MAP / sigma_critical(0.305, source_redshift_map(), cosmology.Planck18).value
    mu, sig = a2744_mass_gaussian()
    obs = np.load(Path(opts.summaries) / "obs.npz")
    p_a = float(obs["softmax"][:, 0, 0].mean()) if obs["softmax"].ndim == 3 else float(obs["softmax"][:, 0].mean())
    res = {"summaries": opts.summaries, "a2744_p_sidm": p_a, "a2744_profile": d.tolist(), "a2744_profile_err": err.tolist(),
           "radii_kpc": rk.tolist(), "domains": {}}
    plt = sens.style()
    fig, axes = plt.subplots(1, 2 * len(opts.domains), figsize=(5.5 * 2 * len(opts.domains), 4.4), constrained_layout=True)

    for j, dom in enumerate(opts.domains):
        P = dict(np.load(Path(opts.summaries) / f"params_{dom}.npz"))
        sm = np.load(Path(opts.summaries) / f"softmax_{dom}.npy")[:, 0].mean(0)        # ensemble, noise draw 0
        prof = np.full((len(sm), len(d)), np.nan)
        for f in np.unique(P["filename"]):
            rows = np.where(P["filename"] == f)[0]
            _, maps = pkl.load(open(f"{opts.maps}/{f}", "rb"))
            for r in rows:
                g = maps[P["image_idx"][r], :2].astype(np.float64) * rescale
                chi = 2 * g / (1 + np.sum(g ** 2, axis=0))
                prof[r] = profile(chi, occupied & np.any(g != 0, axis=0))
        resid = prof - d
        chi2 = np.einsum("ij,jk,ik->i", resid, icov, resid)
        w_prof = np.exp(-0.5 * (chi2 - np.nanmin(chi2)))
        w_prof[~np.isfinite(w_prof)] = 0
        cs, m, nc = P["cross_section"], P["log10_mass_750"], P["ncomponents"]
        dres = {"chi2_min": float(np.nanmin(chi2)), "ndof": len(d), "sigma": {}}
        print(f"\n{dom}: best chi2 {np.nanmin(chi2):.1f} for {len(d)} bins")
        for s in np.unique(cs):
            k = cs == s
            row = {}
            for lab, w in [("profile", w_prof * k), ("profile_ncomp3", w_prof * k * (nc >= 3))]:
                n_eff = w.sum() ** 2 / np.sum(w ** 2)
                pm = np.sum(w * sm) / w.sum()
                row[lab] = {"n_eff": float(n_eff), "p_mean": float(pm),
                            "p_std": float(np.sqrt(np.sum(w * (sm - pm) ** 2) / w.sum())),
                            "frac_ge_a2744": float(np.sum(w * (sm >= p_a)) / w.sum()),
                            "log10_mass_750_mean": float(np.nansum(w * m) / w[np.isfinite(m)].sum()),
                            "chi2_median_matched": float(np.median(chi2[k][w[k] > 0.1 * w[k].max()]))}
            km = k & (nc >= 3) & np.isfinite(m)
            wm = np.zeros(len(sm))
            wm[km] = gaussian_mass_weights(m[km], mu, sig)
            pm = np.sum(wm * sm) / wm.sum()
            row["mass_ncomp3"] = {"n_eff": float(wm.sum() ** 2 / np.sum(wm ** 2)), "p_mean": float(pm),
                                  "p_std": float(np.sqrt(np.sum(wm * (sm - pm) ** 2) / wm.sum())),
                                  "frac_ge_a2744": float(np.sum(wm * (sm >= p_a)) / wm.sum()),
                                  "log10_mass_750_mean": float(np.nansum(wm * m) / wm.sum()),
                                  "mean_profile": (np.nansum(wm[:, None] * prof, 0) / wm.sum()).tolist()}
            row["profile"]["mean_profile"] = (np.nansum(w_prof[k][:, None] * prof[k], 0) / w_prof[k].sum()).tolist()
            dres["sigma"][f"{s:g}"] = row
            print(f"  σ={s:<5g} " + " | ".join(
                f"{lab}: n_eff {row[lab]['n_eff']:6.1f}  logM750 {row[lab]['log10_mass_750_mean']:.2f}  "
                f"P {row[lab]['p_mean']:.3f}±{row[lab]['p_std']:.3f}  ≥A2744 {row[lab]['frac_ge_a2744']:.3f}"
                for lab in ["mass_ncomp3", "profile", "profile_ncomp3"]))
        res["domains"][dom] = dres

        colours = plt.cm.Blues if dom == "bahamas" else plt.cm.Oranges
        sig_list = np.unique(cs)
        ax = axes[2 * j]
        for i, s in enumerate(sig_list):
            c = colours(0.35 + 0.6 * i / max(1, len(sig_list) - 1))
            ax.plot(rk, dres["sigma"][f"{s:g}"]["mass_ncomp3"]["mean_profile"], ":", color=c)
            ax.plot(rk, dres["sigma"][f"{s:g}"]["profile"]["mean_profile"], "-", color=c,
                    label=f"σ/m={s:g}" if s else "CDM")
        ax.errorbar(rk, d, err, fmt="o", color="k", label="A2744")
        ax.set_xlabel("radius from the main halo [kpc]")
        ax.set_ylabel(r"tangential $\chi_t$")
        ax.set_title(f"{dom}: profile-matched (solid), mass-matched (dotted)", fontsize=9)
        ax.legend(fontsize=8)
        ax = axes[2 * j + 1]
        for i, s in enumerate(sig_list):
            c = colours(0.35 + 0.6 * i / max(1, len(sig_list) - 1))
            k = cs == s
            ax.hist(sm[k], bins=np.linspace(0, 1, 41), weights=w_prof[k] / w_prof[k].sum(), histtype="step", lw=1.8,
                    color=c, label=f"σ/m={s:g}" if s else "CDM")
        ax.axvline(p_a, color="k", lw=2, label="A2744")
        ax.set_xlabel("ensemble P(SIDM)")
        ax.set_title(f"{dom}: profile-matched clusters", fontsize=9)
        ax.legend(fontsize=8)

    json.dump(res, open(out / "profile_matching.json", "w"), indent=1)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"profile_matching.{ext}", dpi=150)


if __name__ == "__main__":
    main()
