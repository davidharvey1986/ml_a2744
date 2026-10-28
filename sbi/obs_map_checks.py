#!/usr/bin/env python
"""
Sanity checks of the A2744 CNN input against the simulated maps it is compared with.

The data map (add_shear_to_data.make_obs_maps, map 0) is the per-pixel mean of the
galaxy distortions chi = 2 gamma / (1 + |gamma|^2) from the catalogue shear estimates.
The simulated maps are the reduced shear sampled at the same galaxies (galaxies with
kappa >= KAPPA_THRESH dropped), rescaled to the source redshifts and combined with
per-galaxy shape noise drawn from a chi fit to the catalogue ellipticities
(dataset.apply_intrinsic_ell). Both are normalised as BalancedClusterDataset.

Checks, A2744 against simulated BAHAMAS clusters matched to its mass and merger class:

  1  per galaxy   |chi| of the data (from gamma1/2) against the simulated noise |chi|
                  (from the chi fit to e1/e2), and the ratio |gamma| / |e|
  2  conventions  tangential and cross chi profiles about the map centre: a flipped
                  axis or e2 sign moves signal into the cross component
  3  amplitude    core and outer-field chi, pixel scatter, mean (additive bias)
  4  normalisation  min / max of each raw map, which set the scale of the CNN input,
                  and the value an empty pixel takes after normalisation
  5  kappa cut    pixels with galaxies in the data that are empty in the simulations
  6  pure noise   CNN P(SIDM) of A2744's rotated (noise-only) maps against simulated
                  noise-only maps through the same forward model
  7  centring     the A2744 stamp is centred on the median galaxy position, so its
                  main mass peak (Lenstool O2) sits ~700 kpc from the centre, while the
                  simulated clusters are centred. The matched clusters are rebuilt from
                  their convergence maps shifted to A2744's peak (same galaxies, kappa cut,
                  noise and normalisation as add_shear_to_data / make_summaries) and run
                  through the CNN, against the same clusters unshifted

Outputs: sbi/results/obs_map_checks/{checks.json, checks.png}

Usage
-----
    cd ml_check && astro-env/bin/python sbi/obs_map_checks.py
    astro-env/bin/python sbi/obs_map_checks.py --model tweakedalign_k0p9_bonsai_best --n 100
"""

import argparse
import json
import pickle as pkl
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import make_summaries as ms  # noqa: E402  (chdirs to notebooks/)
from dataset import MAP_ZL, MAP_ZS  # noqa: E402
import sensitivity as sens  # noqa: E402
from add_shear_to_data import (  # noqa: E402
    KAPPA_THRESH, SHEAR_CUTS, data_to_shear, galaxy_distortion, get_background_sigma, get_boxsize, get_obs_data,
    get_surface_density_norms, ra_dec_to_simulation_image_pos, sample_maps_on_galaxies,
)
from lenspack.image.inversion import ks93  # noqa: E402
from scipy.ndimage import gaussian_filter, shift as nd_shift  # noqa: E402
from mass_weighting import a2744_mass_gaussian  # noqa: E402

NPIX = 100
_c = np.arange(NPIX) - (NPIX / 2 - 0.5)
XG, YG = np.meshgrid(_c, _c)                       # maps are indexed [dec, ra]
RG = np.hypot(XG, YG)
PHI = np.arctan2(YG, XG)
R_BINS = np.array([0, 5, 10, 15, 20, 30, 40, 50, 71])


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="tweakedalign_ba_best")
    p.add_argument("--include-str", default="nob1", help="only checkpoints whose name contains this (e.g. spin2)")
    p.add_argument("--nmodels", type=int, default=5)
    p.add_argument("--n", type=int, default=60, help="matched clusters per cross-section (CDM and the largest)")
    p.add_argument("--summaries", default=str(REPO / "sbi" / "summaries_ba_best_n5"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(REPO / "sbi" / "results" / "obs_map_checks"))
    return p.parse_args()


def tangential(raw):
    '''Tangential and cross chi per pixel about the map centre.'''
    c2, s2 = np.cos(2 * PHI), np.sin(2 * PHI)
    return -(raw[0] * c2 + raw[1] * s2), raw[0] * s2 - raw[1] * c2


def profile(field, occupied):
    return np.array([field[occupied & (RG >= lo) & (RG < hi)].mean()
                     for lo, hi in zip(R_BINS[:-1], R_BINS[1:])])


def ks_peak(raw, smooth=3):
    '''(row, col) of the smoothed Kaiser-Squires convergence peak, in pixels from the map centre.'''
    ke, _ = ks93(raw[0] / 2., raw[1] / 2.)
    iy, ix = np.unravel_index(np.argmax(gaussian_filter(ke, smooth)), ke.shape)
    return iy - (NPIX / 2 - 0.5), ix - (NPIX / 2 - 0.5)


def offset_test(sel, offset, models, transform, device):
    '''
    P(SIDM) of the selected clusters rebuilt with the obs pipeline from their convergence
    maps, unshifted and shifted by offset (rows, cols), per cross-section.
    '''
    cat = get_obs_data("concat", data_dir="../data/100/a2744", photoz=True, cuts=SHEAR_CUTS)
    ra, dec = ra_dec_to_simulation_image_pos(cat, zl=0.305, image_size=NPIX)
    out = {}
    for s in np.unique(sel["cross_section"]):
        rows = np.where(sel["cross_section"] == s)[0]
        conv = f"../data/100/convergence/{sel['filename'][rows[0]]}"
        idx = sel["image_idx"][rows]
        meta, data = pkl.load(open(conv, "rb"))
        if "redshift" not in meta:
            meta["redshift"] = np.full(data.shape[0], 0.305)
        norms, bkg = get_surface_density_norms(meta, conv), get_background_sigma(meta, conv, meta["redshift"])
        for lab, off in [("centred", (0, 0)), ("at_a2744_offset", offset)]:
            imgs = np.array([nd_shift(data[i, 0], off, order=1, mode="nearest") for i in idx])
            e1, e2, k = data_to_shear(imgs, norms[idx, 0], meta["redshift"][idx], get_boxsize(conv),
                                      reduce_shear=False, background=bkg[idx], zs=MAP_ZS, zl=MAP_ZL)
            maps, _ = sample_maps_on_galaxies(e1, e2, k, ra, dec, thresh_k=KAPPA_THRESH, reduce_shear=True, npix=NPIX)
            torch.manual_seed(1)
            np.random.seed(1)
            imgs_n = torch.stack([sens.normalise(transform(torch.tensor(m, dtype=torch.float32)))[0] for m in maps])
            p = sens.model_outputs(models, imgs_n, device, 64).mean(0)
            out[f"{s:g}_{lab}"] = [float(p.mean()), float(p.std() / np.sqrt(len(p))), len(p)]
    return out


def map_stats(raw, occupied):
    t, x = tangential(raw)
    vals = raw[:, occupied]
    mn, mx = raw.min(), raw.max()
    return {"chi_t_profile": profile(t, occupied), "chi_x_profile": profile(x, occupied),
            "mean_chi": vals.mean(1), "std_chi": vals.std(1),
            "std_outer": raw[:, occupied & (RG > 35)].std(1),
            "min": float(mn), "max": float(mx), "empty_value": float((0 - mn) / mx)}


def main():
    opts = parse()
    out = Path(opts.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(opts.seed)
    device = ms.get_device()
    args = ms.setup_args("concat")
    results = {}

    # 1. per-galaxy distortions: data (gamma) against the noise model (fit to e)
    cat = get_obs_data("concat", data_dir="../data/100/a2744/", cuts=SHEAR_CUTS, drop_strong_lensing=True)
    c1, c2 = galaxy_distortion(cat)
    chi_data = np.hypot(c1, c2)
    noise = pkl.load(open("../data/100/a2744/ngal_concat.pkl", "rb"))
    from scipy.stats import chi as chi_dist
    df, loc, scale = noise["chi_fit_function"]
    e_sim = chi_dist.rvs(df=df, loc=loc, scale=scale, size=200_000, random_state=1)
    chi_sim = 2 * e_sim / (1 + e_sim ** 2)
    g_over_e = np.hypot(cat["gamma1"], cat["gamma2"]) / np.hypot(cat["e1"], cat["e2"])
    q = [5, 25, 50, 75, 95, 99]
    results["per_galaxy"] = {
        "n_galaxies": int(len(chi_data)),
        "chi_data_percentiles": dict(zip(map(str, q), np.percentile(chi_data, q).round(4).tolist())),
        "chi_noise_model_percentiles": dict(zip(map(str, q), np.percentile(chi_sim, q).round(4).tolist())),
        "rms_chi_component_data": float(np.sqrt(np.mean(np.r_[c1, c2] ** 2))),
        "rms_chi_component_noise_model": float(np.sqrt(np.mean(chi_sim ** 2) / 2)),
        "gamma_over_e_percentiles": dict(zip(map(str, q), np.percentile(g_over_e[np.isfinite(g_over_e)], q).round(3).tolist())),
        "mean_chi1_chi2_data": [float(c1.mean()), float(c2.mean())],
    }

    # A2744 data map and its noise-only (rotated) maps
    raw_a, _, _, _ = sens.a2744_map()
    occupied = np.any(raw_a != 0, axis=0)
    _, stored = pkl.load(open("../data/100/a2744/obs_data_concat.pkl", "rb"))
    results["a2744"] = map_stats(raw_a, occupied)

    # simulated matched clusters, forward modelled exactly as in the summaries
    mu, sig = a2744_mass_gaussian()
    sel = sens.pick_clusters(opts.summaries, "bahamas", mu, sig, opts.n, rng)
    keep = np.isin(sel["cross_section"], [0, sel["cross_section"].max()])
    sel = {k: v[keep] for k, v in sel.items()}
    raws, _ = sens.simulated_maps(sel, args, opts.seed)
    sims = {}
    for s in np.unique(sel["cross_section"]):
        idx = np.where(sel["cross_section"] == s)[0]
        st = [map_stats(raws[i].numpy(), occupied) for i in idx]
        sims[f"{s:g}"] = st
        results[f"bahamas_{s:g}"] = {k: (np.mean([x[k] for x in st], axis=0).tolist() if np.ndim(st[0][k]) else
                                         [float(np.percentile([x[k] for x in st], p)) for p in (16, 50, 84)])
                                     for k in st[0]}

    # 5. kappa cut: data pixels with galaxies that the noise-free simulated map leaves empty
    empty_frac = []
    for f in np.unique(sel["filename"]):
        _, obs = pkl.load(open(f"../data/100/obs/concat/{f}", "rb"))
        for i in sel["image_idx"][sel["filename"] == f]:
            empty_frac.append(np.mean(np.all(obs[i, :2] == 0, axis=0)[occupied]))
    results["kappa_cut"] = {"fraction_of_occupied_pixels_emptied_in_sims": [float(np.percentile(empty_frac, p)) for p in (16, 50, 84)],
                            "n_pixels_occupied_in_data": int(occupied.sum())}

    # 6. CNN on pure noise: A2744 rotated maps vs simulated noise-only maps
    models = ms.load_models(SimpleNamespace(filter="concat", model=opts.model, include_str=opts.include_str,
                                            nmodels=opts.nmodels), args, device)
    transform = ms.get_test_transform(args)
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    _, obs0 = pkl.load(open(f"../data/100/obs/concat/{sel['filename'][0]}", "rb"))
    template = torch.tensor(obs0[sel["image_idx"][0], :2], dtype=torch.float32)
    tiny = torch.where(template != 0, torch.full_like(template, 1e-9), template)   # same mask, no signal
    noise_maps = torch.stack([sens.normalise(transform(tiny.clone()))[0] for _ in range(50)])
    data_maps = torch.tensor(stored, dtype=torch.float32)                          # already normalised
    p_noise = sens.model_outputs(models, noise_maps.to(torch.float32), device, 64).mean(0)
    p_data = sens.model_outputs(models, data_maps, device, 8).mean(0)
    sim_full = torch.stack([sens.normalise(r)[0] for r in raws])
    p_sims = sens.model_outputs(models, sim_full, device, 64).mean(0)
    results["cnn"] = {
        "model": opts.model, "n_models": len(models),
        "a2744_data": float(p_data[0]),
        "a2744_rotated_noise_only": p_data[1:].round(4).tolist(),
        "sim_noise_only_mean_std": [float(p_noise.mean()), float(p_noise.std())],
        **{f"bahamas_{s:g}_matched_mean": float(p_sims[sel["cross_section"] == s].mean())
           for s in np.unique(sel["cross_section"])},
    }

    # 7. centring: matched clusters moved to A2744's convergence peak
    offset = ks_peak(raw_a)
    results["centring"] = {"a2744_ks_peak_rowcol_px": list(offset),
                           "a2744_peak_offset_kpc": float(20 * np.hypot(*offset)),
                           "p_sidm_mean_err_n": offset_test(sel, offset, models, transform, device)}

    json.dump(results, open(out / "checks.json", "w"), indent=1, default=lambda o: np.asarray(o).tolist())
    plot(results, sims, chi_data, chi_sim, raw_a, occupied, out)
    report(results)


def plot(results, sims, chi_data, chi_sim, raw_a, occupied, out):
    plt = sens.style()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    rc = 0.5 * (R_BINS[1:] + R_BINS[:-1]) * 20
    ax = axes[0, 0]
    bins = np.linspace(0, 1.2, 61)
    ax.hist(chi_data, bins, density=True, histtype="step", lw=2, color="k", label="A2744 galaxies (from gamma)")
    ax.hist(chi_sim, bins, density=True, histtype="step", lw=2, color="#D55E00", label="noise model (chi fit to e)")
    ax.set_xlabel(r"$|\chi|$ per galaxy")
    ax.legend()
    for j, (key, lab) in enumerate([("chi_t_profile", r"tangential $\chi_t$"), ("chi_x_profile", r"cross $\chi_\times$")]):
        ax = axes[0, 1 + j]
        for s, st in sims.items():
            arr = np.array([x[key] for x in st])
            ax.fill_between(rc, *np.percentile(arr, [16, 84], axis=0), alpha=0.2,
                            color="#56B4E9" if s == "0" else "#0072B2", lw=0)
            ax.plot(rc, arr.mean(0), color="#56B4E9" if s == "0" else "#0072B2", lw=2,
                    label=f"BAHAMAS {'CDM' if s == '0' else 'σ/m=' + s} (mean, 16-84%)")
        ax.plot(rc, results["a2744"][key], "o-", color="k", lw=2, label="A2744")
        ax.axhline(0, color="#6b6b6b", lw=1)
        ax.set_xlabel("radius from map centre [kpc]")
        ax.set_ylabel(lab + " (pixel mean)")
        ax.legend(fontsize=8)
    for j, (key, lab) in enumerate([("min", "raw map minimum"), ("max", "raw map maximum"), ("empty_value", "empty pixel after normalisation")]):
        ax = axes[1, j]
        for s, st in sims.items():
            ax.hist([x[key] for x in st], 20, alpha=0.5, color="#56B4E9" if s == "0" else "#0072B2",
                    label=f"BAHAMAS {'CDM' if s == '0' else 'σ/m=' + s}")
        ax.axvline(results["a2744"][key], color="k", lw=2, label="A2744")
        ax.set_xlabel(lab)
        ax.legend(fontsize=8)
    fig.suptitle("A2744 CNN input against mass-matched BAHAMAS clusters, same galaxies and forward model")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"checks.{ext}", dpi=150)


def report(r):
    g = r["per_galaxy"]
    print(f"1 per galaxy ({g['n_galaxies']}): |chi| data {g['chi_data_percentiles']}")
    print(f"             noise model {g['chi_noise_model_percentiles']}")
    print(f"  rms chi component: data {g['rms_chi_component_data']:.4f}, noise model {g['rms_chi_component_noise_model']:.4f}; "
          f"|gamma|/|e| {g['gamma_over_e_percentiles']}; mean chi1, chi2 {np.round(g['mean_chi1_chi2_data'], 4)}")
    for k in ["a2744"] + [k for k in r if k.startswith("bahamas_")]:
        v = r[k]
        print(f"{k:14s} chi_t {np.round(v['chi_t_profile'], 3)}\n{'':14s} chi_x {np.round(v['chi_x_profile'], 3)}\n"
              f"{'':14s} pixel std {np.round(v['std_chi'], 3)} outer std {np.round(v['std_outer'], 3)} mean {np.round(v['mean_chi'], 4)} "
              f"min {np.round(v['min'], 3)} max {np.round(v['max'], 3)} empty {np.round(v['empty_value'], 3)}")
    print(f"5 kappa cut: {r['kappa_cut']}")
    print(f"6 CNN: {r['cnn']}")
    print(f"7 centring: {r['centring']}")


if __name__ == "__main__":
    main()
