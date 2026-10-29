#!/usr/bin/env python
'''
Is A2744's high P(SIDM) its three-halo geometry? Build synthetic triples from BAHAMAS clusters
of one cross-section: a main cluster at the stamp centre, matched to the mass of A2744's core
(Lenstool O2), plus two lower-mass clusters placed where the NW (O1) and N (O3) halos sit
relative to O2 in the adopted A2744 stamp, scaled to their masses (mass_weighting.A2744_MASSES).
Every map goes through the forward model of the simulated obs maps (add_shear_to_data.main:
shear at MAP_ZS sampled at the A2744 galaxies, kappa >= KAPPA_THRESH dropped, reduced shear,
then the test transform's source-redshift rescaling and shape noise, and the normalisation),
and the CNN P(SIDM) is compared between variants with the same main clusters and noise draws:

    single      the main cluster alone
    +O1, +O3    with one secondary
    triple      with both
    total-mass  clusters matched to A2744's summed mass in the merger class n_comp >= 2
                (the usual reference; drawn separately)

If CDM triples reach A2744's value, the network reads the geometry, not the cross-section.

Secondaries: clusters of the same cross-section with log M750 < --sec-max-logm (preferring
n_comp = 1), whose convergence (background subtracted, as data_to_shear) has the median over
45 < r < 50 px removed and is tapered to zero between 35 and 50 px, scaled by
M_halo / M750, shifted to the halo's position on a zero-padded canvas and Kaiser-Squires
inverted there (no wrap-around), then added to the main cluster's shear and convergence. The
main cluster's own shear is exactly the training pipeline's (unpadded KS).

    astro-env/bin/python sbi/synthetic_triples.py --include-str spin2_ft_
'''
import argparse
import json
import pickle as pkl
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from lenspack.image.inversion import ks93inv
from scipy.ndimage import shift as nd_shift

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import make_summaries as ms  # noqa: E402  (chdirs to notebooks/)
import sensitivity as sens  # noqa: E402
from dataset import MAP_ZL, MAP_ZS  # noqa: E402
from add_shear_to_data import (  # noqa: E402
    KAPPA_THRESH, SHEAR_CUTS, data_to_shear, get_background_sigma, get_boxsize, get_obs_data,
    get_surface_density_norms, ra_dec_to_simulation_image_pos, sample_maps_on_galaxies,
)
from mass_weighting import A2744_ERRORS, A2744_MASSES, a2744_mass_gaussian  # noqa: E402

NPIX = 100
PAD = 300
_c = np.arange(NPIX) - (NPIX / 2 - 0.5)
RG = np.hypot(*np.meshgrid(_c, _c))
TAPER = np.clip(np.cos(0.5 * np.pi * (RG - 35) / 15), 0, 1) * (RG < 50)
TAPER[RG <= 35] = 1.
HALO_KEYS = {"O1": "nw", "O3": "n"}
VARIANTS = ["single", "+O1", "+O3", "triple"]


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="tweakedalign_k0p9_ctr_ba_best")
    p.add_argument("--include-str", default="spin2_ft_")
    p.add_argument("--nmodels", type=int, default=None)
    p.add_argument("--study", default="harvey", choices=sorted(A2744_MASSES))
    p.add_argument("--n", type=int, default=60, help="main clusters per cross-section")
    p.add_argument("--min-sigma", type=float, default=0.1, help="floor on the width of the core-mass Gaussian [dex]")
    p.add_argument("--main-min-ncomp", type=int, default=1, help="merger class of the main clusters: n_comp >= this")
    p.add_argument("--sec-max-logm", type=float, default=14.25)
    p.add_argument("--summaries", default=str(REPO / "sbi" / "summaries_spin2_best_n5"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(REPO / "sbi" / "results" / "synthetic_triples"))
    return p.parse_args()


def load_set(filename):
    conv = f"../data/100/convergence/{filename}"
    meta, data = pkl.load(open(conv, "rb"))
    if "redshift" not in meta:
        meta["redshift"] = np.full(data.shape[0], 0.305)
    return conv, meta, data


def shear(conv, meta, data, idx):
    '''Non-reduced shear and convergence of maps idx at MAP_ZS, exactly as add_shear_to_data.main.'''
    norms, bkg = get_surface_density_norms(meta, conv), get_background_sigma(meta, conv, meta["redshift"][idx])
    return data_to_shear(data[idx, 0], norms[idx, 0], meta["redshift"][idx], get_boxsize(conv),
                         reduce_shear=False, background=bkg, zs=MAP_ZS, zl=MAP_ZL)


def place(kappa, scale, dx, dy):
    '''Shear and convergence on the stamp of a tapered, scaled halo moved by (dx, dy) pixels, KS on a padded canvas.'''
    k = scale * TAPER * (kappa - np.median(kappa[(RG > 45) & (RG < 50)]))
    canvas = np.zeros((PAD, PAD))
    o = (PAD - NPIX) // 2
    canvas[o:o + NPIX, o:o + NPIX] = k
    canvas = nd_shift(canvas, (dy, dx), order=1, mode="constant", cval=0.)
    g1, g2 = ks93inv(canvas, np.zeros_like(canvas))
    crop = (slice(o, o + NPIX), slice(o, o + NPIX))
    return g1[crop], g2[crop], canvas[crop]


def cnn(models, raws, transform, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    imgs = torch.stack([sens.normalise(transform(torch.tensor(r, dtype=torch.float32)))[0] for r in raws])
    return sens.model_outputs(models, imgs, device, 64).mean(0)


def main():
    opts = parse()
    out = Path(opts.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(opts.seed)
    device = ms.get_device()
    args = ms.setup_args("concat")
    transform = ms.get_test_transform(args)
    models = ms.load_models(SimpleNamespace(filter="concat", model=opts.model, include_str=opts.include_str,
                                            nmodels=opts.nmodels), args, device)
    print(f"{len(models)} models: {opts.model} containing '{opts.include_str}'")

    # A2744: its P(SIDM), and the halo positions relative to O2 on the adopted stamp
    raw_a, cat_a, ra_a, dec_a = sens.a2744_map()
    p_a = float(sens.model_outputs(models, sens.normalise(torch.tensor(raw_a))[0][None], device, 1).mean())
    _, inverse = sens.stamp_to_lenstool(cat_a, ra_a, dec_a)
    o2 = inverse(*sens.LENSTOOL_HALOS["O2"])
    offsets = {h: tuple(inverse(*sens.LENSTOOL_HALOS[h]) - o2) for h in HALO_KEYS}
    for h, (dx, dy) in offsets.items():
        print(f"{h} from O2: dx {dx:+.1f} px, dy {dy:+.1f} px ({20 * np.hypot(dx, dy):.0f} kpc)")

    # galaxies as the simulated obs maps (add_shear_to_data.main)
    cat = get_obs_data("concat", data_dir="../data/100/a2744", photoz=True, cuts=SHEAR_CUTS)
    ra, dec = ra_dec_to_simulation_image_pos(cat, zl=MAP_ZL, image_size=NPIX)

    masses = A2744_MASSES[opts.study]
    log_m = {k: np.log10(v * 1e13) for k, v in masses.items()}
    lo, hi = A2744_ERRORS[opts.study]["core"]
    mu_core = log_m["core"]
    sig_core = max(0.5 * np.log10((masses["core"] + hi) / (masses["core"] - lo)), opts.min_sigma)
    P = dict(np.load(Path(opts.summaries) / "params_bahamas.npz"))

    sens.A2744_NCOMP = range(opts.main_min_ncomp, 7)
    mains = sens.pick_clusters(opts.summaries, "bahamas", mu_core, sig_core, opts.n, rng)
    sens.A2744_NCOMP = range(2, 7)
    mu_tot, sig_tot = a2744_mass_gaussian(opts.study)
    totals = sens.pick_clusters(opts.summaries, "bahamas", mu_tot, sig_tot, opts.n, rng)

    res = {"study": opts.study, "model": opts.model, "include_str": opts.include_str, "n_models": len(models),
           "a2744_p_sidm": p_a, "log_masses": log_m, "core_gaussian": [mu_core, sig_core],
           "offsets_px": offsets, "o2_px": o2, "rows": {}}
    examples = {}
    for s in np.unique(mains["cross_section"]):
        rows = np.where(mains["cross_section"] == s)[0]
        conv, meta, data = load_set(mains["filename"][rows[0]])
        idx = mains["image_idx"][rows]
        e1, e2, k = shear(conv, meta, data, idx)

        # secondaries of the same cross-section, low mass, preferring n_comp = 1
        cand = np.where((P["cross_section"] == s) & (P["log10_mass_750"] < opts.sec_max_logm))[0]
        relaxed = cand[P["ncomponents"][cand] == 1]
        cand = relaxed if len(relaxed) >= 20 else cand
        sec = {h: rng.choice(cand, len(idx)) for h in HALO_KEYS}
        add = {}
        for h, key in HALO_KEYS.items():
            j = P["image_idx"][sec[h]]
            _, _, ks = shear(conv, meta, data, j)
            scale = 10 ** (log_m[key] - P["log10_mass_750"][sec[h]])
            add[h] = np.array([place(kk, sc, *offsets[h]) for kk, sc in zip(ks, scale)])   # (n, 3, npix, npix)

        res["rows"][f"{s:g}"] = row = {}
        p_var = {}
        for v in VARIANTS:
            hs = {"single": [], "+O1": ["O1"], "+O3": ["O3"], "triple": ["O1", "O3"]}[v]
            g1, g2, kap = e1.copy(), e2.copy(), k.copy()
            for h in hs:
                g1, g2, kap = g1 + add[h][:, 0], g2 + add[h][:, 1], kap + add[h][:, 2]
            maps, removed = sample_maps_on_galaxies(g1, g2, kap, ra, dec, thresh_k=KAPPA_THRESH, reduce_shear=True,
                                                    npix=NPIX)
            if v == "single":                    # must reproduce the stored obs maps
                _, obs = pkl.load(open(f"../data/100/obs/concat/{mains['filename'][rows[0]]}", "rb"))
                assert np.allclose(maps[:5], obs[idx[:5], :2], atol=1e-5), "forward model differs from the obs maps"
            p = cnn(models, maps, transform, device, opts.seed)
            p_var[v] = p
            row[v] = [float(p.mean()), float(p.std() / np.sqrt(len(p))), len(p), float(np.median(removed))]
            if s == 0 and v in ("single", "triple"):
                examples[v] = (kap[0], maps[0])
        for v in VARIANTS[1:]:
            d = p_var[v] - p_var["single"]
            row[f"{v}_minus_single"] = [float(d.mean()), float(d.std() / np.sqrt(len(d)))]

        t = np.where(totals["cross_section"] == s)[0]
        assert totals["filename"][t[0]] == mains["filename"][rows[0]]
        conv_t, meta_t, data_t = conv, meta, data
        e1t, e2t, kt = shear(conv_t, meta_t, data_t, totals["image_idx"][t])
        maps_t, _ = sample_maps_on_galaxies(e1t, e2t, kt, ra, dec, thresh_k=KAPPA_THRESH, reduce_shear=True, npix=NPIX)
        p = cnn(models, maps_t, transform, device, opts.seed)
        row["total-mass"] = [float(p.mean()), float(p.std() / np.sqrt(len(p))), len(p)]
        print(f"σ/m {s:g}: " + " | ".join(f"{v} {row[v][0]:.3f}±{row[v][1]:.3f}" for v in VARIANTS + ["total-mass"]))
        print("          " + " | ".join(f"{v}−single {row[f'{v}_minus_single'][0]:+.3f}±{row[f'{v}_minus_single'][1]:.3f}"
                                       for v in VARIANTS[1:]))
        del data, data_t

    tag = f"{opts.model}_{opts.include_str}_{opts.study}"
    json.dump(res, open(out / f"{tag}.json", "w"), indent=1, default=lambda o: np.asarray(o).tolist())
    plot(res, examples, raw_a, offsets, o2, out / tag)
    print(f"wrote {out / tag}.png")


def plot(res, examples, raw_a, offsets, o2, out):
    plt = sens.style()
    fig = plt.figure(figsize=(15, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.6, 1, 1, 1])
    ax = fig.add_subplot(gs[0])
    sig = np.array([float(s) for s in res["rows"]])
    x = np.where(sig == 0, 0.03, sig)
    cols = {"single": "#000000", "+O1": "#CC79A7", "+O3": "#009E73", "triple": "#D55E00", "total-mass": "#0072B2"}
    for v, c in cols.items():
        y = np.array([res["rows"][s][v] for s in res["rows"]])
        ax.errorbar(x, y[:, 0], y[:, 1], fmt="o-" if v != "total-mass" else "s--", color=c, capsize=2, ms=5,
                    label={"single": "main (core mass) alone", "triple": "main + O1 + O3",
                           "total-mass": "single, total A2744 mass (n_comp ≥ 2)"}.get(v, f"main + {v[1:]}"))
    ax.axhline(res["a2744_p_sidm"], color="k", lw=2, alpha=0.5, label=f"A2744: {res['a2744_p_sidm']:.3f}")
    ax.set_xscale("log")
    ax.set_xticks([0.03, 0.1, 0.3, 1], ["CDM", "0.1", "0.3", "1"])
    ax.minorticks_off()
    ax.set_xlabel("σ/m of all components [cm²/g]")
    ax.set_ylabel("ensemble P(SIDM)")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title(f"BAHAMAS synthetic triples ({res['study']} masses, {res['n_models']} models)", fontsize=10)
    ext = [-50, 50, -50, 50]
    kap_a = sens.reconstruct_kappa(raw_a)[0]
    for i, (title, img) in enumerate([("CDM main alone: κ", examples["single"][0]),
                                      ("CDM triple: κ", examples["triple"][0]),
                                      ("A2744: KS κ from the data", kap_a)]):
        a = fig.add_subplot(gs[i + 1])
        a.imshow(img, origin="lower", extent=ext, cmap="magma",
                 vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.7))
        c = o2 if i == 2 else (0, 0)          # A2744's stamp is not centred on O2
        for dx, dy in [(0, 0), *offsets.values()]:
            a.plot(c[0] + dx, c[1] + dy, "c+", ms=10, mew=1.5)
        a.set_title(title, fontsize=10)
        a.set_xticks([])
        a.set_yticks([])
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=150)


if __name__ == "__main__":
    main()
