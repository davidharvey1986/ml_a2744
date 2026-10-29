#!/usr/bin/env python
'''
A stricter strong-lensing cut at evaluation time. The A2744 map is rebuilt without the galaxies
where the Lenstool convergence (at z_s = 2.28) is >= each threshold, and mass-matched BAHAMAS and
DARKSKIES clusters (n_comp >= --min-ncomp) are rebuilt from their convergence maps with the
forward model of add_shear_to_data.main, dropping the galaxies where their own convergence is
>= the same threshold (sample_maps_on_galaxies thresh_k), then the test transform (shape noise
composed as complex numbers) and the normalisation. The models are the existing ones, trained on
maps cut at KAPPA_THRESH = 0.9, so this only says whether a stricter cut is worth retraining for.

With --footprint the simulated maps instead lose the same galaxies as A2744: those where A2744's
Lenstool convergence is >= the threshold (on top of their own convergence cut at the same
threshold), so data and simulations have the same holes.

    astro-env/bin/python sbi/kappa_cut_test.py --include-str spin2_ft_ --thresholds 0.9 0.7
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

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import make_summaries as ms  # noqa: E402  (chdirs to notebooks/)
import sensitivity as sens  # noqa: E402
from dataset import MAP_ZL  # noqa: E402
from add_shear_to_data import (  # noqa: E402
    KAPPA_THRESH, SHEAR_CUTS, galaxy_distortion, get_obs_data, ra_dec_to_simulation_image_pos,
    sample_maps_on_galaxies, strong_lensing_kappa,
)
from mass_weighting import a2744_mass_gaussian  # noqa: E402
from synthetic_triples import cnn, load_set, shear  # noqa: E402

NPIX = 100


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="tweakedalign_k0p9_ctr_ba_best")
    p.add_argument("--include-str", default="spin2_ft_")
    p.add_argument("--nmodels", type=int, default=None)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.9, 0.7])
    p.add_argument("--n", type=int, default=150, help="mass-matched clusters per cross-section")
    p.add_argument("--min-ncomp", type=int, default=2)
    p.add_argument("--summaries", default=str(REPO / "sbi" / "summaries_spin2_best_n5_chifix"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--footprint", action="store_true", help="cut the simulations on A2744's Lenstool convergence")
    p.add_argument("--out-dir", default=str(REPO / "sbi" / "results" / "kappa_cut_test"))
    return p.parse_args()


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

    # A2744 as sensitivity.a2744_map, with the strong-lensing cut at each threshold
    cat_a = get_obs_data("concat", data_dir="../data/100/a2744/", cuts=SHEAR_CUTS, drop_strong_lensing=False)
    k_a = strong_lensing_kappa(cat_a)
    xa, ya = ra_dec_to_simulation_image_pos(cat_a)
    c1, c2 = galaxy_distortion(cat_a)
    # galaxies of the simulated maps, as add_shear_to_data.main
    cat = get_obs_data("concat", data_dir="../data/100/a2744", photoz=True, cuts=SHEAR_CUTS)
    ra, dec = ra_dec_to_simulation_image_pos(cat, zl=MAP_ZL, image_size=NPIX)
    k_sim_gal = strong_lensing_kappa(cat)
    sens.a2744_map()                                       # leaves the filtered files for the current cuts

    sens.A2744_NCOMP = range(opts.min_ncomp, 7)
    mu, sig = a2744_mass_gaussian()
    res = {"thresholds": opts.thresholds, "n_models": len(models), "a2744": {}, "sims": {}}
    for th in opts.thresholds:
        keep = k_a < th
        raw = np.array(bin2d(xa[keep], ya[keep], v=(c1[keep], c2[keep]), npix=NPIX, extent=[-50, 50, -50, 50]),
                       dtype=np.float32)
        p = float(sens.model_outputs(models, sens.normalise(torch.tensor(raw))[0][None], device, 1).mean())
        res["a2744"][f"{th:g}"] = {"p_sidm": p, "n_removed": int((~keep).sum())}
        print(f"A2744, Lenstool kappa < {th:g} ({(~keep).sum()} galaxies removed): P(SIDM) {p:.3f}")

    for dom in ("bahamas", "darkskies"):
        sel = sens.pick_clusters(opts.summaries, dom, mu, sig, opts.n, rng)
        res["sims"][dom] = {}
        for s in np.unique(sel["cross_section"]):
            rows = np.where(sel["cross_section"] == s)[0]
            conv, meta, data = load_set(sel["filename"][rows[0]])
            e1, e2, k = shear(conv, meta, data, sel["image_idx"][rows])
            del data
            row = res["sims"][dom][f"{s:g}"] = {}
            for th in opts.thresholds:
                g = k_sim_gal < th if opts.footprint else np.ones(len(ra), bool)
                maps, removed = sample_maps_on_galaxies(e1, e2, k, ra[g], dec[g], thresh_k=th, reduce_shear=True,
                                                        npix=NPIX)
                removed = removed * g.sum() / len(g) + 100 * (1 - g.mean())
                if th == KAPPA_THRESH and not opts.footprint:  # must reproduce the stored obs maps
                    _, obs = pkl.load(open(f"../data/100/obs/concat/{sel['filename'][rows[0]]}", "rb"))
                    assert np.allclose(maps[:5], obs[sel["image_idx"][rows[:5]], :2], atol=1e-5)
                v = cnn(models, maps, transform, device, opts.seed)
                pa = res["a2744"][f"{th:g}"]["p_sidm"]
                row[f"{th:g}"] = {"mean": float(v.mean()), "err": float(v.std() / np.sqrt(len(v))),
                                  "p84": float(np.percentile(v, 84)), "frac_ge_a2744": float(np.mean(v >= pa)),
                                  "pct_galaxies_removed": float(np.median(removed))}
            print(f"{dom:9s} σ={s:<5g} " + " | ".join(
                f"κ<{th:g}: {row[f'{th:g}']['mean']:.3f}±{row[f'{th:g}']['err']:.3f} (≥A2744 {row[f'{th:g}']['frac_ge_a2744']:.2f}, "
                f"{row[f'{th:g}']['pct_galaxies_removed']:.2f}% removed)" for th in opts.thresholds))
    tag = f"{opts.model}_{opts.include_str}" + ("_footprint" if opts.footprint else "")
    res["footprint"] = opts.footprint
    json.dump(res, open(out / f"{tag}.json", "w"), indent=1)
    plot(res, out / tag)


def plot(res, out):
    plt = sens.style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True, sharey=True)
    for ax, dom in zip(axes, ("bahamas", "darkskies")):
        rows = res["sims"][dom]
        sig = np.array([float(s) for s in rows])
        x = np.where(sig == 0, 0.004, sig)
        for th, col in zip(res["thresholds"], ["#000000", "#D55E00", "#0072B2", "#009E73"]):
            y = np.array([[rows[s][f"{th:g}"]["mean"], rows[s][f"{th:g}"]["err"]] for s in rows])
            ax.errorbar(x, y[:, 0], y[:, 1], fmt="o-", color=col, capsize=2, label=f"simulations, κ < {th:g}")
            ax.axhline(res["a2744"][f"{th:g}"]["p_sidm"], color=col, ls="--", lw=1.5,
                       label=f"A2744, Lenstool κ < {th:g}: {res['a2744'][f'{th:g}']['p_sidm']:.3f}")
        ax.set_xscale("log")
        ax.set_xticks([0.004, 0.01, 0.05, 0.1, 0.2, 0.3, 1], ["CDM", "0.01", "0.05", "0.1", "0.2", "0.3", "1"])
        ax.minorticks_off()
        ax.set_xlabel("σ/m [cm²/g]")
        ax.set_title(f"{dom.upper()}: mass-matched, strong-lensing cut at evaluation only"
                     + (" (simulations cut on A2744's Lenstool κ)" if res.get("footprint") else ""), fontsize=9)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("ensemble P(SIDM)")
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=150)
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
