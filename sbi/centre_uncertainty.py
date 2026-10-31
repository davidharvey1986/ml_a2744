#!/usr/bin/env python
'''
Systematic error on A2744's P(SIDM) from the uncertainty in its stamp centre.

The stamp is centred on the projected NFW that best fits the main peak's tangential shear
(add_shear_to_data.A2744_STAMP_CENTRE). The chi^2 of that fit over trial centres
(sbi/results/catalogue_comparison/nfw_centre_chi2.npz: a grid of offsets from Lenstool O2
in stamp pixels, mass and concentration refitted at each) gives a likelihood
exp(-delta chi^2 / 2) for the centre. Centres are drawn from it, A2744's data map is rebuilt
about each (current SHEAR_CUTS, strong-lensing kappa < 0.9 dropped, as make_obs_maps) and
the ensemble P(SIDM) is evaluated. The spread over draws is the centring systematic.

    astro-env/bin/python sbi/centre_uncertainty.py --model tweakedalign_k0p9_ctr_ba_best --include-str spin2
'''
import argparse
import json
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
from add_shear_to_data import (  # noqa: E402
    A2744_LENSTOOL_O2, SHEAR_CUTS, galaxy_distortion, get_obs_data, normalise_map, ra_dec_to_simulation_image_pos,
)


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="tweakedalign_k0p9_ctr_ba_best")
    p.add_argument("--include-str", default="spin2")
    p.add_argument("--nmodels", type=int, default=None)
    p.add_argument("--ndraws", type=int, default=100)
    p.add_argument("--chi2", default=str(REPO / "sbi" / "results" / "catalogue_comparison" / "nfw_centre_chi2.npz"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(REPO / "sbi" / "results" / "centre_uncertainty"))
    return p.parse_args()


def main():
    opts = parse()
    out = Path(opts.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = ms.get_device()
    args = ms.setup_args("concat")
    models = ms.load_models(SimpleNamespace(filter="concat", model=opts.model, include_str=opts.include_str,
                                            nmodels=opts.nmodels), args, device)
    print(f"{len(models)} models: {opts.model} containing '{opts.include_str}'")

    cat = get_obs_data("concat", data_dir="../data/100/a2744/", cuts=SHEAR_CUTS, drop_strong_lensing=True)
    x, y = ra_dec_to_simulation_image_pos(cat)                  # stamp pixels about A2744_STAMP_CENTRE
    c1, c2 = galaxy_distortion(cat)
    ox, oy = (float(v[0]) for v in ra_dec_to_simulation_image_pos(
        {"x": np.array([A2744_LENSTOOL_O2[0]]), "y": np.array([A2744_LENSTOOL_O2[1]])}))

    z = np.load(opts.chi2)
    grid = z["grid"]                                            # offsets from O2 [stamp px]
    gx, gy = np.meshgrid(grid, grid)
    dchi2 = z["chi2map"] - np.nanmin(z["chi2map"])
    w = np.where(np.isfinite(dchi2), np.exp(-0.5 * dchi2), 0.).ravel()
    rng = np.random.default_rng(opts.seed)
    pick = rng.choice(w.size, opts.ndraws, p=w / w.sum())
    # centre offsets relative to the current stamp centre: O2 sits at (ox, oy) in the current frame
    cx, cy = gx.ravel()[pick] + ox, gy.ravel()[pick] + oy

    def image(dx, dy):
        raw = np.array(bin2d(x - dx, y - dy, v=(c1, c2), npix=100, extent=[-50, 50, -50, 50]))
        return np.array(normalise_map(*raw))

    imgs = torch.tensor(np.array([image(0., 0.), image(ox, oy)] + [image(a, b) for a, b in zip(cx, cy)]),
                        dtype=torch.float32)
    p = sens.model_outputs(models, imgs, device, 32)            # (models, maps)
    ens = p.mean(0)
    draws = ens[2:]
    res = {
        "model": opts.model, "include_str": opts.include_str, "n_models": len(models), "n_draws": opts.ndraws,
        "p_sidm_adopted_centre": [float(ens[0]), float(p[:, 0].std())],
        "p_sidm_lenstool_o2": [float(ens[1]), float(p[:, 1].std())],
        "p_sidm_over_centre_draws": {"mean": float(draws.mean()), "std": float(draws.std()),
                                     "percentiles_16_50_84": np.percentile(draws, [16, 50, 84]).round(4).tolist()},
        "centre_draw_offsets_kpc_rms": float(20 * np.sqrt(np.mean(cx ** 2 + cy ** 2))),
        # each draw: centre offset from the adopted centre [stamp px] and ensemble P(SIDM)
        "draws": [{"dx_px": float(a), "dy_px": float(b), "p_sidm": float(q)} for a, b, q in zip(cx, cy, draws)],
    }
    print(f"A2744 P(SIDM): adopted centre {ens[0]:.3f} | Lenstool O2 {ens[1]:.3f} | over {opts.ndraws} centre draws "
          f"{draws.mean():.3f} ± {draws.std():.3f} (16-84%: {res['p_sidm_over_centre_draws']['percentiles_16_50_84'][0]:.3f}"
          f"-{res['p_sidm_over_centre_draws']['percentiles_16_50_84'][2]:.3f}); draws rms {res['centre_draw_offsets_kpc_rms']:.0f} kpc from the adopted centre")
    json.dump(res, open(out / f"{opts.model}_{opts.include_str}.json", "w"), indent=1)

    plt = sens.style()
    fig, ax = plt.subplots(figsize=(5.5, 4.3), constrained_layout=True)
    ax.hist(draws, bins=20, color="#0072B2", alpha=0.7, label=f"{opts.ndraws} centres drawn from the NFW-fit likelihood")
    ax.axvline(ens[0], color="k", lw=2, label=f"adopted centre: {ens[0]:.3f}")
    ax.axvline(ens[1], color="#D55E00", lw=2, ls="--", label=f"Lenstool O2: {ens[1]:.3f}")
    ax.set_xlabel("A2744 ensemble P(SIDM)")
    ax.set_ylabel("centre draws")
    ax.set_title(f"centring systematic, {opts.model} ({opts.include_str}, {len(models)} models)", fontsize=9)
    ax.legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{opts.model}_{opts.include_str}.{ext}", dpi=150)


if __name__ == "__main__":
    main()
